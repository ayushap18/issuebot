"""Run the agent (or the no-tools baseline) over the eval set, score it, write results/<name>.json."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
from datetime import date
from pathlib import Path

from issuebot import agent, tools
from issuebot.judge import CAUSES, JUDGE_MODEL, judge, judge_backend, tag_failure
from issuebot.tools import LABELS, SUBMIT, checkout, clone


def load(path: str, split: str = "all", limit: int | None = None, stratify: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if split == "all" or r["split"] == split]
    if stratify and limit:  # seeded shuffle per label (file is time-sorted), then round-robin: balanced until a label runs out
        groups = {}
        for r in rows:
            groups.setdefault(r.get("label_override") or r["gold_label"], []).append(r)
        for g in groups.values():
            random.Random(0).shuffle(g)
        rows = [r for tier in itertools.zip_longest(*(groups[k] for k in sorted(groups))) for r in tier if r]
    return rows[:limit] if limit else rows


def run_case(row: dict, src: Path, mode: str, model: str, client, do_judge: bool = True,
             threshold: float = agent.ROUTE_THRESHOLD, search: dict | None = None,
             judge_client=None, judge2_client=None) -> dict:
    gold = row.get("label_override") or row["gold_label"]
    case = {"number": row["number"], "created_at": row["created_at"], "gold": gold,
            "gold_dup": row.get("gold_duplicate_of"), "pred": None, "pred_dup": None, "confidence": 0.0,
            "score": None, "wrong": None, "cost": 0.0, "judge_cost": 0.0, "latency_s": None, "steps": 0, "error": None, "reply": None}
    try:
        ctx = {"repo": row["repo"], "dir": checkout(src, row["sha"]),
               "number": row["number"], "created_at": row["created_at"], **(search or {})}
        issue = {k: row[k] for k in ("number", "title", "body", "created_at")}  # the agent sees nothing else
        if mode == "baseline":  # same prompt and output, no retrieval tools: a clean ablation on one code path
            rec = agent.run(issue, ctx, model, tools=[SUBMIT], max_steps=2, client=client)
        elif mode == "routed":  # Haiku triage, Sonnet draft only when needed; --model is ignored
            rec = agent.route(issue, ctx, threshold, client=client)
        else:
            rec = agent.run(issue, ctx, model, client=client)
        case.update(pred=rec["label"], pred_dup=rec["duplicate_of"], confidence=rec["confidence"], cost=rec["cost"],
                    latency_s=rec["latency_s"], steps=rec["steps"], error=rec["error"], reply=rec["reply"],
                    route=rec.get("route"), capped=rec.get("capped"), backend=rec.get("backend"),
                    tool_calls=[{k: t[k] for k in ("name", "input", "hits") if k in t} for t in rec.get("tool_calls", [])])
        if do_judge and rec["reply"]:
            j = judge(issue, row["maintainer_reply"], rec["reply"], client=judge_client or client)
            case.update(score=j["score"], wrong=j["wrong"], judge_cost=j["cost"], judge_reason=j["reason"])
        if do_judge and judge2_client and rec["reply"]:
            try:  # a second-judge failure must not lose the first judge's score
                j = judge(issue, row["maintainer_reply"], rec["reply"], client=judge2_client)
                case.update(judge2_score=j["score"], judge2_wrong=j["wrong"], judge2_reason=j["reason"])
                case["judge_cost"] += j["cost"]
            except Exception as e:
                case["judge2_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # one broken case must not kill a 300-case run
        case["error"] = f"{type(e).__name__}: {e}"
    if do_judge and case["score"] is None and not case["reply"]:  # no reply scores 1, so a flakier system isn't judged on an easier subset; judge errors stay None
        case.update(score=1, wrong=False)
    return case


def retriever(repo: str, search: str, issues: list | None = None) -> dict:
    """ctx keys for search_issues. local/fts read the cached corpus (REST list pages; reruns fetch only updates)."""
    if search == "github":
        return {"search": "github"}
    if issues is None:
        from issuebot.backtest import corpus  # backtest imports this module
        issues = corpus(repo)
    fts = tools.index(repo, issues) if search == "fts" else None
    return {"search": "fts" if fts else "local", "corpus": issues, "fts": fts}  # no FTS5: local, recorded as such


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[max(0, math.ceil(q * len(s)) - 1)] if s else 0.0


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _div(a, b) -> float:
    return a / b if b else 0.0


def dup_hit(c: dict) -> bool:
    return c["pred"] == "duplicate" and c["pred_dup"] is not None and c["pred_dup"] == c["gold_dup"]


def dup_seen(c: dict) -> bool:
    """Retrieval-only: the gold duplicate was in the top 5 of any search the agent ran (needs `hits` in tool_calls)."""
    return any(c["gold_dup"] in t.get("hits", ()) for t in c.get("tool_calls") or ())


def metrics(cases: list[dict]) -> dict:
    n = len(cases)
    if not n:
        return {"n": 0}
    g = [c["gold"] for c in cases]
    p = [c["pred"] for c in cases]  # errored cases have pred None, so they count as wrong
    f1s = []
    for lab in [l for l in LABELS if l in g or l in p]:  # absent labels would cap macro F1 below 1
        tp = sum(a == b == lab for a, b in zip(g, p))
        prec, rec = _div(tp, p.count(lab)), _div(tp, g.count(lab))
        f1s.append(_div(2 * prec * rec, prec + rec))
    judged = [c for c in cases if c["score"] is not None]
    hi = [c for c in cases if c["confidence"] >= 0.8]
    lat = [c["latency_s"] for c in cases if c["latency_s"] is not None]
    agent_cost = sum(c["cost"] for c in cases)
    judge_cost = sum(c["judge_cost"] for c in cases)
    quarters = {}
    for c in cases:
        y, m = c["created_at"][:4], int(c["created_at"][5:7])
        quarters.setdefault(f"{y}Q{(m - 1) // 3 + 1}", []).append(c)
    return {
        "n": n,
        "label_accuracy": _div(sum(a == b for a, b in zip(g, p)), n),
        "majority_floor": max(g.count(lab) for lab in set(g)) / n,
        "macro_f1": sum(f1s) / len(f1s),
        "label_dist": {lab: g.count(lab) for lab in LABELS},
        "confusion": {a: {b: sum(x == a and y == b for x, y in zip(g, p)) for b in (*LABELS, None)} for a in LABELS},
        "dup_precision": _div(sum(map(dup_hit, cases)), p.count("duplicate")),
        "dup_recall": _div(sum(map(dup_hit, cases)), g.count("duplicate")),
        "dup_recall_at5": _div(sum(map(dup_seen, dups := [c for c in cases if c["gold"] == "duplicate" and c["gold_dup"]])),
                               len(dups)),
        "judged": len(judged),
        "judge_mean": _mean(c["score"] for c in judged),
        "judge_ge4": _mean(c["score"] >= 4 for c in judged),
        "wrong_rate": _mean(c["wrong"] for c in judged),
        "confidently_wrong": _mean(bool(c["wrong"]) and c["confidence"] >= 0.7 for c in judged),
        "acc_at_0.8": _div(sum(c["pred"] == c["gold"] for c in hi), len(hi)),
        "coverage_at_0.8": len(hi) / n,
        "cost_total": agent_cost + judge_cost,
        "cost_per_issue": (agent_cost + judge_cost) / n,
        "agent_cost_per_issue": agent_cost / n,
        "latency_p50": statistics.median(lat) if lat else 0.0,
        "latency_p95": pct(lat, 0.95),
        "avg_steps": sum(c["steps"] for c in cases) / n,
        "error_rate": sum(c["error"] is not None for c in cases) / n,
        "by_quarter": {q: {"n": len(cs), "label_accuracy": _div(sum(c["pred"] == c["gold"] for c in cs), len(cs)),
                           "judge_mean": _mean(c["score"] for c in cs)} for q, cs in sorted(quarters.items())},
    }


def bootstrap(a: list[dict], b: list[dict], key, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap of mean(key(b)) - mean(key(a)) over cases matched by number. key -> float or None (skip)."""
    bm = {c["number"]: c for c in b}
    pairs = [(key(c), key(bm[c["number"]])) for c in a if c["number"] in bm]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if not pairs:
        return 0.0, 0.0, 0.0
    # key may return (num, den) for a ratio metric like precision; a plain float is (x, 1), i.e. a mean
    rate = lambda xs: _div(*map(sum, zip(*(x if isinstance(x, tuple) else (x, 1) for x in xs))))
    delta = lambda ps: rate(y for _, y in ps) - rate(x for x, _ in ps)
    rng = random.Random(seed)
    ds = sorted(delta([rng.choice(pairs) for _ in pairs]) for _ in range(n))
    return delta(pairs), ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


COMPARE = {
    "label_accuracy": lambda c: float(c["pred"] == c["gold"]),
    "judge_mean": lambda c: c["score"],
    "dup_recall": lambda c: float(dup_hit(c)) if c["gold"] == "duplicate" else None,
    "dup_recall_at5": lambda c: float(dup_seen(c)) if c["gold"] == "duplicate" and c["gold_dup"] else None,
    "confidently_wrong": lambda c: None if c["wrong"] is None else float(c["wrong"] and c["confidence"] >= 0.7),
    "cost": lambda c: c["cost"] + c["judge_cost"],
}


def compare(pa: str, pb: str) -> None:
    a, b = (json.loads(Path(x).read_text()) for x in (pa, pb))
    print(f"A = {a['name']}  B = {b['name']}")
    for k, f in COMPARE.items():
        d, lo, hi = bootstrap(a["cases"], b["cases"], f)
        sig = "*" if lo > 0 or hi < 0 else " "
        print(f"{k:18} {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] {sig}")


# metric -> (paired key, floor). The gate fails only when the 95% CI is entirely below 0 AND the drop exceeds the floor.
GATE = {
    "label_accuracy": (COMPARE["label_accuracy"], 0.03),
    "dup_precision": (lambda c: (float(dup_hit(c)), float(c["pred"] == "duplicate")), 0.05),
    "judge_mean": (COMPARE["judge_mean"], 0.2),
}


def gate(pa: str, pb: str) -> bool:
    """Compare results pb to baseline pa on their shared cases; print a markdown table; True = pass."""
    a, b = (json.loads(Path(x).read_text())["cases"] for x in (pa, pb))
    shared = {c["number"] for c in a} & {c["number"] for c in b}
    if not shared:
        print("gate: no shared cases with baseline")
        return False
    ma, mb = (metrics([c for c in cs if c["number"] in shared]) for cs in (a, b))
    lines, ok = [f"Eval gate: {pb} vs baseline {pa} ({len(shared)} shared cases)", "",
                 "| metric | baseline | new | delta | CI | verdict |", "|---|---|---|---|---|---|"], True
    for k, (f, floor) in GATE.items():
        if ma.get(k) is None or mb.get(k) is None:  # e.g. judge_mean on a --no-judge run
            lines.append(f"| {k} | {ma.get(k)} | {mb.get(k)} | | | n/a |")
            continue
        d, lo, hi = bootstrap(a, b, f)
        bad = hi < 0 and d < -floor
        ok &= not bad
        lines.append(f"| {k} | {ma[k]:.3f} | {mb[k]:.3f} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                     f"{'FAIL' if bad else 'pass'} |")
    lines.append(f"\n**{'PASS' if ok else 'FAIL'}** (fails only past the CI and floors: label -3pts, dup precision -5pts, judge -0.2)")
    out = "\n".join(lines)
    print(out)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(out + "\n")
    return ok


def kappa(a: list, b: list, weighted: bool = False) -> float:
    """Cohen's kappa; weighted=True uses quadratic weights on numeric scores. 1 - observed/expected disagreement."""
    w = (lambda x, y: (x - y) ** 2) if weighted else (lambda x, y: float(x != y))
    obs = sum(map(w, a, b)) / len(a)
    exp = sum(w(x, y) for x in a for y in b) / len(a) ** 2  # marginals independent
    return 1 - obs / exp if exp else 1.0  # exp 0: both raters gave one identical value throughout


GRADE_COLS = ["number", "title", "body", "maintainer_reply", "agent_reply", "human_score", "human_wrong"]


def export_grading(results: str, dataset: str, n: int = 50, out=sys.stdout, seed: int = 0) -> None:
    """Blind grading sheet: a seeded sample of replied cases, without the judge's score."""
    rows = {r["number"]: r for r in load(dataset)}
    pool = sorted((c for c in json.loads(Path(results).read_text())["cases"] if c.get("reply") and c.get("judge_reason") is not None), key=lambda c: c["number"])
    w = csv.DictWriter(out, GRADE_COLS)
    w.writeheader()
    for c in sorted(random.Random(seed).sample(pool, min(n, len(pool))), key=lambda c: c["number"]):
        r = rows[c["number"]]
        w.writerow({"number": c["number"], "title": r["title"], "body": (r.get("body") or "")[:1500],
                    "maintainer_reply": r["maintainer_reply"], "agent_reply": c["reply"], "human_score": "", "human_wrong": ""})


def calibrate(grading: str, results: str) -> bool:
    """Judge vs human agreement on the hand-graded sheet; True = weighted kappa >= 0.6."""
    cases = {c["number"]: c for c in json.loads(Path(results).read_text())["cases"]}
    with open(grading, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["human_score"].strip()]
    if not rows:
        raise SystemExit("no human_score filled in")
    h = [int(r["human_score"]) for r in rows]
    j = [cases[int(r["number"])]["score"] for r in rows]
    k = kappa(h, j, weighted=True)
    print(f"n={len(h)}  exact={_mean(x == y for x, y in zip(h, j)):.3f}  "
          f"within1={_mean(abs(x - y) <= 1 for x, y in zip(h, j)):.3f}  weighted_kappa={k:.3f}")
    wr = [r for r in rows if r.get("human_wrong", "").strip()]
    if wr:
        hw = [r["human_wrong"].strip().lower() in ("1", "true", "yes", "y") for r in wr]
        print(f"wrong_kappa={kappa(hw, [bool(cases[int(r['number'])]['wrong']) for r in wr]):.3f} (n={len(wr)})")
    print(f"{'PASS' if k >= 0.6 else 'FAIL'} (weighted kappa >= 0.6)")
    return k >= 0.6


def agreement(cases: list[dict]) -> dict:
    """Judge vs judge2 on cases both scored: cross-model agreement on the same drafts."""
    both = [c for c in cases if c["score"] is not None and c.get("judge2_score") is not None]
    if not both:
        return {"n": 0}
    a, b = [c["score"] for c in both], [c["judge2_score"] for c in both]
    return {"n": len(both), "exact": _mean(x == y for x, y in zip(a, b)),
            "within1": _mean(abs(x - y) <= 1 for x, y in zip(a, b)), "weighted_kappa": kappa(a, b, weighted=True),
            "wrong_kappa": kappa([bool(c["wrong"]) for c in both], [bool(c["judge2_wrong"]) for c in both]),
            "judge2_mean": _mean(b)}


def failed(c: dict) -> bool:
    return (c["pred"] != c["gold"] or (c["gold"] == "duplicate" and not dup_hit(c))
            or (c["score"] is not None and c["score"] <= 2) or bool(c["wrong"]))


def tag_failures(results: str, dataset: str, client=None) -> bool:
    """Tag each failed case's primary cause in place; True = Stage 3 trigger (retrieval_miss >= 30%)."""
    res = json.loads(Path(results).read_text())
    rows = {r["number"]: r for r in load(dataset)}
    client = client or agent.make_client(judge_backend())
    fails = [c for c in res["cases"] if failed(c)]
    errors = 0
    for c in fails:
        r = rows[c["number"]]
        try:
            t = tag_failure(r, r["maintainer_reply"], c, client=client)
            c.update(failure_tag=t["cause"], failure_reason=t["reason"])
        except Exception as e:  # untagged counts as other, so shares still sum over all failures
            c.update(failure_tag="other", failure_reason=f"tag error: {type(e).__name__}: {e}")
            errors += 1
    if fails and errors == len(fails):  # nothing tagged (no key, replay misses): don't save bogus tags
        raise SystemExit(f"{errors} tag errors, all failures untagged; results not written")
    counts = {k: sum(c["failure_tag"] == k for c in fails) for k in CAUSES}
    res["failure_tags"] = counts
    Path(results).write_text(json.dumps(res, indent=1))
    print(f"{len(fails)} failures of {len(res['cases'])} cases" + (f", {errors} tag errors" if errors else ""))
    for k, v in counts.items():
        print(f"{k:16} {v:4}  {_div(v, len(fails)):.1%}")
    met = bool(fails) and counts["retrieval_miss"] / len(fails) >= 0.3
    print("STAGE 3 TRIGGER MET" if met else "stage 3 trigger not met (retrieval_miss < 30%)")
    return met


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="agent", choices=["agent", "baseline", "routed"])
    ap.add_argument("--threshold", type=float, default=agent.ROUTE_THRESHOLD, help="routed mode confidence gate")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--model", default=agent.REPLY_MODEL)
    ap.add_argument("--name")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--dataset", default="eval/dataset.jsonl")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--stratify", action="store_true", help="balance --limit across gold labels")
    ap.add_argument("--search", default="github", choices=tools.SEARCH,
                    help="search_issues backend; local/fts fetch and cache each repo's issue corpus first")
    ap.add_argument("--gate", nargs="+", metavar="FILE",
                    help="BASELINE [NEW]: gate NEW (or this run's results) against BASELINE; exit 1 on regression")
    ap.add_argument("--export-grading", metavar="RESULTS", help="write a blind hand-grading CSV to stdout")
    ap.add_argument("--n", type=int, default=50, help="--export-grading sample size")
    ap.add_argument("--calibrate", nargs=2, metavar=("CSV", "RESULTS"), help="judge vs hand grades; exit 1 if kappa < 0.6")
    ap.add_argument("--judge2", choices=["agy", "codex", "claude-cli"],
                    help="second judge on another model; scores stored as judge2_*, agreement reported")
    ap.add_argument("--tag-failures", metavar="RESULTS", help="tag each failure's cause in place (Stage 3 trigger)")
    a = ap.parse_args()
    if a.tag_failures:
        tag_failures(a.tag_failures, a.dataset)
        return
    if a.export_grading:
        return export_grading(a.export_grading, a.dataset, a.n)
    if a.calibrate:
        raise SystemExit(0 if calibrate(*a.calibrate) else 1)
    if a.compare:
        return compare(*a.compare)
    if a.gate and len(a.gate) > 2:
        ap.error("--gate takes BASELINE [NEW]")
    if a.gate and len(a.gate) == 2:
        raise SystemExit(0 if gate(*a.gate) else 1)

    rows = load(a.dataset, a.split, a.limit, a.stratify)
    short = a.model.removeprefix("claude-").split("-2025")[0]
    name = a.name or f"{a.mode}-{short}-{a.split}-{date.today()}"
    client, jb = agent.make_client(), judge_backend()
    jclient, j2client = agent.make_client(jb), a.judge2 and agent.make_client(a.judge2)
    srcs = {r: clone(r) for r in {row["repo"] for row in rows}}  # fetch once, not per case
    search = {r: retriever(r, a.search) for r in srcs}
    cases = []
    for row in rows:
        c = run_case(row, srcs[row["repo"]], a.mode, a.model, client, not a.no_judge, a.threshold,
                     search=search[row["repo"]], judge_client=jclient, judge2_client=j2client)
        cases.append(c)
        print(f"#{c['number']} {c['gold']}->{c['pred']} dup={c['pred_dup'] or '-'} score={c['score'] or '-'} "
              f"${c['cost'] + c['judge_cost']:.3f} {c['latency_s'] or 0:.1f}s{' ERR ' + c['error'] if c['error'] else ''}",
              flush=True)
    m = metrics(cases)
    if a.judge2:
        m["judge_agreement"] = agreement(cases)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        commit = None
    out = {"name": name, "mode": a.mode, "model": a.model, "judge_model": None if a.no_judge else JUDGE_MODEL,
           "backend": getattr(client, "backend", "api"), "judge_backend": None if a.no_judge else jb, "judge2_backend": a.judge2,
           "split": a.split, "search": "local" if a.search == "fts" and not tools.fts5() else a.search,  # effective
           "n": len(cases), "issuebot_commit": commit,
           "dataset_sha256": hashlib.sha256(Path(a.dataset).read_bytes()).hexdigest(), "metrics": m, "cases": cases}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in m.items() if not isinstance(v, dict)}, indent=1))
    if a.judge2:
        print(f"judge ({jb}) vs judge2 ({a.judge2}): {json.dumps(m['judge_agreement'])}")
    print(f"wrote results/{name}.json")
    if os.environ.get("ISSUEBOT_REPLAY") == "replay" and any("replay miss" in (c["error"] or "") for c in cases):
        print("::notice::eval-gate skipped: replay cache misses (fork PR without API key)")
        raise SystemExit(0)
    if a.gate:
        raise SystemExit(0 if gate(a.gate[0], f"results/{name}.json") else 1)


if __name__ == "__main__":
    main()
