"""Run the agent (or the no-tools baseline) over the eval set, score it, write results/<name>.json."""
import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
from datetime import date
from pathlib import Path

from issuebot import agent
from issuebot.judge import JUDGE_MODEL, judge
from issuebot.tools import LABELS, SUBMIT, checkout, clone


def load(path: str, split: str = "all", limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if split == "all" or r["split"] == split]
    return rows[:limit] if limit else rows


def run_case(row: dict, src: Path, mode: str, model: str, client, do_judge: bool = True,
             threshold: float = agent.ROUTE_THRESHOLD) -> dict:
    gold = row.get("label_override") or row["gold_label"]
    case = {"number": row["number"], "created_at": row["created_at"], "gold": gold,
            "gold_dup": row.get("gold_duplicate_of"), "pred": None, "pred_dup": None, "confidence": 0.0,
            "score": None, "wrong": None, "cost": 0.0, "judge_cost": 0.0, "latency_s": None, "steps": 0, "error": None}
    try:
        ctx = {"repo": row["repo"], "dir": checkout(src, row["sha"]),
               "number": row["number"], "created_at": row["created_at"]}
        issue = {k: row[k] for k in ("number", "title", "body", "created_at")}  # the agent sees nothing else
        if mode == "baseline":  # same prompt and output, no retrieval tools: a clean ablation on one code path
            rec = agent.run(issue, ctx, model, tools=[SUBMIT], max_steps=2, client=client)
        elif mode == "routed":  # Haiku triage, Sonnet draft only when needed; --model is ignored
            rec = agent.route(issue, ctx, threshold, client=client)
        else:
            rec = agent.run(issue, ctx, model, client=client)
        case.update(pred=rec["label"], pred_dup=rec["duplicate_of"], confidence=rec["confidence"], cost=rec["cost"],
                    latency_s=rec["latency_s"], steps=rec["steps"], error=rec["error"],
                    route=rec.get("route"), capped=rec.get("capped"))
        if do_judge and rec["reply"]:
            j = judge(issue, row["maintainer_reply"], rec["reply"], client=client)
            case.update(score=j["score"], wrong=j["wrong"], judge_cost=j["cost"], judge_reason=j["reason"])
    except Exception as e:  # one broken case must not kill a 300-case run
        case["error"] = f"{type(e).__name__}: {e}"
    if do_judge and case["score"] is None:  # no reply scores 1, so a flakier system isn't judged on an easier subset
        case.update(score=1, wrong=False)
    return case


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
    delta = lambda ps: sum(y - x for x, y in ps) / len(ps)
    rng = random.Random(seed)
    ds = sorted(delta([rng.choice(pairs) for _ in pairs]) for _ in range(n))
    return delta(pairs), ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


COMPARE = {
    "label_accuracy": lambda c: float(c["pred"] == c["gold"]),
    "judge_mean": lambda c: c["score"],
    "dup_recall": lambda c: float(dup_hit(c)) if c["gold"] == "duplicate" else None,
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
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)

    rows = load(a.dataset, a.split, a.limit)
    short = a.model.removeprefix("claude-").split("-2025")[0]
    name = a.name or f"{a.mode}-{short}-{a.split}-{date.today()}"
    client = agent.make_client()
    srcs = {r: clone(r) for r in {row["repo"] for row in rows}}  # fetch once, not per case
    cases = []
    for row in rows:
        c = run_case(row, srcs[row["repo"]], a.mode, a.model, client, not a.no_judge, a.threshold)
        cases.append(c)
        print(f"#{c['number']} {c['gold']}->{c['pred']} dup={c['pred_dup'] or '-'} score={c['score'] or '-'} "
              f"${c['cost'] + c['judge_cost']:.3f} {c['latency_s'] or 0:.1f}s{' ERR ' + c['error'] if c['error'] else ''}",
              flush=True)
    m = metrics(cases)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        commit = None
    out = {"name": name, "mode": a.mode, "model": a.model, "judge_model": None if a.no_judge else JUDGE_MODEL,
           "split": a.split, "n": len(cases), "issuebot_commit": commit,
           "dataset_sha256": hashlib.sha256(Path(a.dataset).read_bytes()).hexdigest(), "metrics": m, "cases": cases}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in m.items() if not isinstance(v, dict)}, indent=1))
    print(f"wrote results/{name}.json")


if __name__ == "__main__":
    main()
