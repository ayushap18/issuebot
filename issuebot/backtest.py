"""Backtest one repo: its last N closed issues with a maintainer reply, leakage-safe, scored like the eval."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from issuebot import agent, build_eval, run_eval
from issuebot.judge import JUDGE_MODEL, judge_backend
from issuebot.tools import clone, pages

KEYS = ("number", "title", "body", "created_at", "html_url", "state", "state_reason", "author_association")


def slim(i: dict) -> dict:
    return {**{k: i.get(k) for k in KEYS}, "labels": [{"name": l["name"]} for l in i["labels"]],
            "user": {k: i["user"][k] for k in ("login", "type")}}


def corpus(repo: str, cache: Path = Path("cache")) -> list[dict]:
    """Every issue (no PRs) via the REST list, cached as JSON; reruns fetch only issues updated since the last fetch."""
    f = cache / f"corpus-{repo.replace('/', '__')}.json"
    old = json.loads(f.read_text()) if f.exists() else {"fetched_at": None, "issues": []}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"state": "all", **({"since": old["fetched_at"]} if old["fetched_at"] else {})}
    new = [slim(i) for i in pages(f"/repos/{repo}/issues", params) if "pull_request" not in i]
    issues = list(({i["number"]: i for i in old["issues"]} | {i["number"]: i for i in new}).values())
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"fetched_at": now, "issues": issues}))
    return issues


def worst(cases: list[dict], k: int = 10) -> list[dict]:
    return sorted((c for c in cases if c.get("reply") and c["score"] is not None),
                  key=lambda c: (c["score"], not c["wrong"], c["number"]))[:k]


def scorecard(res: dict) -> str:
    m, cs = res["metrics"], res["cases"]
    if not m["n"]:
        return f"## issuebot backtest: {res['repo']}\n\nNo scorable issues found."
    f = lambda x, s="{:.2f}": "-" if x is None else s.format(x)
    lines = [f"## issuebot backtest: {res['repo']} ({res['mode']}, {m['n']} issues)", "",
             "| metric | value |", "|---|---|",
             f"| label accuracy | {m['label_accuracy']:.0%} |",
             f"| duplicates found | {sum(map(run_eval.dup_hit, cs))}/{sum(c['gold'] == 'duplicate' for c in cs)} |",
             f"| judge mean (1-5) | {f(m['judge_mean'])} |",
             f"| $ total | ${m['cost_total']:.2f} |",
             f"| $ / issue | ${m['cost_per_issue']:.4f} |",
             f"| latency p50 / p95 | {m['latency_p50']:.1f}s / {m['latency_p95']:.1f}s |",
             f"| errors | {m['error_rate']:.0%} |", "",
             "Duplicate search used the local retriever (every-term match on the cached corpus, GitHub search only on a miss).", "",
             "### 10 worst replies", ""]
    for c in worst(cs):
        why = " ".join((c.get("judge_reason") or "").split())[:200]
        lines.append(f"- [#{c['number']} {c.get('title', '')}]({c.get('url', '')}) score {c['score']}"
                     f"{' (wrong)' if c['wrong'] else ''}, {c['gold']} -> {c['pred']}: {why}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="issuebot.backtest")
    ap.add_argument("repo", help="owner/name")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--mode", default="agent", choices=["agent", "routed"])
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    a.n = min(a.n, 500)
    out = Path(a.out or f"results/backtest-{a.repo.replace('/', '__')}.json")

    issues = corpus(a.repo)
    closed = sorted((i for i in issues if i["state"] == "closed"), key=lambda i: i["created_at"], reverse=True)
    rows = build_eval.build(a.repo, a.n, issues=closed)
    client, jclient, src, cases = agent.make_client(), agent.make_client(judge_backend()), clone(a.repo), []
    for row in rows:  # checkout at the issue's SHA, search only issues created before it (run_case + tools enforce both)
        c = run_eval.run_case(row, src, a.mode, agent.REPLY_MODEL, client, corpus=issues, judge_client=jclient)
        cases.append({**c, "url": row["url"], "title": row["title"]})
        print(f"#{c['number']} {c['gold']}->{c['pred']} score={c['score'] or '-'} ${c['cost'] + c['judge_cost']:.3f}",
              flush=True)
    res = {"name": out.stem, "repo": a.repo, "mode": a.mode, "model": agent.REPLY_MODEL, "judge_model": JUDGE_MODEL,
           "judge_backend": judge_backend(),
           "n": len(cases), "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "metrics": run_eval.metrics(cases), "cases": cases}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    md = scorecard(res)
    print(md + f"\n\nwrote {out}")
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as fh:
            fh.write(md + "\n")


if __name__ == "__main__":
    main()
