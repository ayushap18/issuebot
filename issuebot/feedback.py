"""Weekly feedback loop: read adopters' public issue timelines, score the bot's predictions, collect misses, flag drift."""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from issuebot import build_eval as be
from issuebot.agent import read_marker
from issuebot.tools import clone, sha_at

PREFIX = "bot:"  # default label_prefix; mapped (verbatim) labels are found via the marker comment's author instead
KEEP = timedelta(days=7)


def pages(path: str, params: dict | None = None, stop=lambda x: False) -> list:
    out, page = [], 1
    while True:
        batch = be.gh(path, params={**(params or {}), "per_page": 100, "page": page})
        out += [x for x in batch if not stop(x)]
        if len(batch) < 100 or any(map(stop, batch)):
            return out
        page += 1


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def label_kept(tl: list[dict], name: str | None, now: datetime) -> bool | None:
    """True = still on 7 days after first applied, False = removed within 7 days, None = no such label / too soon."""
    on = next((ts(e["created_at"]) for e in tl if e["event"] == "labeled" and e["label"]["name"] == name), None)
    if not on:
        return None
    if any(e["event"] == "unlabeled" and e["label"]["name"] == name and on <= ts(e["created_at"]) <= on + KEEP for e in tl):
        return False
    return True if now - on >= KEEP else None


def judge_issue(repo: str, i: dict, now: datetime) -> dict | None:
    """Prediction + outcome for one issue, or None if the bot never touched it."""
    tl = pages(f"/repos/{repo}/issues/{i['number']}/timeline")
    comments = [e for e in tl if e["event"] == "commented"]
    mc = next((c for c in comments if read_marker(c.get("body") or "")), None)
    bot = (mc or {}).get("user", {}).get("login")
    applied = next((e["label"]["name"] for e in tl if e["event"] == "labeled" and
                    (e["label"]["name"].startswith(PREFIX) or (bot and e.get("actor", {}).get("login") == bot))), None)
    if mc:
        pred = read_marker(mc["body"])
    elif applied:
        pred = {"label": applied.removeprefix(PREFIX), "duplicate_of": None}
    else:
        return None
    g = be.gold(i, comments)
    confirmed = bool(g) and g[0] == "duplicate" and pred["label"] == "duplicate" and g[1] == pred.get("duplicate_of")
    contradicted = bool(g) and g[0] == "duplicate" and not confirmed
    kept = label_kept(tl, applied, now)
    agree = True if confirmed else False if contradicted or kept is False else kept
    return {"pred": pred, "applied": applied, "gold": g, "agree": agree, "comments": comments}


def candidate(repo: str, i: dict, r: dict) -> dict | None:
    """A miss as a build_eval row (sha=None: --promote computes it from a clone). None when it can't be scored."""
    g, comments = r["gold"], r["comments"]
    if not g:
        return None
    m = (next((c for c in comments if be.is_maint(c) and be.DUP_RE.search(c.get("body") or "")), None)
         if g[0] == "duplicate" else be.maintainer_reply(comments))
    if not m or not (orig := be.original_text(repo, i, comments)):
        return None
    return {"repo": repo, "number": i["number"], "url": i["html_url"], "title": orig[0], "body": orig[1],
            "author": i["user"]["login"], "created_at": i["created_at"], "sha": None, "split": None,
            "gold_label": g[0], "gold_duplicate_of": g[1], "label_override": None,
            "maintainer_reply": m["body"], "maintainer": m["user"]["login"],
            "labels": [l["name"] for l in i["labels"]], "state_reason": i.get("state_reason")}


def read_rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def collect(repo: str, days: int, out: Path, baseline: float | None, now: datetime) -> tuple[list[str], str]:
    """Score one adopter; append misses to out/<owner>__<repo>.jsonl. Returns (DRIFT lines, markdown table row).
    The window is issues created `days` days before now-7d, so every bot label has had its 7 days."""
    end, start = now - KEEP, now - KEEP - timedelta(days=days)
    issues = pages(f"/repos/{repo}/issues", {"state": "all", "sort": "created", "direction": "desc"},
                   stop=lambda i: ts(i["created_at"]) < start)
    rs = [(i, r) for i in issues if "pull_request" not in i and ts(i["created_at"]) <= end
          if (r := judge_issue(repo, i, now))]
    decided = [(i, r) for i, r in rs if r["agree"] is not None]
    agree = sum(r["agree"] for _, r in decided) / len(decided) if decided else None
    f = out / f"{repo.replace('/', '__')}.jsonl"
    rows = read_rows(f)
    seen = {r["number"] for r in rows}
    new = [c for i, r in decided if not r["agree"] and i["number"] not in seen if (c := candidate(repo, i, r))]
    if new:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("".join(json.dumps(r) + "\n" for r in rows + new))
    labels = {l["name"] for l in pages(f"/repos/{repo}/labels")}
    drift = []
    if agree is not None and baseline is not None and agree < baseline - 0.10:
        drift.append(f"DRIFT {repo}: 7-day agreement {agree:.0%} is more than 10pts below offline baseline {baseline:.0%}")
    for name in sorted({r["applied"] for _, r in rs if r["applied"]} - labels):
        drift.append(f"DRIFT {repo}: predicted label {name!r} is not in the repo's current label set")
    fmt = lambda x: "-" if x is None else f"{x:.0%}"
    row = f"| {repo} | {len(rs)} | {len(decided)} | {fmt(agree)} | {fmt(baseline)} | {len(new)} | {'yes' if drift else 'no'} |"
    return drift, row


def promote(cand: Path, dataset: Path, min_n: int, force: bool) -> int:
    """Merge candidate rows into the dataset once >= min_n accumulate. New rows get split by date among themselves
    (oldest 2/3 dev) so existing dev/test assignments, and baselines scored on them, don't move."""
    files = sorted(cand.glob("*.jsonl"))
    rows = [r for f in files for r in read_rows(f)]
    if len(rows) < min_n and not force:
        print(f"{len(rows)} candidates, need {min_n} (or --force)")
        return 0
    data = read_rows(dataset)
    have = {(r["repo"], r["number"]) for r in data}
    new = []
    for r in sorted(rows, key=lambda r: r["created_at"]):
        if (r["repo"], r["number"]) not in have:
            have.add((r["repo"], r["number"]))
            new.append({**r, "sha": sha_at(clone(r["repo"]), r["created_at"])})
    for k, r in enumerate(new):
        r["split"] = "dev" if k < len(new) * 2 // 3 else "test"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text("".join(json.dumps(r) + "\n" for r in data + new))
    for f in files:
        f.unlink()
    print(f"promoted {len(new)} rows into {dataset} ({len(rows) - len(new)} already present)")
    return len(new)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="issuebot.feedback")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", default="eval/candidates")
    ap.add_argument("--adopters", default="adopters.txt")
    ap.add_argument("--baseline", default="eval/baseline.json")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--min", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dataset", default="eval/dataset.jsonl")
    a = ap.parse_args(argv)
    if a.promote:
        promote(Path(a.out), Path(a.dataset), a.min, a.force)
        return
    b = Path(a.baseline)
    baseline = json.loads(b.read_text())["metrics"]["label_accuracy"] if b.exists() else None
    repos = [l.split("#")[0].strip() for l in Path(a.adopters).read_text().splitlines()]
    now, drift, table = datetime.now(timezone.utc), [], []
    for repo in filter(None, repos):
        d, row = collect(repo, a.days, Path(a.out), baseline, now)
        drift += d
        table.append(row)
    print("\n".join(drift))
    md = "\n".join([f"## issuebot feedback ({a.days} days, ending 7 days ago)", "",
                    "| repo | bot-touched | decided | agreement | baseline | new candidates | drift |",
                    "|---|---|---|---|---|---|---|", *table, "", *[f"- {d}" for d in drift]])
    print(md)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as fh:
            fh.write(md + "\n")


if __name__ == "__main__":
    main()
