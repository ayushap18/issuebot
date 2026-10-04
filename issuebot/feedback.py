"""Weekly feedback loop: read adopters' public issue timelines, score the bot's predictions, collect misses, flag drift."""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from issuebot import build_eval as be
from issuebot import run_eval
from issuebot.agent import CONFIG, read_marker
from issuebot.tools import clone, sha_at

PREFIX = "bot:"  # default label_prefix; mapped (verbatim) labels are read from the marker's "applied" instead
KEEP = timedelta(days=7)


def pages(path: str, params: dict | None = None, stop=lambda x: False) -> list:
    out, page = [], 1
    while True:
        batch = be.gh(path, params={**(params or {}), "per_page": 100, "page": page})
        out += [x for x in batch if not stop(x)]
        if len(batch) < 100 or any(map(stop, batch)):
            return out
        page += 1


def label_kept(tl: list[dict], name: str | None, now: datetime) -> bool | None:
    """True = still on 7 days after first applied (or bot:X swapped for the repo's own X), False = removed within
    7 days, None = no such label / too soon."""
    at = lambda e: datetime.fromisoformat(e["created_at"])
    on = next((at(e) for e in tl if e["event"] == "labeled" and e["label"]["name"] == name), None)
    if not on:
        return None
    within = lambda e, event, n: e["event"] == event and e["label"]["name"] == n and on <= at(e) <= on + KEEP
    if name.startswith(PREFIX) and any(within(e, "labeled", name.removeprefix(PREFIX)) for e in tl):
        return True
    if any(within(e, "unlabeled", name) for e in tl):
        return False
    return True if now - on >= KEEP else None


def judge_issue(repo: str, i: dict, now: datetime) -> dict | None:
    """Prediction + outcome for one issue, or None if the bot never touched it."""
    tl = pages(f"/repos/{repo}/issues/{i['number']}/timeline")
    comments = [e for e in tl if e["event"] == "commented"]
    # Only a bot's marker counts: anyone can paste a marker-shaped comment.
    mc = next((c for c in comments if (c.get("user") or {}).get("type") == "Bot" and read_marker(c.get("body") or "")), None)
    if mc:
        pred = read_marker(mc["body"])
        applied = pred.get("applied")  # recorded by the bot, so other workflows' labels can't be mistaken for it
    elif applied := next((e["label"]["name"] for e in tl if e["event"] == "labeled" and e["label"]["name"].startswith(PREFIX)), None):
        pred = {"label": applied.removeprefix(PREFIX), "duplicate_of": None}
    else:
        return None
    g = be.gold(i, comments)
    dup = bool(g) and g[0] == "duplicate"  # label only, like offline label_accuracy (label mode has no duplicate_of)
    confirmed, contradicted = dup and pred["label"] == "duplicate", dup and pred["label"] != "duplicate"
    kept = label_kept(tl, applied, now)
    agree = True if confirmed else False if contradicted or kept is False else kept
    return {"pred": pred, "applied": applied, "gold": g, "agree": agree, "comments": comments}


def candidate(repo: str, i: dict, r: dict) -> dict | None:
    """A miss as a build_eval row (sha=None: --promote computes it), with build_eval's population filter. Gold is
    filled only for duplicates (from the maintainer's pointer): gold()'s label names are vitest's, so other misses
    get gold_label=None and wait for a hand label."""
    g = r["gold"]
    if (not g or i["state"] != "closed" or i["user"]["type"] == "Bot" or i["author_association"] in be.MAINTAINER
            or not (i.get("body") or "").strip()):
        return None
    return be.row(repo, i, r["comments"], g if g[0] == "duplicate" else (None, None))


def read_rows(p: Path) -> list[dict]:
    return run_eval.load(str(p)) if p.exists() else []


def collect(repo: str, days: int, out: Path, baseline: float | None, now: datetime,
            about: str = "") -> tuple[list[str], str]:
    """Score one adopter; append misses to out/<owner>__<repo>.jsonl. Returns (DRIFT lines, markdown table row).
    The window is issues created `days` days before now-7d, so every bot label has had its 7 days."""
    end, start = now - KEEP, now - KEEP - timedelta(days=days)
    issues = pages(f"/repos/{repo}/issues", {"state": "all", "sort": "created", "direction": "desc"},
                   stop=lambda i: datetime.fromisoformat(i["created_at"]) < start)
    rs = [(i, r) for i in issues if "pull_request" not in i and datetime.fromisoformat(i["created_at"]) <= end
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
        drift.append(f"DRIFT {repo}: 7-day agreement {agree:.0%} is more than 10pts below offline baseline {baseline:.0%}{about}")
    for name in sorted({r["applied"] for _, r in rs if r["applied"]} - labels):
        drift.append(f"DRIFT {repo}: predicted label {name!r} is not in the repo's current label set")
    fmt = lambda x: "-" if x is None else f"{x:.0%}"
    row = f"| {repo} | {len(rs)} | {len(decided)} | {fmt(agree)} | {fmt(baseline)} | {len(new)} | {'yes' if drift else 'no'} |"
    return drift, row


def promote(cand: Path, dataset: Path, min_n: int, force: bool) -> int:
    """Merge labeled candidate rows into the dataset once >= min_n new ones accumulate. New rows get split by date
    among themselves (oldest 2/3 dev) so existing dev/test assignments don't move. Rows without a gold label stay in
    their candidate file until someone sets label_override."""
    files = sorted(cand.glob("*.jsonl"))
    rows = [(f, r) for f in files for r in read_rows(f)]
    todo = [(f, r) for f, r in rows if not (r.get("label_override") or r.get("gold_label"))]
    num = {r["number"]: r["repo"] for r in read_rows(dataset)}
    new, clash = [], 0
    for r in sorted((r for f, r in rows if (f, r) not in todo), key=lambda r: r["created_at"]):
        if num.get(r["number"], r["repo"]) != r["repo"]:
            clash += 1  # ponytail: gate/bootstrap key cases on number alone; key on (repo, number) to accept these
        elif r["number"] not in num:
            num[r["number"]] = r["repo"]
            new.append(r)
    if todo:
        print(f"{len(todo)} candidates need a hand label (label_override) before they can be promoted")
    if clash:
        print(f"refused {clash} rows whose number is already in the dataset under another repo")
    if len(new) < min_n and not force:
        print(f"{len(new)} new labeled candidates, need {min_n} (or --force)")
        return 0
    new = [{**r, "sha": sha_at(clone(r["repo"]), r["created_at"])} for r in new]
    for k, r in enumerate(new):
        r["split"] = "dev" if k < len(new) * 2 // 3 else "test"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text("".join(json.dumps(r) + "\n" for r in read_rows(dataset) + new))
    for f in files:
        keep = [r for g, r in todo if g == f]
        if keep:
            f.write_text("".join(json.dumps(r) + "\n" for r in keep))
        else:
            f.unlink()
    print(f"promoted {len(new)} rows into {dataset} ({len(rows) - len(todo) - len(new)} already present or refused). "
          "Re-record eval/baseline.json: the CI gate's --stratify slice reshuffles when dev rows are added.")
    return len(new)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="issuebot.feedback")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--min", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = Path("eval/candidates")
    if a.promote:
        promote(out, Path("eval/dataset.jsonl"), a.min, a.force)
        return
    # Like for like: the live bot only labels at confidence >= min_confidence, so score offline cases the same way.
    # ponytail: uses the default min_confidence and the offline dataset's repos, not each adopter's own.
    b, conf = Path("eval/baseline.json"), CONFIG["min_confidence"][0]
    cs = [c for c in json.loads(b.read_text())["cases"] if c["confidence"] >= conf] if b.exists() else []
    baseline = sum(c["pred"] == c["gold"] for c in cs) / len(cs) if cs else None
    src = ", ".join(sorted({r["repo"] for r in read_rows(Path("eval/dataset.jsonl"))})) or "offline"
    about = f" ({src} cases at confidence >= {conf}, n={len(cs)})"
    repos = [l.split("#")[0].strip() for l in Path("adopters.txt").read_text().splitlines()]
    now, drift, table = datetime.now(timezone.utc), [], []
    for repo in filter(None, repos):
        d, row = collect(repo, a.days, out, baseline, now, about)
        drift += d
        table.append(row)
    print("\n".join(drift))
    md = "\n".join([f"## issuebot feedback ({a.days} days, ending 7 days ago)", "",
                    f"Baseline: {'-' if baseline is None else f'{baseline:.0%}'}{about}", "",
                    "| repo | bot-touched | decided | agreement | baseline | new candidates | drift |",
                    "|---|---|---|---|---|---|---|", *table, "", *[f"- {d}" for d in drift]])
    print(md)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as fh:
            fh.write(md + "\n")


if __name__ == "__main__":
    main()
