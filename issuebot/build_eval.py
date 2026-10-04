"""Pull closed issues with a maintainer reply into eval/dataset.jsonl, with gold labels and the SHA at issue time."""
import argparse
import itertools
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from issuebot.tools import clone, gh, sha_at

MAINTAINER = {"OWNER", "MEMBER", "COLLABORATOR"}
BUG_LABELS = {"p2-edge-case", "p3-minor-bug", "p3-significant", "p4-important",
              "p5-urgent", "upstream", "needs reproduction"}   # exact names from the vitest label list
FEATURE_LABELS = {"enhancement", "enhancement: pending triage", "p2-nice-to-have"}
DUP_RE = re.compile(r"(?i)(?:duplicate of|dupe of|same as|tracked in)\s+(?:[\w.-]+/[\w.-]+)?#(\d+)")


def is_maint(c: dict) -> bool:
    return c.get("author_association") in MAINTAINER and (c.get("user") or {}).get("type") != "Bot"


def maintainer_reply(comments: list[dict]) -> dict | None:
    return next((c for c in comments if is_maint(c) and len((c.get("body") or "").strip()) >= 40), None)


def gold(issue: dict, comments: list[dict]) -> tuple[str, int | None] | None:
    """(label, duplicate_of), or None when the case can't be scored and must be dropped."""
    labels = {l["name"] for l in issue.get("labels", [])}
    dups = [int(m) for c in comments if is_maint(c) for m in DUP_RE.findall(c.get("body") or "")]
    if issue.get("state_reason") == "duplicate" or "duplicate" in labels or dups:
        if not dups or dups[0] >= issue["number"]:
            return None  # no pointer, or a pointer to a later issue (unfindable without leakage)
        return "duplicate", dups[0]
    if labels & FEATURE_LABELS:
        return "feature", None
    if labels & BUG_LABELS:
        return "bug", None
    return "question", None


def original_text(repo: str, i: dict, comments: list[dict]) -> tuple[str, str] | None:
    """Title and body as opened. None when the body was edited after the first maintainer comment (answer may leak in)."""
    ev = gh(f"/repos/{repo}/issues/{i['number']}/events", params={"per_page": 100})
    title = next((e["rename"]["from"] for e in ev if e["event"] == "renamed"), i["title"])
    owner, name = repo.split("/")
    q = "query($o:String!,$r:String!,$n:Int!){repository(owner:$o,name:$r){issue(number:$n){lastEditedAt}}}"
    r = gh("/graphql", "POST", json={"query": q, "variables": {"o": owner, "r": name, "n": i["number"]}})
    edited = r["data"]["repository"]["issue"]["lastEditedAt"]
    first = next((c["created_at"] for c in comments if is_maint(c)), None)
    return None if edited and first and edited > first else (title, i["body"])


def row(repo: str, i: dict, comments: list[dict], g: tuple) -> dict | None:
    """A dataset row (sha=None, split=None: the caller fills them), or None without a usable reply or original text."""
    # A bare "Duplicate of #12" is the usual dup close, so the 40-char rule would filter dups out.
    m = (next((c for c in comments if is_maint(c) and DUP_RE.search(c.get("body") or "")), None) if g[0] == "duplicate"
         else maintainer_reply(comments))
    if not m or not (orig := original_text(repo, i, comments)):
        return None
    return {"repo": repo, "number": i["number"], "url": i["html_url"], "title": orig[0],
            "body": orig[1], "author": i["user"]["login"], "created_at": i["created_at"],
            "sha": None, "split": None,
            "gold_label": g[0], "gold_duplicate_of": g[1], "label_override": None,
            "maintainer_reply": m["body"], "maintainer": m["user"]["login"],
            "labels": [l["name"] for l in i["labels"]], "state_reason": i.get("state_reason")}


def closed_issues(repo: str):
    for page in itertools.count(1):
        batch = gh(f"/repos/{repo}/issues", params={"state": "closed", "sort": "created", "direction": "desc",
                                                    "per_page": 100, "page": page})
        if not batch:
            return
        yield from batch


def build(repo: str, limit: int, branch: str = "origin/HEAD", issues=None) -> list[dict]:
    """Up to `limit` rows (with sha, split=None) from closed issues, newest first. `issues` defaults to the REST list."""
    src = clone(repo)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for i in closed_issues(repo) if issues is None else issues:
        if ("pull_request" in i or i["user"]["type"] == "Bot" or i["author_association"] in MAINTAINER
                or not (i.get("body") or "").strip() or i["created_at"] > cutoff):
            continue
        comments = gh(f"/repos/{repo}/issues/{i['number']}/comments", params={"per_page": 100})
        if not (g := gold(i, comments)) or not (r := row(repo, i, comments, g)):
            continue
        rows.append({**r, "sha": sha_at(src, i["created_at"], branch)})
        print(f"#{i['number']} {g[0]} ({len(rows)}/{limit})", flush=True)
        if len(rows) >= limit:
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="vitest-dev/vitest")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default="eval/dataset.jsonl")
    ap.add_argument("--branch", default="origin/HEAD")
    a = ap.parse_args()

    rows = build(a.repo, a.limit, a.branch)
    rows.sort(key=lambda r: r["created_at"])
    for k, r in enumerate(rows):  # oldest 2/3 dev, newest 1/3 test (test is least likely to be memorized)
        r["split"] = "dev" if k < len(rows) * 2 // 3 else "test"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
