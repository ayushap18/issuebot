"""GitHub REST helper, repo checkout at a point in time, and the agent's read-only tools."""
import os
import subprocess
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import httpx

API = "https://api.github.com"
CACHE = Path.home() / ".cache/issuebot"
LABELS = ("question", "bug", "duplicate", "feature")
_http = httpx.Client(timeout=30)  # tests swap this for one with httpx.MockTransport
SEARCH_GAP = 60 / 25  # search API allows 30/min; stay at <= 25/min
_last_search = 0.0


def gh(path: str, method="GET", **kw) -> dict | list:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if tok := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {tok}"
    global _last_search
    for attempt in range(4):
        if path.startswith("/search/"):  # ponytail: per-process spacing; parallel jobs on one token each get 25/min
            time.sleep(max(0.0, _last_search + SEARCH_GAP - time.monotonic()))
            _last_search = time.monotonic()
        r = _http.request(method, API + path, headers=headers, **kw)
        limited = r.headers.get("x-ratelimit-remaining") == "0" or "retry-after" in r.headers
        if r.status_code in (403, 429) and limited and attempt < 3:
            wait = float(r.headers.get("retry-after") or max(int(r.headers.get("x-ratelimit-reset", 0)) - time.time(), 1))
            time.sleep(min(wait, 900) + 1)
            continue
        r.raise_for_status()
        return r.json()


def pages(path: str, params: dict | None = None, stop=lambda x: False) -> list:
    out, page = [], 1
    while True:
        batch = gh(path, params={**(params or {}), "per_page": 100, "page": page})
        out += [x for x in batch if not stop(x)]
        if len(batch) < 100 or any(map(stop, batch)):
            return out
        page += 1


def git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def clone(repo: str) -> Path:
    d = CACHE / repo.replace("/", "__")
    if d.exists():
        git("-C", str(d), "fetch", "-q", "origin")
    else:
        d.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "-q", f"https://github.com/{repo}.git", str(d))
    return d


def sha_at(repo_dir: Path, iso: str, branch="origin/HEAD") -> str:
    # Walk the first-parent chain oldest first and stop at the first commit dated >= the issue. Committer dates
    # are not monotonic (fast-forwarded old commits), so `rev-list --before` could return a tree with later commits.
    ts, sha = datetime.fromisoformat(iso).timestamp(), ""
    for line in git("-C", str(repo_dir), "log", "--first-parent", "--reverse", "--format=%H %ct", branch).splitlines():
        h, t = line.split()
        if int(t) >= ts:
            break
        sha = h
    if not sha:
        raise ValueError(f"no commit before {iso}")
    return sha


def checkout(repo_dir: Path, sha: str) -> Path:
    # ponytail: single worktree means sequential eval. One worktree per worker if 300 cases x ~20s gets too slow.
    wt = Path(repo_dir).parent / f"{Path(repo_dir).name}-wt"
    if not wt.exists():
        git("-C", str(repo_dir), "worktree", "add", "-q", "--detach", str(wt), sha)
    else:
        git("-C", str(wt), "checkout", "-q", "--detach", "-f", sha)
        git("-C", str(wt), "clean", "-qfdx")
    return wt


# ---- tools (ctx = {"repo", "dir", "number", "created_at"}) ----

def _cap(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "\n...[truncated]"


def _safe(ctx, rel: str) -> Path:
    # Issue text is untrusted and steers these args, so this is a trust boundary.
    root = Path(ctx["dir"]).resolve()
    p = (root / rel).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"path outside repo: {rel}")
    parts = p.relative_to(root).parts
    if ".git" in parts or any(x.startswith(".env") for x in parts) or p.suffix in (".pem", ".key"):
        raise ValueError(f"path not allowed: {rel}")
    return p


def grep_repo(ctx, pattern: str, path: str = ".", max_results: int = 50) -> str:
    path = path or "."
    _safe(ctx, path)
    deny = [f":(exclude,glob)**/{g}" for g in (".env*", ".env*/**", "*.pem", "*.key")]  # same denylist as _safe
    r = subprocess.run(["git", "-C", str(ctx["dir"]), "grep", "-n", "-I", "-E", "-e", pattern, "--", path, *deny],
                       capture_output=True, text=True)
    if r.returncode == 1:
        return "no matches"
    if r.returncode:
        raise RuntimeError(r.stderr.strip())
    return _cap("\n".join(r.stdout.splitlines()[:max(1, min(max_results, 200))]), 8000) or "no matches"


def read_file(ctx, path: str, start_line: int = 1, end_line: int | None = None) -> str:
    p = _safe(ctx, path)
    rel = str(p.relative_to(Path(ctx["dir"]).resolve()))
    # Tracked files only (same set git grep sees): untracked files like gha-creds-*.json can hold secrets.
    if subprocess.run(["git", "--literal-pathspecs", "-C", str(ctx["dir"]), "ls-files", "--error-unmatch", "--", rel],
                      capture_output=True).returncode:
        raise ValueError(f"not a tracked file: {path}")
    lines = p.read_text(errors="replace").splitlines()
    start = max(1, start_line or 1)
    end = min(end_line or start + 199, start + 399, len(lines))
    return _cap("\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1)), 16000) or "(empty range)"


def list_docs(ctx, subdir: str = "") -> str:
    dirs = ctx.get("docs") or ["docs"]  # allowlist from the repo config's `docs`
    if subdir:
        sub = Path(os.path.normpath(subdir))
        if not any(sub.is_relative_to(os.path.normpath(d)) for d in dirs):
            raise ValueError(f"not a docs dir: {subdir} (allowed: {', '.join(dirs)})")
        dirs = [str(sub)]
    for d in dirs:
        _safe(ctx, d)
    out = git("-C", str(ctx["dir"]), "ls-files", "--", *(p for d in dirs for p in (f"{d}/**/*.md", f"{d}/*.md")))
    return _cap(out.strip(), 8000) or "no docs"


def _hit(i: dict) -> tuple:
    # Leakage rule: keep only fields known when the issue was opened. No state, labels, comments, closed_at.
    return i["number"], i["created_at"], i["title"], (i.get("body") or "")[:400]


@lru_cache(maxsize=1024)  # ponytail: in-process cache; disk cache if reruns become frequent
def _search(repo: str, created_at: str, query: str, n: int) -> tuple:
    # in:title,body: comments can postdate the issue being answered, so matching on them leaks.
    q = f"repo:{repo} is:issue in:title,body created:<{created_at} {query}"
    items = gh("/search/issues", params={"q": q, "per_page": n})["items"]
    return tuple(map(_hit, items))


def _local(corpus: list[dict], created_at: str, number: int, query: str, n: int) -> list[tuple]:
    # ponytail: every-term substring match, newest first, not GitHub's ranking; SQLite FTS5 is the Stage 3 upgrade.
    terms = [t.lower() for t in query.replace('"', " ").split()  # qualifiers, operators and exclusions dropped
             if ":" not in t and t not in ("OR", "AND", "NOT") and not t.startswith("-")]
    hits = [i for i in corpus if i["created_at"] < created_at and i["number"] != number
            and all(t in f"{i['title']}\n{i.get('body') or ''}".lower() for t in terms)]
    hits.sort(key=lambda i: i["created_at"], reverse=True)
    return [_hit(i) for i in hits[:n]]


def search_issues(ctx, query: str, max_results: int = 10) -> str:
    n = max(1, min(max_results or 10, 20))
    # ctx["corpus"] (backtest's cached issue list) answers locally; only a local miss costs a search API call.
    local = _local(ctx["corpus"], ctx["created_at"], ctx["number"], query, n) if ctx.get("corpus") else []
    # Recheck locally: the model writes `query` and qualifiers in it (created:>..., repo:...) can widen the filter.
    hits = [h for h in local or _search(ctx["repo"], ctx["created_at"], query, n + 1)
            if h[0] != ctx["number"] and h[1] < ctx["created_at"]][:n]
    return "\n".join(f"#{num} ({d[:10]}) {t}\n  {b}" for num, d, t, b in hits) or "no results"


def _schema(props: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "properties": props, "required": list(props)}


TOOLS = [
    {"name": "grep_repo", "strict": True,
     "description": "Search tracked files in the repo (checked out as of the issue date) with an extended regex via git grep. "
                    "Returns path:line:text. Use for docs (path 'docs') and source (path 'packages').",
     "input_schema": _schema({"pattern": {"type": "string", "description": "Extended regex"},
                              "path": {"type": "string", "description": "Subdirectory or pathspec, default '.'"},
                              "max_results": {"type": "integer", "description": "Default 50"}})},
    {"name": "read_file", "strict": True,
     "description": "Read a line range of a repo file. Lines are numbered. Max 400 lines per call.",
     "input_schema": _schema({"path": {"type": "string"}, "start_line": {"type": "integer"},
                              "end_line": {"type": "integer"}})},
    {"name": "list_docs", "strict": True,
     "description": "List markdown documentation files in the repo.",
     "input_schema": _schema({"subdir": {"type": "string", "description": "One of the repo's doc dirs; empty = all of them"}})},
    {"name": "search_issues", "strict": True,
     "description": "Full-text search of this repo's issues created BEFORE the current issue. Returns number, date, title "
                    "and a body snippet. Use it to find duplicates. Use short keyword queries such as an error message "
                    "fragment or API name.",
     "input_schema": _schema({"query": {"type": "string"},
                              "max_results": {"type": "integer", "description": "Default 10, max 20"}})},
]

SUBMIT = {"name": "submit", "strict": True,
          "description": "Submit the final triage. Call exactly once, when done.",
          "input_schema": _schema({
              "label": {"type": "string", "enum": list(LABELS)},
              "duplicate_of": {"type": ["integer", "null"], "description": "Earlier issue number if label is duplicate, else null"},
              "reply": {"type": "string", "description": "Markdown reply to post as a maintainer"},
              "confidence": {"type": "number", "description": "0-1 probability the label is correct"}})}

FNS = {"grep_repo": grep_repo, "read_file": read_file, "list_docs": list_docs, "search_issues": search_issues}


def call(name: str, args: dict, ctx: dict) -> tuple[str, bool]:
    try:
        return FNS[name](ctx, **args), False
    except Exception as e:  # one bad tool call must not kill the run
        return f"error: {type(e).__name__}: {e}", True
