# issuebot

An agent that triages new GitHub issues and drafts the first maintainer reply. It ships with an eval harness that scores it on real closed issues, and the harness never lets the agent see anything from after the issue was opened.

## Why

Maintainers of mid-size projects spend a lot of time on first responses: "is this a bug or a usage question?", "this is a duplicate of #1234", "please add a minimal reproduction", "see the docs for `test.pool`". That work is repetitive, but a wrong bot answer posted with confidence does more harm than no answer at all.

So the question issuebot tries to answer is not "can an LLM write a reply" but **how often is the reply right, how often is it confidently wrong, and what does each issue cost**. The eval harness is the main deliverable. The bot is what it measures.

Target repo for v1: [`vitest-dev/vitest`](https://github.com/vitest-dev/vitest). It gets about 20+ issues a week, its maintainers are correctly tagged by GitHub's `author_association`, its docs live in the repo, and its labels (`p1..p5`, `enhancement`, `duplicate`, `needs reproduction`) give usable ground truth.

## How it works

```
                 issue (number, title, body, created_at)
                                  |
                                  v
            +-----------------------------------------+
            |  agent loop  (max 8 steps, tool_choice  |
            |  auto, last step only offers `submit`)  |
            +-----------------------------------------+
                 |  tool_use                ^ tool_result
                 v                          |
   +------------------------------------------------------------+
   | grep_repo      git grep over tracked files at the issue SHA |
   | read_file      numbered line range, path-guarded            |
   | list_docs      markdown files under docs/                   |
   | search_issues  GitHub search (or cached FTS5/local index),  |
   |                created:<issue.created_at, own number        |
   |                dropped, no state/labels/comments            |
   +------------------------------------------------------------+
                                  |
                                  v
   submit -> {label: question|bug|duplicate|feature, duplicate_of: int|null,
              reply: str, confidence: 0-1}
                                  |
                                  v
             runs/YYYY-MM-DD.jsonl  (input, tool calls, output,
                                     tokens, cost, latency)
```

- Models are constants at the top of `issuebot/agent.py`: `claude-sonnet-5-5` drafts replies (default agent model), `claude-haiku-4-5` is the judge and the cheap path.
- All tools are read-only. Issue text is treated as untrusted data: `read_file` rejects paths outside the repo, `.git/`, `.env*`, `*.pem` and `*.key`.
- Every run appends one JSON line to `runs/` with the rendered input, each tool call (name, args, size, ms, error), the output, token usage, cost and latency.
- Baseline mode is the same prompt and output schema with only the `submit` tool, so agent vs baseline is a clean ablation of retrieval.

## Eval methodology

### Dataset

`python -m issuebot.build_eval` pulls closed issues from the target repo into `eval/dataset.jsonl` (default 300 rows). An issue is kept only if:

- it was filed by a non-maintainer, non-bot user, has a body, and is at least 14 days old;
- an `OWNER`/`MEMBER`/`COLLABORATOR` (non-bot) replied with at least 40 characters. That reply is the reference answer.

Gold labels, applied in order:

| Label | Rule |
|---|---|
| duplicate | `state_reason == duplicate`, a `duplicate` label, or a maintainer comment matching "duplicate of / same as / tracked in #N". Dropped if there is no number or it points to a later issue. |
| feature | any of `enhancement`, `enhancement: pending triage`, `p2-nice-to-have` |
| bug | any of `p2-edge-case`, `p3-minor-bug`, `p3-significant`, `p4-important`, `p5-urgent`, `upstream`, `needs reproduction` |
| question | everything else |

Rows are sorted by `created_at`. The oldest two thirds are `split: dev`, the newest third is `split: test`. A `label_override` field holds hand labels for a random sample, and scoring uses `label_override or gold_label`.

### Leakage prevention

A triage eval is easy to fool: if the agent can see the fix commit, the updated docs, or the issue's own closed state, it scores well for the wrong reason. Each case is run as if it were the moment the issue was opened.

1. **Code and docs are frozen at the issue date.** For each row, `sha` is the last first-parent commit on the default branch before `created_at` (`git rev-list -1 --first-parent --before=<created_at>`). The agent works in a worktree checked out at that SHA, and `grep_repo` only searches files tracked at that commit.
2. **Issue search only sees the past.** `search_issues` adds `created:<created_at` to the GitHub query, then rechecks dates locally (the model writes the query and could add qualifiers that widen it), and drops the issue's own number.
3. **No post-hoc fields.** Search results contain only number, date, title and a 400-char body snippet. Never state, labels, comments or `closed_at`.
4. **The issue text is the original.** The title is the pre-rename one (first `renamed` event). Rows whose body was edited after the first maintainer comment are dropped, since the answer may have been pasted in.
5. **Duplicates must be findable.** A duplicate pointing to a later issue is dropped, since it can't be found without seeing the future.

Each of these guards has an offline unit test in `tests/`.

Known remaining leakage: earlier issues' bodies may have been edited after the fact, and the model may have seen public vitest issues during pretraining. The test split is the newest third to limit the second, and scores by date make memorization visible as a drop on recent issues.

### Metrics

`python -m issuebot.run_eval` writes `results/<name>.json` with per-case rows and:

| Metric | Meaning |
|---|---|
| `label_accuracy` | predicted label == gold. Printed next to `majority_floor` (share of the most common class) |
| `macro_f1` | mean F1 over the 4 labels, so a bot that always says "bug" can't look good |
| `dup_precision` / `dup_recall` | a duplicate counts only if `duplicate_of` matches the gold issue number |
| `dup_recall_at5` | retrieval only: share of gold duplicates whose original was in the top 5 of any `search_issues` call the agent made (from `hits` in each case's `tool_calls`; results files from before `hits` was recorded score 0) |
| `judge_mean`, `judge_ge4` | LLM-as-judge score 1-5 vs the maintainer's actual reply, and the share scoring 4+ |
| `wrong_rate`, `confidently_wrong` | judge flags a factual contradiction or a fix that would not work; `confidently_wrong` = wrong AND confidence >= 0.7 |
| `acc_at_0.8`, `coverage_at_0.8` | accuracy and share of cases at confidence >= 0.8 (picks the Action threshold) |
| `cost_per_issue` | agent + judge $, from token usage incl. cache reads/writes |
| latency p50 / p95, `avg_steps`, `error_rate` | per case |

The judge (`issuebot/judge.py`, Haiku 4.5, JSON-schema structured output) grades substance, not tone, and never sees which system produced the reply. Its numbers are not quoted until it is calibrated against 50 hand grades:

```bash
python -m issuebot.run_eval --export-grading results/A.json > grading.csv   # --n 50, seeded, judge score hidden
# fill in human_score (1-5) and optionally human_wrong (yes/no) for each row
python -m issuebot.run_eval --calibrate grading.csv results/A.json
```

`--calibrate` prints exact and within-1 agreement, quadratic-weighted Cohen's kappa (plus kappa on `wrong` if graded), and exits 1 if weighted kappa < 0.6.

`--compare A B` pairs two result files by issue number and prints deltas with a 95% paired-bootstrap CI. A change counts as an improvement only if the CI excludes 0.

`--tag-failures RESULTS` is the Stage 3 error analysis. Each failed case (wrong label, missed duplicate, judge <= 2 or wrong) is tagged by Haiku with one primary cause: `retrieval_miss`, `reasoning`, `taxonomy`, `missing_context` or `other`. It gets the issue, the agent's tool calls, its output and the maintainer reply. Tags are written back into the results file, counts and shares are printed, and `STAGE 3 TRIGGER MET` is printed when retrieval misses are >= 30% of failures. It goes through the replay cache like every other call.

`--gate BASELINE [NEW]` is the CI regression gate. It pairs NEW (or, without NEW, the results of the run it is attached to) with BASELINE, prints a markdown table (metric | baseline | new | delta | CI | verdict, also appended to `$GITHUB_STEP_SUMMARY`), and exits 1 only when a drop is outside the 95% CI **and** past its floor: label accuracy -3pts, dup precision -5pts, judge mean -0.2. `--stratify` makes `--limit` round-robin over gold labels so a small slice still covers every class.

`.github/workflows/eval-gate.yml` runs it on PRs touching `issuebot/**`: `--split dev --limit 40 --stratify --gate eval/baseline.json`. By convention the baseline is `eval/baseline.json` (a results file from that exact command on main) and its replay cache is `eval/replay/` (`ISSUEBOT_REPLAY_DIR`). Neither is committed yet, so the gate currently skips with a notice. With the `ANTHROPIC_API_KEY` secret it runs in record mode (cache hits free, misses live); without it (fork PRs) it runs replay-only if `eval/replay/` exists, otherwise skips. To create them:

```bash
ISSUEBOT_REPLAY=record ISSUEBOT_REPLAY_DIR=eval/replay \
  python -m issuebot.run_eval --split dev --limit 40 --stratify --name baseline
cp results/baseline.json eval/baseline.json
```

### Run evals on a subscription CLI

No `ANTHROPIC_API_KEY` needed: `ISSUEBOT_BACKEND=claude-cli` runs every model call through `claude -p` on your Claude subscription (one fresh run per call, empty temp dir, no tools/MCP/settings, `ANTHROPIC_API_KEY` stripped from its env so it can't bill the key). Tool use is emulated: the transcript and tool schemas go in, a JSON-schema'd `tool_calls` list comes back as `tool_use` blocks, so `agent.run` is unchanged.

```bash
export GITHUB_TOKEN=$(gh auth token)
# one issue
ISSUEBOT_BACKEND=claude-cli python -m issuebot.agent --repo vitest-dev/vitest --issue 1234
# eval, judged by claude-cli, with a cross-model second judge
ISSUEBOT_BACKEND=claude-cli python -m issuebot.run_eval --split dev --limit 20 --stratify --judge2 agy
# judge on a different backend than the agent
ISSUEBOT_BACKEND=claude-cli ISSUEBOT_JUDGE_BACKEND=agy python -m issuebot.run_eval --split dev --limit 20
# backtest
ISSUEBOT_BACKEND=claude-cli python -m issuebot.backtest vitest-dev/vitest --n 20
```

- Every trace, case and results file records its `backend` (`judge_backend`, `judge2_backend` at the top level). CLI `$` figures are the CLI's list-price-equivalent (`total_cost_usd`), not what you're billed; agy/codex report no price, so their `$` is 0. Don't compare CLI cost columns against API runs as spend.
- `--judge2 agy|codex|claude-cli` scores each reply with a second judge (`judge2_score`, `judge2_wrong`, `judge2_reason`) and adds `metrics.judge_agreement` (exact, within-1, weighted kappa, wrong kappa).
- **agy and codex are judge-only** (`ISSUEBOT_JUDGE_BACKEND` / `--judge2`). Neither can turn off its own tools (agy browses and searches, codex runs shell commands), so as the agent it could look up how the real issue was resolved: leakage. `ISSUEBOT_BACKEND=agy|codex` and passing tools to them raise an error. As judges they already see the maintainer reply, so there's nothing to leak.
- Knobs: `ISSUEBOT_CLI_TIMEOUT` (s, default 300), `ISSUEBOT_AGY_MODEL` (default `gemini-3.8-flash-medium`, list with `agy models`), `ISSUEBOT_CODEX_MODEL` (default: codex's own config). Replay caching works as usual; the backend is part of the cache key.

## Results

No numbers yet. Every row below is a placeholder until it is produced by the harness and committed to `results/`.

Dashboard: <https://ayushap18.github.io/issuebot/>, built by `.github/workflows/pages.yml` on every push to main and after each feedback run from `results/*.json` and `eval/status.json` (worst repo first: live kept rate, else offline label accuracy; repos in `eval/status.json` get a row even without a results file, and it says "No results yet" only when both are empty). Pages must be enabled once in the repo settings with source "GitHub Actions". Build it locally with `python -m issuebot.dashboard [--results results] [--status eval/status.json] [--out site]`.

| Run | Split | n | Label acc (floor) | Macro-F1 | Dup P / R | Judge mean | Confidently wrong | $/issue | p50 / p95 latency |
|---|---|---|---|---|---|---|---|---|---|
| Baseline, Sonnet 5.5 (no tools) | dev | TBD | TBD — run `python -m issuebot.run_eval` | TBD | TBD | TBD | TBD | TBD | TBD |
| Agent, Sonnet 5.5 | dev | TBD | TBD — run `python -m issuebot.run_eval` | TBD | TBD | TBD | TBD | TBD | TBD |
| Agent, Haiku 4.5 | dev | TBD | TBD — run `python -m issuebot.run_eval` | TBD | TBD | TBD | TBD | TBD | TBD |
| Agent, best config | test | TBD | TBD — run `python -m issuebot.run_eval` | TBD | TBD | TBD | TBD | TBD | TBD |

## Quickstart

Requires Python 3.13 and `git`.

```bash
git clone https://github.com/ayushap18/issuebot && cd issuebot
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .

export ANTHROPIC_API_KEY=...   # agent + judge
export GITHUB_TOKEN=...        # REST, search and GraphQL (a read-only token is enough)
```

Build the dataset (clones the target repo once into `~/.cache/issuebot/`):

```bash
python -m issuebot.build_eval --repo vitest-dev/vitest --limit 300 --out eval/dataset.jsonl
```

Run the baseline, then the agent. Start small on dev:

```bash
python -m issuebot.run_eval --mode baseline --split dev --limit 40
python -m issuebot.run_eval --mode agent    --split dev --limit 40
python -m issuebot.run_eval --mode agent    --split dev --limit 40 --model claude-haiku-4-5
python -m issuebot.run_eval --mode routed   --split dev --limit 40 --threshold 0.8
python -m issuebot.run_eval --compare results/A.json results/B.json
python -m issuebot.run_eval --tag-failures results/A.json
python -m issuebot.run_eval --split dev --limit 40 --stratify --gate eval/baseline.json
```

Routed mode runs Haiku triage first and only runs the Sonnet agent when Haiku's confidence is below `--threshold` or the label is `bug`/`question`; the trace records `route`, `triage_cost` and `draft_cost`. Every run stops tool-looping at $0.15 (`CEILING`), takes one submit-only step, and records `capped: true`.

Other flags: `--name NAME`, `--stratify`, `--gate BASELINE [NEW]`, `--export-grading RESULTS [--n 50]`, `--calibrate CSV RESULTS`, `--tag-failures RESULTS`, `--no-judge`, `--dataset PATH`, `--split dev|test|all`. See [Metrics](#metrics) for what each does.

Replay cache: `ISSUEBOT_REPLAY=record` stores every model call (agent and judge) under `cache/replay/` (override with `ISSUEBOT_REPLAY_DIR`), keyed on the sha256 of the full request. `ISSUEBOT_REPLAY=replay` serves only from the cache and fails on a miss, so unchanged cases cost $0 and need no API key.

Triage a single live issue (prints the JSON result, posts nothing):

```bash
python -m issuebot.agent --repo vitest-dev/vitest --issue 1234
python -m issuebot.agent --repo vitest-dev/vitest --issue 1234 --routed --threshold 0.8
```

Run the offline tests (no network, no API key needed):

```bash
python -m unittest discover -s tests -v
```

## Install on your repo

issuebot runs on `issues.opened` as a composite GitHub Action with your own Anthropic key (BYOK). Nothing runs on my side: your key, your Actions minutes, your bill.

1. **Add the workflow.** Copy [`examples/issuebot.yml`](examples/issuebot.yml) to `.github/workflows/issuebot.yml`:

   ```yaml
   name: issuebot
   on:
     issues:
       types: [opened]
   permissions:
     contents: read
     issues: write   # only used in label/comment modes
   concurrency:      # a spam burst queues instead of fanning out
     group: issuebot-${{ github.repository }}
     cancel-in-progress: false
   jobs:
     triage:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
           with:
             fetch-depth: 1
             persist-credentials: false
         - uses: ayushap18/issuebot@v1
           with:
             anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
   ```

   Required permissions are exactly `contents: read` and `issues: write`, nothing else. Keep `persist-credentials: false` so no token lands in `.git/config` where the read-only tools could see it.

2. **Add the secret.** Repo Settings > Secrets and variables > Actions > New repository secret, named `ANTHROPIC_API_KEY`. Or with `gh`: `gh secret set ANTHROPIC_API_KEY --repo owner/repo`.

3. **Set a spend limit.** Give issuebot its own workspace in the Anthropic Console and set a monthly spend limit on it. That is the only hard money cap. `per_issue_cap_usd` and `monthly_issue_cap` below bound normal runs, but they are best-effort checks inside a stateless job.

4. **Optionally add a config.** Copy [`examples/issuebot.toml`](examples/issuebot.toml) to `.github/issuebot.toml`. Without it every key takes its default (shadow mode).

5. **Promote slowly: shadow, then label, then comment.**
   - `mode = "shadow"` (default): each run writes label, duplicate_of, confidence, $ cost, route and the draft reply to the job summary. Nothing on the issue changes. Read a few weeks of summaries.
   - `mode = "label"`: applies one label at `min_confidence` or above. Unmapped labels get the `bot:` prefix (`bot:bug`), so they never collide with your own. Add `label_map` entries once you trust a class.
   - `mode = "comment"`: also posts the draft reply. Only switch when the labels have held up; one bad public reply costs more than no reply. Comment mode also needs a per-repo unlock (see [Per-repo unlock and auto-demotion](#per-repo-unlock-and-auto-demotion)); until then the Action runs it as label mode.

   To be counted by the feedback loop (below), add your repo to [`adopters.txt`](adopters.txt) with a PR.

### Backtest before installing

See how issuebot would have done on your own repo before it touches a live issue:

```bash
python -m issuebot.backtest owner/repo [--n 100] [--mode agent|routed] [--search fts|local|github] [--out results/backtest-owner__repo.json]
```

It takes the last `--n` closed issues that got a maintainer reply (same filters and gold labels as `build_eval`), runs each with the repo checked out at the commit before the issue was opened and with search limited to earlier issues, judges every reply against the maintainer's, and writes the results file plus a markdown scorecard (label accuracy, duplicates found X/Y, judge mean, $ total and $/issue, p50/p95 latency, the 10 worst replies with links) to stdout and the job summary.

The repo's issue list is fetched once through the REST list endpoint and cached at `cache/corpus-<owner>__<repo>.json` (reruns fetch only issues updated since). By default (`--search fts`) `search_issues` is answered from an SQLite FTS5 index of that corpus and never calls the search API; `--search local` and `--search github` select the other backends (see [Issue search backends](#issue-search-backends-stage-3)). Every search API call in issuebot is spaced to at most 25/min and honors `retry-after` / `x-ratelimit-reset`.

### Issue search backends (Stage 3)

`search_issues` has three backends, chosen per run with `--search` (`run_eval` default `github`, `backtest` default `fts`) and recorded as `search` in the results file so A/B runs can be told apart:

| `--search` | How | API calls |
|---|---|---|
| `github` | GitHub search API, `in:title,body created:<created_at` | one per query, <= 25/min |
| `local` | every query term must appear in title or body of the cached corpus, newest first | only on a local miss |
| `fts` | SQLite FTS5 index (`porter unicode61`) of the cached corpus, terms ORed, ranked by bm25 | none |

The FTS index is `cache/fts-<owner>__<repo>.sqlite`, built by `tools.index(repo, corpus)` from the cached corpus and updated incrementally (only new or edited issues are rewritten; rowid = issue number). It sits next to the corpus in `cache/`, so the backtest workflow's `actions/cache` keeps both. The query is untrusted (the model writes it): qualifiers, `AND`/`OR`/`NOT` and `-exclusions` are dropped, every remaining word token is double-quoted so it is always a plain string to FTS5 (no `NEAR`, `*`, column filters or syntax errors), and the SQL is parameterized. Leakage guards are the same as the other backends: `created_at < case created_at AND number != case number` in SQL, then the same local recheck, and the same `_hit` fields. If the local sqlite3 lacks FTS5, `fts` prints one warning and runs as `local` (and the results file says `local`).

`local` and `fts` need the corpus, so `run_eval --search local|fts` fetches it once per repo in the dataset (REST list pages, ~1 request per 100 issues the first time). The live Action stays on `github`: the config key `search` only accepts `"github"` until an A/B shows FTS wins.

A/B on the same slice (FTS is a flag, off by default; turn it on only if the eval says so):

```bash
python -m issuebot.run_eval --split dev --limit 40 --stratify --search github --name ab-github
python -m issuebot.run_eval --split dev --limit 40 --stratify --search fts    --name ab-fts
python -m issuebot.run_eval --compare results/ab-github.json results/ab-fts.json   # paired bootstrap, incl. dup_recall_at5
```

For one repo's backtest: `python -m issuebot.backtest owner/repo --search github --out results/bt-github.json`, the same with `--search fts --out results/bt-fts.json`, then `--compare` the two files. Both runs judge every reply, so the A/B costs about two eval runs.

To run it in Actions on your key, copy [`examples/issuebot-backtest.yml`](examples/issuebot-backtest.yml) to `.github/workflows/` and run it from the Actions tab (input `n`: 25, 50, 100 or 200; the CLI caps `--n` at 500). It installs issuebot from a pinned tag and needs `v1.1.0` or later (the first release with the backtest); pin a full commit SHA for reproducible runs. It needs only `contents: read` + `issues: read`, caches `cache/` with `actions/cache`, and uploads the results JSON as an artifact. Expect roughly $5-9 per 100 issues. Public repos only (the clone is anonymous).

### Action inputs

| Input | Default | Notes |
|---|---|---|
| `anthropic-api-key` | required | |
| `github-token` | `${{ github.token }}` | |
| `config-path` | `.github/issuebot.toml` | relative to the checkout. The default file is optional; a path set here must exist |
| `mode` | config / `shadow` | blank inputs fall back to the repo config, then its default |
| `model` | `claude-sonnet-5-5` | ignored when routed |
| `min-confidence` | config / `0.8` | |
| `label-map` | config / `{}` | JSON form of `label_map`, e.g. `{"bug":"bug"}`; replaces the file's table |
| `max-steps` | `8` | ignored when routed |
| `routed` | config / `false` | |
| `threshold` | config / `0.8` | |

### Config reference

Read with stdlib `tomllib`. Every key is optional and validated strictly by `CONFIG` in `issuebot/agent.py`: an unknown key or a bad value fails the run with a message naming each bad key. Precedence: an action input / `ISSUEBOT_*` env var that is set (non-blank) > the file > the default. The first three columns are generated from the validator.

| Key | Default | Valid values | Notes |
|---|---|---|---|
| `mode` | `"shadow"` | "shadow", "label" or "comment" | `shadow` = job summary only; `label` = apply a label; `comment` = label + post the draft reply |
| `routed` | `false` | true or false | Haiku triage first; Sonnet drafts only when Haiku confidence < `threshold` or the label is bug/question |
| `threshold` | `0.8` | a number from 0 to 1 | routed mode confidence gate |
| `min_confidence` | `0.8` | a number from 0 to 1 | nothing is written to the issue below this |
| `label_map` | `{}` | a table from question/bug/duplicate/feature to your label names | a mapped label is applied verbatim |
| `label_prefix` | `"bot:"` | a string | unmapped labels are applied as `label_prefix + label`; `""` = apply only mapped labels |
| `docs` | `["docs"]` | a list of directories | the dirs `list_docs` may list (its allowlist) |
| `per_issue_cap_usd` | `0.15` | a number > 0 | past this $ the loop gets one submit-only step, then stops (`capped: true`) |
| `monthly_issue_cap` | `0` | an integer >= 0 (0 = unlimited) | skip once more than this many issues were opened this month (one search call). Counts all issues opened, not only triaged ones |
| `skip_new_accounts_days` | `7` | an integer >= 0 (0 = off) | skip authors with `author_association` NONE / FIRST_TIME_CONTRIBUTOR / FIRST_TIMER whose account is younger than this (one `GET /users/{login}`) |
| `search` | `"github"` | "github" (local and fts need a cached corpus: use --search in run_eval / backtest) | the `search_issues` backend. The Action has no cached corpus, so only the GitHub search API is allowed here for now; see [Issue search backends](#issue-search-backends-stage-3) |

### Guards and output

Free guards run before any model call. The run exits 0 with a one-line job summary when the event is not `issues.opened`, the issue is a pull request, the author is a bot (`type: Bot` or `[bot]` login), the author is a new account (above), or the monthly cap is reached.

- **Shadow** writes label, duplicate_of, confidence, $ cost, route and the draft reply to `$GITHUB_STEP_SUMMARY` only.
- **Label** applies one label, by the label rule above, only at `min_confidence` or higher.
- **Comment** also posts the draft (also gated on `min_confidence`). It ends with an automated-draft footer and a hidden marker, `<!-- issuebot: {"label": ..., "duplicate_of": ..., "confidence": ..., "applied": ...} -->`, that the feedback loop reads back.

`ISSUEBOT_DRY_RUN=1` in the step env runs the guards, validates the config and renders the prompt, then prints the plan. It makes no model calls and no writes. The `action-smoke` job in `test.yml` runs the local action (`uses: ./`) this way on every push, against `tests/fixtures/issue_opened.json` (`ISSUEBOT_EVENT` overrides the event path), with no secrets.

### Security model

Issue text is untrusted input from anyone on the internet.

- **Prompt injection.** Title and body sit in an `<issue>` block in the user turn (a literal `<issue`/`</issue>` in them is escaped), labeled untrusted data, and the system prompt says instructions inside it are never commands. Body capped at 8000 chars. Output must parse as `{label, duplicate_of, reply, confidence}` and `label` is forced into the four known labels. Default mode writes nothing to the issue.
- **Posted replies are defused.** `@` mentions get a zero-width space (no pings), images and off-repo links become inline code (no beacons or phishing links), and `owner/repo#N` cross-references are broken so they don't backlink.
- **File exfiltration.** All four tools are read-only. `read_file` resolves symlinks and rejects paths outside the checkout, `.git/`, `.env*`, `*.pem` and `*.key`; `grep_repo` excludes the same globs; `list_docs` only lists the `docs` allowlist. The checkout uses `persist-credentials: false`.
- **Spam and cost.** The `concurrency` group serializes bursts (GitHub keeps only one pending run per group, so in a flood some runs are dropped rather than queued). Bots and new accounts are skipped before any model call. Then the per-issue $ cap, the monthly issue cap and your workspace spend limit.
- **Token scope.** The job needs only `contents: read` and `issues: write`. The only trigger is `issues.opened`, never `pull_request_target`. Your API key stays in your repo secret; issuebot stores nothing.

### Feedback loop

Adopters send no telemetry. [`adopters.txt`](adopters.txt) lists repos running issuebot (one `owner/repo` per line, `#` comments allowed). `.github/workflows/feedback.yml` runs weekly (Monday 06:00 UTC) and on manual dispatch in this repo, calling `python -m issuebot.feedback [--days 7]`, which reads each adopter's public issue timelines with this repo's token.

- **Window.** Issues created in the `--days` before the last 7 days, so every bot label has had 7 days, that carry a `bot:` label or a bot-authored comment marker.
- **Agreement.** The label was kept 7 days (timeline `labeled`/`unlabeled`; swapping `bot:bug` for the repo's own `bug` counts as kept), or a maintainer closed it as a duplicate (label only, same rules as `gold()`).
- **Misses.** Misses on closed issues with a maintainer reply are appended to `eval/candidates/<owner>__<repo>.jsonl` as dataset rows (`sha: null`, deduped by number). Only duplicates get an automatic gold label; other rows have `gold_label: null` and need a hand `label_override`, since `gold()` knows only vitest's label names.
- **Drift.** A repo whose agreement is more than 10pts under the offline baseline (`eval/baseline.json` cases with confidence >= the default `min_confidence`, i.e. the ones the live bot would label), or whose bot label is gone from its label set, prints a `DRIFT` line. The workflow commits candidates and opens one `Drift: <repo>` issue (skipped while one is open).
- **Promotion.** `python -m issuebot.feedback --promote [--min 20] [--force]` merges labeled candidates into `eval/dataset.jsonl` once 20 new ones exist (dedup by repo+number, numbers already used by another repo are refused, SHA from a clone, new rows split by date among themselves so existing splits don't move) and clears them. Adding dev rows reshuffles the gate's `--stratify` slice, so re-record `eval/baseline.json` after a promote.

### Per-repo unlock and auto-demotion

The weekly feedback run also keeps each adopter's last 100 scored issues from the last 90 days (`eval/scored.json`, repo names lowercased) and writes `eval/status.json`. A removed label always counts; a kept label counts only if a maintainer commented or someone other than the author (not a bot) closed the issue, so ignored labels don't unlock comment mode. Outcomes older than 90 days drop out, so a repo demoted to shadow (which applies no labels) falls back under n=20 and returns to `label`. Repos removed from `adopters.txt` are dropped from both files, and a repo whose API calls fail prints a `DRIFT` line and keeps its previous status:

```json
{"owner/repo": {"kept_rate": 0.93, "n": 100, "status": "comment", "updated": "2026-10-05T06:00:00+00:00"}}
```

| Rule | status |
|---|---|
| n >= 100 and label kept >= 90% | `comment` |
| n >= 20 and label kept < 75% | `shadow` (auto-demoted) |
| otherwise | `label` |

On every non-shadow run the Action fetches `https://raw.githubusercontent.com/ayushap18/issuebot/main/eval/status.json` once (5s timeout; override with `ISSUEBOT_STATUS_URL`) and uses the lower of your configured mode and the repo's status (shadow < label < comment). It fails closed: if the fetch fails, your repo isn't listed, or its entry is malformed or its `updated` is more than 21 days old, `comment` runs as `label`; `shadow` and `label` are unaffected. The effective mode and the reason are in the job summary. Being listed requires your repo in `adopters.txt`.

## Project layout

```
issuebot/
  agent.py        model ids + prices, system prompt, agent loop, trace(), CLI and Action entrypoint
  tools.py        GitHub REST helper, clone / sha_at / checkout, the 4 tools + submit schema
  build_eval.py   closed issues -> eval/dataset.jsonl (gold labels, SHA at issue time)
  run_eval.py     run agent/baseline/routed, score, write results/<name>.json, --compare, --gate,
                  --export-grading / --calibrate, --tag-failures
  cli.py          subscription-CLI backend (claude -p agent + judge; agy / codex judge only)
  judge.py        LLM-as-judge vs the maintainer's reply, failure-cause tagger
  feedback.py     weekly adopter feedback: agreement, miss candidates, drift, per-repo status, --promote
  backtest.py     backtest one repo's last N closed issues, cached corpus, scorecard
  dashboard.py    static site/index.html from results/*.json + eval/status.json
tests/            offline unittest suite (fake Anthropic client, httpx.MockTransport, temp git repos)
eval/             dataset.jsonl, baseline.json + replay/ for the CI gate (not committed yet); status.json +
                  scored.json written by the weekly feedback run
results/          committed result files
runs/             per-run JSONL traces (gitignored)
examples/         example workflows (issuebot.yml, issuebot-backtest.yml) and repo config (issuebot.toml)
.github/workflows test.yml (offline tests), eval-gate.yml (regression gate on PRs), feedback.yml (weekly),
                  pages.yml (dashboard)
adopters.txt      repos the feedback loop reads
action.yml        composite GitHub Action
PLAN.md           build plan and design decisions
SCALING.md        what changes when this runs on many repos
```

## Roadmap

**Stage 0 tooling (built, see SCALING.md)**
- [x] Record/replay cache for every model call (`ISSUEBOT_REPLAY`)
- [x] Per-issue $0.15 ceiling and Haiku-first routed mode
- [x] CI regression gate on a 40-case stratified slice (`--gate`, `eval-gate.yml`)
- [x] Judge calibration tooling (`--export-grading`, `--calibrate`)
- [x] Failure tagging for the Stage 3 trigger (`--tag-failures`)
- [ ] Commit `eval/baseline.json` + `eval/replay/` so the gate runs instead of skipping

**Stage 1: reusable Action, BYOK (built, see SCALING.md)**
- [x] Per-repo `.github/issuebot.toml` (stdlib `tomllib`), strictly validated, inputs/env > file > defaults
- [x] Free guards: non-`issues.opened`, PRs, bots, new accounts, monthly issue cap
- [x] Injection hardening: delimited issue text, defused replies, hidden prediction marker, dry-run smoke test in CI
- [x] Weekly feedback loop over `adopters.txt`: label-kept agreement, miss candidates, drift issues, `--promote`
- [ ] Tag `v1` so `ayushap18/issuebot@v1` resolves, then install in shadow on 1-3 live repos

**Stage 2: multi-repo (built, see SCALING.md)**
- [x] `python -m issuebot.backtest owner/repo`: leakage-safe backtest with a cached issue corpus, throttled search, scorecard; `examples/issuebot-backtest.yml`
- [x] Per-repo unlock/demote in `eval/status.json`, enforced by the Action (fail closed to label)
- [x] Static dashboard on GitHub Pages, worst repo first; `branding:` in `action.yml`
- [ ] Publish backtest scorecards for 3-5 popular public repos; list on the Marketplace; 10-case CI slice per opted-in repo

**Stage 3: retrieval (step 1 built as a flag, see SCALING.md)**
- [x] SQLite FTS5 issue index (`tools.index`), `--search github|local|fts`, `dup_recall_at5` metric
- [ ] Run the github-vs-fts A/B; enable per repo only if it wins. Embeddings (step 2) only if FTS still misses

**Week 1: dataset + baseline**
- [ ] Pull 300 rows into `eval/dataset.jsonl` and commit it
- [ ] Hand-label 50 rows into `label_override`; report agreement with the heuristic
- [ ] Baseline results on dev, with label distribution and majority floor

**Week 2: agent + evals**
- [ ] Agent on dev for Sonnet 5.5 and Haiku 4.5, compared to baseline with bootstrap CIs
- [ ] Error analysis: tag 30 failures as retrieval miss, reasoning, or gold noise

**Week 3: measured improvements (dev only, one change per results file)**
- [ ] Try docs index in the cached system prompt, better duplicate queries, "needs repro" reply style, lower effort, Haiku label + Sonnet draft
- [ ] Calibrate the judge against 50 hand grades (kappa >= 0.6)
- [ ] Pick the Action's `min-confidence` from the accuracy/coverage curve
- [ ] One final run on the test split for the headline number

**Week 4: shadow deploy + write-up**
- [ ] Tag `v1` and publish the Action
- [ ] Run in shadow on new vitest issues for 2 weeks; compare to what maintainers did
- [ ] Write-up: "Evaluating an issue-triage agent without leaking the future"
- [ ] Share results with vitest maintainers and offer label-only shadow mode (no unsolicited bot comments)

## License

MIT. See [LICENSE](LICENSE).
