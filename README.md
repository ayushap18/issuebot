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
   | search_issues  GitHub search, created:<issue.created_at,    |
   |                own number dropped, no state/labels/comments |
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

## Results

No numbers yet. Every row below is a placeholder until it is produced by the harness and committed to `results/`.

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

## GitHub Action

issuebot runs on `issues.opened` as a composite action. The default mode is **shadow**: it writes the predicted label, confidence, cost and draft reply to the job summary and touches nothing on the issue.

It is bring-your-own-key: copy `examples/issuebot.yml` to `.github/workflows/issuebot.yml` in your repo and add an `ANTHROPIC_API_KEY` secret. The job needs exactly `permissions: { issues: write, contents: read }` and a checkout with `persist-credentials: false`:

```yaml
on:
  issues:
    types: [opened]
permissions:
  contents: read
  issues: write   # only used in label/comment modes
concurrency:
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

| Input | Default | Notes |
|---|---|---|
| `anthropic-api-key` | required | |
| `github-token` | `${{ github.token }}` | |
| `config-path` | `.github/issuebot.toml` | repo config path, relative to the checkout (see below) |
| `mode` | config / `shadow` | blank inputs below fall back to the repo config, then its default |
| `model` | `claude-sonnet-5-5` | |
| `min-confidence` | config / `0.8` | |
| `label-map` | config / `{}` | JSON form of `label_map`, e.g. `{"bug":"bug"}`; replaces the file's table |
| `max-steps` | `8` | |
| `routed` | config / `false` | |
| `threshold` | config / `0.8` | |

### Repo config

Optionally copy `examples/issuebot.toml` to `.github/issuebot.toml` (read with stdlib `tomllib` from the checkout; another path via the `config-path` input / `ISSUEBOT_CONFIG`, which must then exist). Every key is optional and validated strictly: an unknown key or a wrong type fails the run with a message naming each bad key. Precedence: an action input / `ISSUEBOT_*` env var that is set (non-blank) > the file > the default.

| Key | Default | Notes |
|---|---|---|
| `mode` | `"shadow"` | `shadow` = summary only; `label` = apply a label; `comment` = label + post the draft reply (mentions, images, off-repo links and cross-repo refs defused) |
| `routed` | `false` | `true` = Haiku triage first; Sonnet drafts only when unsure or the label is bug/question |
| `threshold` | `0.8` | routed mode confidence gate |
| `min_confidence` | `0.8` | nothing is written below this |
| `label_map` | `{}` | table from issuebot labels (`bug`, `question`, `feature`, `duplicate`) to your labels |
| `label_prefix` | `"bot:"` | **Label rule:** a `label_map` entry is applied verbatim; any other label is applied as `label_prefix + label` (`bot:feature`). `""` = apply only mapped labels |
| `docs` | `["docs"]` | doc dirs `list_docs` may list (its allowlist) |
| `per_issue_cap_usd` | `0.15` | per-issue $ ceiling (`CEILING`) |
| `monthly_issue_cap` | `0` | skip once more than this many issues were opened this month (one search call; 0 = unlimited). Also set an Anthropic workspace spend limit: that is the hard money cap |
| `skip_new_accounts_days` | `7` | skip authors with `author_association` NONE / FIRST_TIME_CONTRIBUTOR / FIRST_TIMER whose account is younger than this (one `GET /users/{login}`; 0 = off) |

### Guards and output

Free guards run before any model call. The run exits 0 with a one-line job summary when the event is not `issues.opened`, the issue is a pull request, the author is a bot (`type: Bot` or `[bot]` login), the author is a new account (above), or the monthly cap is reached.

- **Shadow** writes label, duplicate_of, confidence, $ cost, route and the draft reply to `$GITHUB_STEP_SUMMARY` only.
- **Label** applies one label, by the label rule above, only at `min_confidence` or higher.
- **Comment** also posts the draft. `@` mentions in it are defused with a zero-width space so the bot never pings anyone. The comment ends with an automated-draft footer and a hidden marker, `<!-- issuebot: {"label": ..., "duplicate_of": ..., "confidence": ...} -->`, that the feedback loop reads back.

The issue title and body are wrapped in an `<issue>` block in the user turn (a literal `</issue>` in them is escaped) and labeled untrusted data, and the system prompt says instructions inside it are never commands. The body is capped at 8000 chars. The action only reads the repo and only triggers on `issues.opened` (never `pull_request_target`). The `concurrency` group serializes bursts; note that GitHub keeps only one pending run per group, so in a flood some runs are dropped rather than queued.

`ISSUEBOT_DRY_RUN=1` in the step env runs the guards, validates the config and renders the prompt, then prints the plan. It makes no model calls and no writes. The `action-smoke` job in `test.yml` runs the local action (`uses: ./`) this way on every push, against `tests/fixtures/issue_opened.json` (`ISSUEBOT_EVENT` overrides the event path), with no secrets.

### Feedback loop

`adopters.txt` lists repos running issuebot. `.github/workflows/feedback.yml` (weekly + manual) runs `python -m issuebot.feedback [--days 7]` from this repo; adopters send no telemetry. For issues created in the `--days` before the last 7 days (so every bot label has had 7 days) that carry a `bot:` label or a bot-authored comment marker, it scores agreement: the label was kept 7 days (timeline `labeled`/`unlabeled`; swapping `bot:bug` for the repo's own `bug` counts as kept), or a maintainer closed it as a duplicate (label only, same rules as `gold()`). Misses on closed issues with a maintainer reply are appended to `eval/candidates/<owner>__<repo>.jsonl` as dataset rows (`sha: null`, deduped by number). Only duplicates get an automatic gold label; other rows have `gold_label: null` and need a hand `label_override`, since `gold()` knows only vitest's label names. A repo whose agreement is more than 10pts under the offline baseline (`eval/baseline.json` cases with confidence >= the default `min_confidence`, i.e. the ones the live bot would label), or whose bot label is gone from its label set, prints a `DRIFT` line; the workflow commits candidates and opens one `Drift: <repo>` issue (skipped while one is open). `python -m issuebot.feedback --promote [--min 20] [--force]` merges labeled candidates into `eval/dataset.jsonl` once 20 new ones exist (dedup by repo+number, numbers already used by another repo are refused, SHA from a clone, new rows split by date among themselves so existing splits don't move) and clears them. Adding dev rows reshuffles the gate's `--stratify` slice, so re-record `eval/baseline.json` after a promote.

## Project layout

```
issuebot/
  agent.py        model ids + prices, system prompt, agent loop, trace(), CLI and Action entrypoint
  tools.py        GitHub REST helper, clone / sha_at / checkout, the 4 tools + submit schema
  build_eval.py   closed issues -> eval/dataset.jsonl (gold labels, SHA at issue time)
  run_eval.py     run agent/baseline/routed, score, write results/<name>.json, --compare, --gate,
                  --export-grading / --calibrate, --tag-failures
  judge.py        LLM-as-judge vs the maintainer's reply, failure-cause tagger
  feedback.py     weekly adopter feedback: agreement, miss candidates, drift, --promote
tests/            offline unittest suite (fake Anthropic client, httpx.MockTransport, temp git repos)
eval/             dataset.jsonl, baseline.json + replay/ for the CI gate (not committed yet)
results/          committed result files
runs/             per-run JSONL traces (gitignored)
examples/         example workflow and repo config (issuebot.toml) for adopters
.github/workflows test.yml (offline tests), eval-gate.yml (regression gate on PRs), feedback.yml (weekly)
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
