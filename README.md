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

The judge (`issuebot/judge.py`, Haiku 4.5, JSON-schema structured output) grades substance, not tone, and never sees which system produced the reply. It will be calibrated against 50 hand grades (Cohen's kappa) before its numbers are quoted.

`--compare A B` pairs two result files by issue number and prints deltas with a 95% paired-bootstrap CI. A change counts as an improvement only if the CI excludes 0.

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
python -m issuebot.run_eval --compare results/A.json results/B.json
```

Other flags: `--name NAME`, `--no-judge`, `--dataset PATH`, `--split dev|test|all`.

Triage a single live issue (prints the JSON result, posts nothing):

```bash
python -m issuebot.agent --repo vitest-dev/vitest --issue 1234
```

Run the offline tests (no network, no API key needed):

```bash
python -m unittest discover -s tests -v
```

## GitHub Action

issuebot runs on `issues.opened` as a composite action. The default mode is **shadow**: it writes the predicted label, confidence, cost and draft reply to the job summary and touches nothing on the issue.

Copy `examples/issuebot.yml` to `.github/workflows/issuebot.yml` in your repo and add an `ANTHROPIC_API_KEY` secret:

```yaml
on:
  issues:
    types: [opened]
permissions:
  contents: read
  issues: write   # only used in label/comment modes
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
          mode: shadow
```

| Input | Default | Notes |
|---|---|---|
| `anthropic-api-key` | required | |
| `github-token` | `${{ github.token }}` | |
| `mode` | `shadow` | `shadow` = summary only; `label` = apply a label; `comment` = label + post the draft reply |
| `model` | `claude-sonnet-5-5` | |
| `min-confidence` | `0.8` | nothing is written below this |
| `label-map` | `{}` | JSON from issuebot labels to your repo's labels, e.g. `{"bug":"bug","question":"question"}`. Unmapped labels are never applied |
| `max-steps` | `8` | |

Posted comments carry a footer saying they are an automated triage draft. The action only reads the repo, and only triggers on `issues.opened` (never `pull_request_target`).

## Project layout

```
issuebot/
  agent.py        model ids + prices, system prompt, agent loop, trace(), CLI and Action entrypoint
  tools.py        GitHub REST helper, clone / sha_at / checkout, the 4 tools + submit schema
  build_eval.py   closed issues -> eval/dataset.jsonl (gold labels, SHA at issue time)
  run_eval.py     run agent or baseline, score, write results/<name>.json, --compare
  judge.py        LLM-as-judge vs the maintainer's reply
tests/            offline unittest suite (fake Anthropic client, httpx.MockTransport, temp git repos)
eval/             dataset.jsonl
results/          committed result files
runs/             per-run JSONL traces (gitignored)
examples/         example workflow for adopters
action.yml        composite GitHub Action
PLAN.md           build plan and design decisions
SCALING.md        what changes when this runs on many repos
```

## Roadmap

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
