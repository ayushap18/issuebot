import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import httpx

from issuebot import backtest, tools


class Clock:
    def __init__(self):
        self.t, self.sleeps = 1000.0, []

    def monotonic(self):
        return self.t

    time = monotonic

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class ThrottleTest(unittest.TestCase):
    def gh(self, handler, *paths, clock=None):
        clock = clock or Clock()
        with mock.patch.object(tools, "_http", httpx.Client(transport=httpx.MockTransport(handler))), \
                mock.patch.object(tools, "time", NS(monotonic=clock.monotonic, time=clock.time, sleep=clock.sleep)), \
                mock.patch.object(tools, "_last_search", 0.0):
            for p in paths:
                tools.gh(p)
        return clock

    def test_search_spacing(self):
        clock, at = Clock(), []

        def handler(req):
            at.append((req.url.path, clock.t))
            return httpx.Response(200, json={"items": []})

        self.gh(handler, "/search/issues", "/search/issues", "/repos/o/r", "/search/issues", clock=clock)
        searches = [t for p, t in at if p.startswith("/search/")]
        self.assertEqual(len(searches), 3)
        self.assertTrue(all(b - a >= 60 / 25 - 1e-9 for a, b in zip(searches, searches[1:])), searches)
        self.assertEqual(clock.sleeps.count(0.0), 1)  # first search and the REST call don't wait

    def test_retry_after_honored(self):
        n = []

        def handler(req):
            n.append(1)
            return httpx.Response(429, headers={"retry-after": "7"}) if len(n) == 1 else httpx.Response(200, json={})

        clock = self.gh(handler, "/repos/o/r")
        self.assertEqual(clock.sleeps, [8.0])

    def test_rate_limit_reset_honored(self):
        n = []

        def handler(req):
            n.append(1)
            if len(n) == 1:
                return httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1030"})
            return httpx.Response(200, json={})

        clock = self.gh(handler, "/repos/o/r")
        self.assertEqual(clock.sleeps, [31.0])


def issue(n, at, title="t", body="b", state="closed"):
    return {"number": n, "created_at": at, "title": title, "body": body, "state": state}


class LocalSearchTest(unittest.TestCase):
    CORPUS = [issue(3, "2026-03-01T00:00:00Z", "Pool crash on start", "segfault in worker"),
              issue(5, "2026-05-01T10:00:00Z", "Pool crash again", "me"),
              issue(8, "2026-06-01T00:00:00Z", "Pool crash future", "later"),
              issue(4, "2026-04-01T00:00:00Z", "Unrelated", "nothing")]

    def setUp(self):
        self.ctx = {"repo": "o/r", "dir": Path("."), "number": 5, "created_at": "2026-05-01T10:00:00Z",
                    "corpus": self.CORPUS}

    def test_cutoff_and_self_exclusion(self):
        with mock.patch.object(tools, "_search", side_effect=AssertionError("network")):
            out = tools.search_issues(self.ctx, 'pool "crash" created:>2027-01-01', 10)
        self.assertIn("#3 (2026-03-01) Pool crash on start", out)
        for leak in ("#5", "#8", "future", "#4"):
            self.assertNotIn(leak, out)

    def test_operators_and_exclusions_ignored(self):
        with mock.patch.object(tools, "_search", side_effect=AssertionError("network")):
            out = tools.search_issues(self.ctx, "pool OR crash AND NOT -later", 10)
        self.assertIn("#3", out)

    def test_local_miss_falls_back_to_search(self):
        with mock.patch.object(tools, "_search", return_value=((2, "2026-01-01T00:00:00Z", "Remote", ""),)) as s:
            self.assertIn("#2", tools.search_issues(self.ctx, "nomatchword", 10))
        s.assert_called_once()

    def test_corpus_cache_incremental(self):
        gh_issue = lambda n, t: {"number": n, "title": t, "body": "", "created_at": "2026-01-01T00:00:00Z",
                                 "html_url": "u", "state": "open", "state_reason": None, "author_association": "NONE",
                                 "labels": [{"name": "bug", "color": "x"}], "user": {"login": "a", "type": "User", "id": 1}}
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(backtest, "pages", return_value=[gh_issue(1, "a"), {**gh_issue(2, "pr"), "pull_request": {}}]) as p:
                self.assertEqual([i["number"] for i in backtest.corpus("o/r", Path(d))], [1])
            self.assertNotIn("since", p.call_args.args[1])
            with mock.patch.object(backtest, "pages", return_value=[gh_issue(1, "renamed"), gh_issue(3, "c")]) as p:
                got = backtest.corpus("o/r", Path(d))
            self.assertIn("since", p.call_args.args[1])
            self.assertEqual({i["number"]: i["title"] for i in got}, {1: "renamed", 3: "c"})
            self.assertEqual(got[0]["labels"], [{"name": "bug"}])


class MainTest(unittest.TestCase):
    def test_n_clamped_and_no_commit_field(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(backtest, "corpus", return_value=[]), \
                mock.patch.object(backtest.build_eval, "build", return_value=[]) as build, \
                mock.patch.object(backtest.agent, "make_client"), mock.patch.object(backtest, "clone"), \
                mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}), mock.patch("builtins.print"):
            backtest.main(["o/r", "--n", "9999", "--out", f"{d}/r.json"])
            res = json.loads(Path(d, "r.json").read_text())
        self.assertEqual(build.call_args.args[1], 500)
        self.assertNotIn("issuebot_commit", res)


class ScorecardTest(unittest.TestCase):
    def test_render_from_results_file(self):
        def case(n, gold, pred, score, wrong=False, dup=None, pdup=None):
            return {"number": n, "created_at": "2026-05-01T00:00:00Z", "gold": gold, "gold_dup": dup, "pred": pred,
                    "pred_dup": pdup, "confidence": 0.9, "score": score, "wrong": wrong, "cost": 0.05, "judge_cost": 0.01,
                    "latency_s": float(n), "steps": 3, "error": None, "reply": "r", "judge_reason": f"why {n}",
                    "url": f"https://github.com/o/r/issues/{n}", "title": f"title {n}"}
        from issuebot import run_eval
        cases = [case(1, "bug", "bug", 5), case(2, "duplicate", "duplicate", 4, dup=1, pdup=1),
                 case(3, "duplicate", "bug", 2), case(4, "question", "bug", 1, wrong=True)]
        cases += [case(n, "bug", "bug", 3) for n in range(5, 16)]
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "r.json"
            f.write_text(json.dumps({"repo": "o/r", "mode": "agent", "metrics": run_eval.metrics(cases), "cases": cases}))
            md = backtest.scorecard(json.loads(f.read_text()))
        self.assertIn("| label accuracy | 87% |", md)
        self.assertIn("| duplicates found | 1/2 |", md)
        self.assertIn("| $ total | $0.90 |", md)
        self.assertIn("| $ / issue | $0.0600 |", md)
        worst = [l for l in md.splitlines() if l.startswith("- [#")]
        self.assertEqual(len(worst), 10)
        self.assertTrue(worst[0].startswith("- [#4 title 4](https://github.com/o/r/issues/4) score 1 (wrong)"))
        self.assertTrue(worst[1].startswith("- [#3"))
        self.assertIn("local retriever", md)

    def test_empty(self):
        self.assertIn("No scorable", backtest.scorecard({"repo": "o/r", "metrics": {"n": 0}, "cases": []}))


if __name__ == "__main__":
    unittest.main()
