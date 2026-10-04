import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from issuebot import agent, backtest, run_eval, tools


def issue(n, created, title, body=""):
    return {"number": n, "created_at": created, "title": title, "body": body, "state": "open"}


CORPUS = [issue(3, "2026-03-01T00:00:00Z", "Pool crash on start", "segfault in worker pool"),
          issue(5, "2026-05-01T10:00:00Z", "Pool crash again", "me"),
          issue(8, "2026-06-01T00:00:00Z", "Pool crash future", "later"),
          issue(4, "2026-04-01T00:00:00Z", "Unrelated", "the pool mentioned once"),
          issue(2, "2026-02-01T00:00:00Z", "Crashing pools", "pools crashed")]  # porter: crashing/pools -> crash/pool


@unittest.skipUnless(tools.fts5(), "sqlite3 built without FTS5")
class FtsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = tools.index("o/r", CORPUS, Path(self.tmp.name))
        self.ctx = {"repo": "o/r", "number": 5, "created_at": "2026-05-01T10:00:00Z", "search": "fts", "fts": self.db,
                    "corpus": CORPUS}
        self.net = mock.patch.object(tools, "_search", side_effect=AssertionError("fts must not call the API"))
        self.net.start()

    def tearDown(self):
        self.net.stop()
        self.tmp.cleanup()

    def rows(self):
        with sqlite3.connect(self.db) as db:
            r = dict(db.execute("SELECT rowid, title FROM issues"))
        db.close()
        return r

    def test_index_build_and_incremental_upsert(self):
        self.assertEqual(self.db.name, "fts-o__r.sqlite")
        self.assertEqual(len(self.rows()), 5)
        tools.index("o/r", [issue(3, "2026-03-01T00:00:00Z", "Renamed"), issue(9, "2026-07-01T00:00:00Z", "New")],
                    Path(self.tmp.name))
        r = self.rows()
        self.assertEqual((len(r), r[3], r[9]), (6, "Renamed", "New"))  # upsert by number, nothing duplicated

    def test_cutoff_and_self_exclusion(self):
        nums = [h[0] for h in tools._fts(self.db, self.ctx["created_at"], 5, "pool crash", 20)]
        self.assertEqual(set(nums), {2, 3, 4})
        out = tools.search_issues(self.ctx, "pool crash created:>2027-01-01 repo:other/x", 20)
        for leak in ("#5", "#8", "future"):
            self.assertNotIn(leak, out)

    def test_bm25_ordering(self):
        nums = [h[0] for h in tools._fts(self.db, self.ctx["created_at"], 5, "pool crash", 20)]
        self.assertEqual(nums[-1], 4)  # one "pool" mention, no "crash": ranked last
        self.assertEqual(len(tools._fts(self.db, self.ctx["created_at"], 5, "pool crash", 1)), 1)

    def test_hostile_queries(self):
        for q in ('"; DROP TABLE issues; --', "NEAR(pool crash)", "*", 'pool "crash', '"', "", "   ", "pool*",
                  "title:pool", "^pool", "pool AND", "OR", "(", "{title}: pool", "pool -crash", "\0pool", "pool' OR 1=1"):
            out = tools.search_issues(self.ctx, q, 10)  # never raises, never leaks
            self.assertNotIn("#8", out)
            self.assertNotIn("#5", out)
        self.assertEqual(tools.match('"; DROP TABLE issues; --'), '"drop" OR "table" OR "issues"')
        self.assertEqual(tools.match("NEAR(pool"), '"near" OR "pool"')  # quoted words, never the NEAR operator
        self.assertEqual(tools.match('* " -x a:b AND'), "")
        self.assertEqual(tools.search_issues(self.ctx, "*", 10), "no results")
        self.assertEqual(len(self.rows()), 5)

    def test_missing_fts5_falls_back_to_local(self):
        with mock.patch.object(tools, "fts5", return_value=False), mock.patch("sys.stderr") as err:
            self.assertIsNone(tools.index("o/r", CORPUS, Path(self.tmp.name)))
            ctx = run_eval.retriever("o/r", "fts", CORPUS)
        self.assertIn("warning", "".join(c.args[0] for c in err.write.call_args_list))
        self.assertEqual((ctx["search"], ctx["fts"]), ("local", None))
        out = tools.search_issues({**self.ctx, **ctx}, "pool crash", 10)  # local: every term, newest first
        self.assertTrue(out.startswith("#3"), out)

    def test_github_mode_ignores_corpus(self):
        self.net.stop()
        with mock.patch.object(tools, "_search", return_value=((1, "2026-01-01T00:00:00Z", "Remote", ""),)) as s:
            self.assertIn("#1", tools.search_issues({**self.ctx, "search": "github"}, "pool", 10))
        s.assert_called_once()
        self.net.start()


class HitsAndRecallTest(unittest.TestCase):
    def test_hits_parse_top5(self):
        text = "\n".join(f"#{n} (2026-01-0{n}) t{n}\n  body #99 (2020-01-01) inline" for n in range(1, 8))
        self.assertEqual(agent.hits(text), [1, 2, 3, 4, 5])
        self.assertEqual(agent.hits("no results"), [])

    def test_dup_recall_at5(self):
        base = {"created_at": "2026-05-01T00:00:00Z", "pred": "bug", "pred_dup": None, "confidence": 0.5,
                "score": None, "wrong": None, "cost": 0, "judge_cost": 0, "latency_s": 1, "steps": 1, "error": None}
        cases = [{**base, "number": 1, "gold": "duplicate", "gold_dup": 7,
                  "tool_calls": [{"name": "search_issues", "input": {}, "hits": [3, 7]}]},
                 {**base, "number": 2, "gold": "duplicate", "gold_dup": 9, "tool_calls": []},
                 {**base, "number": 3, "gold": "bug", "gold_dup": None}]
        self.assertEqual(run_eval.metrics(cases)["dup_recall_at5"], 0.5)
        self.assertEqual(run_eval.metrics(cases)["dup_recall"], 0.0)  # seen but not picked


class ConfigAndRecordTest(unittest.TestCase):
    def test_config_search_validated(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "c.toml")
            self.assertEqual(agent.load_config(p)["search"], "github")
            for bad in ('"fts"', '"local"', "1"):
                p.write_text(f"search = {bad}\n")
                with self.assertRaisesRegex(ValueError, "search: must be .*--search"):
                    agent.load_config(p)

    def test_backtest_records_search(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(backtest, "corpus", return_value=CORPUS), \
                mock.patch.object(backtest.build_eval, "build", return_value=[]), \
                mock.patch.object(backtest.agent, "make_client"), mock.patch.object(backtest, "clone"), \
                mock.patch.object(tools, "index", return_value=None) as idx, \
                mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}), mock.patch("builtins.print"):
            backtest.main(["o/r", "--out", f"{d}/r.json"])  # default fts; index None = no FTS5 -> recorded local
            self.assertEqual(json.loads(Path(d, "r.json").read_text())["search"], "local")
            idx.assert_called_once_with("o/r", CORPUS)
            backtest.main(["o/r", "--search", "github", "--out", f"{d}/g.json"])
            self.assertEqual(json.loads(Path(d, "g.json").read_text())["search"], "github")

    def test_run_eval_records_search(self):
        row = {"repo": "o/r", "number": 1, "split": "dev", "gold_label": "bug", "created_at": "2026-01-01T00:00:00Z"}
        case = {"number": 1, "created_at": row["created_at"], "gold": "bug", "gold_dup": None, "pred": "bug",
                "pred_dup": None, "confidence": 0.9, "score": None, "wrong": None, "cost": 0.0, "judge_cost": 0.0,
                "latency_s": 1.0, "steps": 1, "error": None, "reply": None}
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            Path(d, "ds.jsonl").write_text(json.dumps(row) + "\n")
            os.chdir(d)
            try:
                with mock.patch.object(run_eval, "clone"), mock.patch.object(run_eval.agent, "make_client", return_value=object()), \
                        mock.patch.object(run_eval, "run_case", return_value=case) as rc, \
                        mock.patch.object(backtest, "corpus", return_value=CORPUS), \
                        mock.patch.object(sys, "argv", ["x", "--dataset", "ds.jsonl", "--no-judge", "--name", "a",
                                                        "--search", "local"]), mock.patch("builtins.print"):
                    run_eval.main()
                res = json.loads(Path("results/a.json").read_text())
            finally:
                os.chdir(cwd)
        self.assertEqual(res["search"], "local")
        self.assertEqual(rc.call_args.kwargs["search"]["search"], "local")
        self.assertIs(rc.call_args.kwargs["search"]["corpus"], CORPUS)


if __name__ == "__main__":
    unittest.main()
