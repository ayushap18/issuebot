import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from issuebot import dashboard


def res(repo, acc, title="t"):
    case = {"number": 1, "gold": "bug", "pred": "question", "score": 1, "wrong": True, "reply": "r",
            "url": "https://github.com/x/y/issues/1", "title": title}
    return {"name": f"backtest-{repo}", "repo": repo, "metrics": {"n": 10, "label_accuracy": acc, "dup_recall": 0.5,
            "judge_mean": 3.2, "cost_per_issue": 0.04}, "cases": [case]}


class DashboardTest(unittest.TestCase):
    def build(self, results=(), status=None):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d) / "results"
            r.mkdir()
            for k, x in enumerate(results):
                (r / f"{k}.json").write_text(json.dumps(x))
            (r / "junk.json").write_text("[1]")  # not a results file: ignored
            if status is not None:
                (Path(d) / "status.json").write_text(json.dumps(status))
            with mock.patch("builtins.print"):
                dashboard.main(["--results", str(r), "--status", str(Path(d) / "status.json"), "--out", str(Path(d) / "site")])
            return (Path(d) / "site/index.html").read_text()

    def test_empty_state(self):
        html = self.build()
        self.assertIn("No results yet", html)
        self.assertNotIn("<table", html)
        self.assertNotIn("<td>", html)  # no invented numbers

    def test_worst_first_and_live_status(self):
        html = self.build([res("good/repo", 0.9), res("bad/repo", 0.6)],
                          {"bad/repo": {"kept_rate": 0.7, "n": 30, "status": "shadow"},
                           "live/only": {"kept_rate": 0.65, "n": 25, "status": "shadow"},
                           "new/repo": {"kept_rate": None, "n": 0, "status": "label"}})
        # live kept rate wins over offline accuracy: live/only 65% < bad/repo 70% (acc 60%) < good/repo acc 90%
        order = [html.index(r) for r in ("live/only", "bad/repo", "good/repo", "new/repo")]
        self.assertEqual(order, sorted(order))
        self.assertIn("<td>60%</td>", html)
        self.assertIn("<td>70% (30)</td>", html)
        self.assertIn("<td>10</td>", html)  # results n, not the status n

    def test_non_numeric_status_sorts_worst_and_renders(self):
        html = self.build([res("ok/repo", 0.2)], {"odd/repo": {"kept_rate": "93%", "n": 5, "status": "label"},
                                                  "bad/entry": "junk"})
        self.assertLess(html.index("odd/repo"), html.index("ok/repo"))
        self.assertIn("bad/entry", html)

    def test_repo_casing_matches_status(self):
        html = self.build([res("Bad/Repo", 0.6)], {"bad/repo": {"kept_rate": 0.7, "n": 30, "status": "shadow"}})
        self.assertEqual(html.lower().count("<td>bad/repo</td>"), 1)  # one row, not a results row plus a status-only row
        self.assertIn("<td>70% (30)</td>", html)

    def test_escapes_html(self):
        html = self.build([res("<script>alert(1)</script>/r", 0.5, title='<img src=x onerror="a()">')])
        self.assertNotIn("<script>alert", html)
        self.assertNotIn("<img", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;img src=x onerror=&quot;a()&quot;&gt;", html)
        bad = res("o/r", 0.5)
        bad["cases"][0]["url"] = "javascript:alert(1)"
        self.assertNotIn("javascript:", self.build([bad]))


if __name__ == "__main__":
    unittest.main()
