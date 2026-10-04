import json
import unittest
from unittest import mock

from fake import FakeClient, msg, text
from issuebot import build_eval, judge, run_eval
from issuebot.tools import SUBMIT


def comment(body, assoc="MEMBER", kind="User", login="maint"):
    return {"body": body, "author_association": assoc, "user": {"type": kind, "login": login}}


def issue(number=100, labels=(), state_reason="completed"):
    return {"number": number, "labels": [{"name": l} for l in labels], "state_reason": state_reason}


class GoldTest(unittest.TestCase):
    def test_duplicate_via_state_reason_and_regex(self):
        self.assertEqual(build_eval.gold(issue(state_reason="duplicate"), [comment("Duplicate of #42")]),
                         ("duplicate", 42))
        self.assertEqual(build_eval.gold(issue(), [comment("same as vitest-dev/vitest#7, closing")]), ("duplicate", 7))

    def test_duplicate_without_number_dropped(self):
        self.assertIsNone(build_eval.gold(issue(labels=["duplicate"]), [comment("closing as dup")]))

    def test_duplicate_of_later_issue_dropped(self):
        self.assertIsNone(build_eval.gold(issue(100, state_reason="duplicate"), [comment("duplicate of #150")]))

    def test_non_maintainer_dup_claim_ignored(self):
        self.assertEqual(build_eval.gold(issue(), [comment("dupe of #3", assoc="NONE")]), ("question", None))

    def test_feature_bug_question(self):
        self.assertEqual(build_eval.gold(issue(labels=["enhancement"]), []), ("feature", None))
        self.assertEqual(build_eval.gold(issue(labels=["p3-minor-bug", "feat: browser"]), []), ("bug", None))
        self.assertEqual(build_eval.gold(issue(labels=["feat: browser"]), []), ("question", None))

    def test_maintainer_reply(self):
        long = "This is expected, see the docs for the `pool` option in config."
        cs = [comment(long, assoc="CONTRIBUTOR"), comment(long, kind="Bot"), comment("thanks!"),
              comment(long, assoc="COLLABORATOR", login="real")]
        self.assertEqual(build_eval.maintainer_reply(cs)["user"]["login"], "real")
        self.assertIsNone(build_eval.maintainer_reply(cs[:3]))

    def test_original_text(self):
        i = {"number": 9, "title": "Root cause: X", "body": "b"}
        cs = [{**comment("hi"), "created_at": "2026-01-02T00:00:00Z"}]
        ev = [{"event": "labeled"}, {"event": "renamed", "rename": {"from": "it crashes", "to": "y"}},
              {"event": "renamed", "rename": {"from": "y", "to": "Root cause: X"}}]
        gql = lambda t: {"data": {"repository": {"issue": {"lastEditedAt": t}}}}
        with mock.patch.object(build_eval, "gh", side_effect=[ev, gql("2026-01-01T00:00:00Z")]):
            self.assertEqual(build_eval.original_text("o/r", i, cs), ("it crashes", "b"))
        with mock.patch.object(build_eval, "gh", side_effect=[[], gql("2026-01-03T00:00:00Z")]):
            self.assertIsNone(build_eval.original_text("o/r", i, cs))


def case(n, gold, pred, conf, score, wrong, lat, gd=None, pd=None, err=None):
    return {"number": n, "created_at": "2026-02-01T00:00:00Z", "gold": gold, "pred": pred, "gold_dup": gd,
            "pred_dup": pd, "confidence": conf, "score": score, "wrong": wrong, "cost": 0.1, "judge_cost": 0.01,
            "latency_s": lat, "steps": 4, "error": err}


CASES = [
    case(1, "bug", "bug", 0.9, 5, False, 1.0),
    case(2, "bug", "question", 0.8, 2, True, 2.0),
    case(3, "duplicate", "duplicate", 0.9, 4, False, 3.0, gd=10, pd=10),
    case(4, "duplicate", "duplicate", 0.6, 1, True, 4.0, gd=11, pd=12),
    case(5, "question", "question", 0.5, 3, False, 5.0),
    case(6, "feature", None, 0.0, None, None, None, err="boom"),
]


class MetricsTest(unittest.TestCase):
    def test_hand_computed(self):
        m = run_eval.metrics(CASES)
        self.assertAlmostEqual(m["label_accuracy"], 4 / 6)
        self.assertAlmostEqual(m["majority_floor"], 2 / 6)
        self.assertAlmostEqual(m["dup_precision"], 0.5)
        self.assertAlmostEqual(m["dup_recall"], 0.5)
        # F1: bug 2/3, question 2/3, duplicate 1, feature 0
        self.assertAlmostEqual(m["macro_f1"], 7 / 12)
        self.assertAlmostEqual(m["judge_mean"], 3.0)
        self.assertAlmostEqual(m["judge_ge4"], 2 / 5)
        self.assertAlmostEqual(m["wrong_rate"], 2 / 5)
        self.assertAlmostEqual(m["confidently_wrong"], 1 / 5)
        self.assertAlmostEqual(m["acc_at_0.8"], 2 / 3)
        self.assertAlmostEqual(m["coverage_at_0.8"], 0.5)
        self.assertEqual((m["latency_p50"], m["latency_p95"]), (3.0, 5.0))
        self.assertAlmostEqual(m["error_rate"], 1 / 6)
        self.assertAlmostEqual(m["cost_per_issue"], 0.11)
        self.assertEqual(m["confusion"]["bug"]["question"], 1)
        self.assertEqual(m["confusion"]["feature"][None], 1)
        json.dumps(m)  # must serialize for results/<name>.json

    def test_empty_dup_denominators(self):
        m = run_eval.metrics([c for c in CASES if c["gold"] not in ("duplicate",) and c["pred"] != "duplicate"])
        self.assertEqual((m["dup_precision"], m["dup_recall"]), (0.0, 0.0))
        self.assertEqual(run_eval.metrics([]), {"n": 0})

    def test_pct(self):
        self.assertEqual(run_eval.pct(list(range(1, 21)), 0.95), 19)
        self.assertEqual(run_eval.pct([7.0], 0.95), 7.0)
        self.assertEqual(run_eval.pct([], 0.95), 0.0)

    def test_macro_f1_skips_absent_labels(self):
        self.assertAlmostEqual(run_eval.metrics([c for c in CASES if c["gold"] == c["pred"] != "duplicate"])["macro_f1"], 1.0)

    def test_bootstrap(self):
        a = [{"number": i, "x": float(i % 3 == 0)} for i in range(60)]
        b = [{"number": i, "x": float(i % 3 != 2)} for i in range(60)]
        key = lambda c: c["x"]
        r1, r2 = run_eval.bootstrap(a, b, key), run_eval.bootstrap(a, b, key)
        self.assertEqual(r1, r2)
        d, lo, hi = r1
        self.assertGreater(d, 0)
        self.assertGreater(lo, 0)
        self.assertLessEqual(lo, d)
        self.assertLessEqual(d, hi)


class JudgeTest(unittest.TestCase):
    ISSUE = {"title": "t", "body": "b"}

    def test_parses_structured_output(self):
        fc = FakeClient(msg(text('{"reason": "same fix", "score": 4, "wrong": false}'), stop="end_turn"))
        j = judge.judge(self.ISSUE, "maintainer said X", "bot said X", client=fc)
        self.assertEqual((j["score"], j["wrong"], j["reason"]), (4, False, "same fix"))
        self.assertGreater(j["cost"], 0)
        self.assertEqual(fc.calls[0]["model"], judge.JUDGE_MODEL)
        self.assertEqual(fc.calls[0]["output_config"]["format"]["schema"], judge.SCHEMA)

    def test_out_of_range_raises(self):
        for bad in ('{"reason": "", "score": 9, "wrong": false}', '{"reason": "", "score": 3, "wrong": "no"}', "nope"):
            with self.assertRaises(ValueError):
                judge.judge(self.ISSUE, "m", "r", client=FakeClient(msg(text(bad))))


class RunCaseTest(unittest.TestCase):
    ROW = {"repo": "o/r", "number": 9, "title": "t", "body": "b", "created_at": "2026-01-01T00:00:00Z",
           "sha": "abc", "gold_label": "bug", "label_override": "question", "gold_duplicate_of": None,
           "maintainer_reply": "m", "labels": ["p4-important"], "state_reason": "completed"}
    REC = {"label": "question", "duplicate_of": None, "reply": "r", "confidence": 0.7, "cost": 0.02,
           "latency_s": 1.0, "steps": 1, "error": None}

    def run_case(self, mode):
        with mock.patch.object(run_eval, "checkout", return_value="/wt"), \
                mock.patch.object(run_eval.agent, "run", return_value=self.REC) as run:
            c = run_eval.run_case(self.ROW, "/src", mode, "m", client=None, do_judge=False)
        return c, run

    def test_baseline_has_no_retrieval_tools(self):
        c, run = self.run_case("baseline")
        self.assertEqual(run.call_args.kwargs["tools"], [SUBMIT])
        self.assertEqual(c["gold"], "question")  # label_override wins
        self.assertEqual(c["pred"], "question")
        issue = run.call_args.args[0]
        self.assertEqual(set(issue), {"number", "title", "body", "created_at"})  # no gold fields leak in

    def test_agent_mode_and_errors(self):
        c, run = self.run_case("agent")
        self.assertNotIn("tools", run.call_args.kwargs)
        with mock.patch.object(run_eval, "checkout", side_effect=RuntimeError("net down")):
            c = run_eval.run_case(self.ROW, "/src", "agent", "m", client=None)
        self.assertIn("net down", c["error"])
        self.assertIsNone(c["pred"])
        self.assertEqual((c["score"], c["wrong"]), (1, False))  # failures count in judge metrics


if __name__ == "__main__":
    unittest.main()
