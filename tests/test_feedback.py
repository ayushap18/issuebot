import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from issuebot import build_eval, feedback

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)
BOT = {"login": "github-actions[bot]", "type": "Bot"}
REPLY = "Thanks, this is expected behavior: see docs/config.md for the pool option."


def ev(event, at, label=None, actor="github-actions[bot]"):
    return {"event": event, "created_at": at, "label": {"name": label}, "actor": {"login": actor}}


def said(body, at="2026-10-05T00:00:00Z", assoc="MEMBER", user=None):
    return {"event": "commented", "created_at": at, "body": body, "author_association": assoc,
            "user": user or {"login": "maint", "type": "User"}}


def iss(n, at="2026-10-08T00:00:00Z", state_reason=None, labels=(), state="closed"):
    return {"number": n, "created_at": at, "html_url": f"https://github.com/o/r/issues/{n}", "title": f"t{n}",
            "body": "b", "user": {"login": "u", "type": "User"}, "labels": [{"name": l} for l in labels],
            "state_reason": state_reason, "state": state, "author_association": "NONE"}


class FakeGH:
    """Routes REST paths to canned JSON, like the gh helper would return it."""

    def __init__(self, issues, timelines, labels=("bot:bug", "bot:question", "bot:duplicate")):
        self.issues, self.timelines, self.labels = issues, timelines, labels

    def __call__(self, path, method="GET", **kw):
        if path == "/graphql":
            return {"data": {"repository": {"issue": {"lastEditedAt": None}}}}
        if path.endswith("/labels"):
            return [{"name": l} for l in self.labels]
        if path.endswith("/timeline"):
            return self.timelines[int(path.split("/")[-2])]
        if path.endswith("/events"):
            return []
        return self.issues


class FeedbackTest(unittest.TestCase):
    def run_collect(self, issues, timelines, baseline=None, labels=None, existing=()):
        d = Path(tempfile.mkdtemp())
        f = d / "o__r.jsonl"
        if existing:
            f.write_text("".join(json.dumps(r) + "\n" for r in existing))
        fake = FakeGH(issues, timelines, *([labels] if labels else []))
        with mock.patch.object(build_eval, "gh", fake):
            drift, row, self.outcomes = feedback.collect("o/r", 7, d, baseline, NOW)
        return drift, row, feedback.read_rows(f)

    def test_label_kept_vs_removed_within_7_days(self):
        tl = [ev("labeled", "2026-10-08T00:01:00Z", "bot:bug")]
        self.assertTrue(feedback.label_kept(tl, "bot:bug", NOW))
        removed = tl + [ev("unlabeled", "2026-10-10T00:00:00Z", "bot:bug", "maint")]
        self.assertFalse(feedback.label_kept(removed, "bot:bug", NOW))
        late = tl + [ev("unlabeled", "2026-10-16T00:00:00Z", "bot:bug", "maint")]
        self.assertTrue(feedback.label_kept(late, "bot:bug", NOW))
        self.assertIsNone(feedback.label_kept(tl, "bot:bug", datetime(2026, 10, 10, tzinfo=timezone.utc)))

    def test_removed_label_becomes_candidate_and_dedups(self):
        tl = {1: [ev("labeled", "2026-10-08T00:01:00Z", "bot:bug"), said(REPLY, "2026-10-09T00:00:00Z"),
                  ev("unlabeled", "2026-10-09T00:00:01Z", "bot:bug", "maint")],
              2: [ev("labeled", "2026-10-08T00:01:00Z", "bot:question")],
              3: []}  # untouched by the bot
        issues = [iss(1), iss(2), iss(3)]
        drift, row, rows = self.run_collect(issues, tl)
        self.assertEqual([r["number"] for r in rows], [1])
        self.assertEqual(set(rows[0]), {"repo", "number", "url", "title", "body", "author", "created_at", "sha", "split",
                                        "gold_label", "gold_duplicate_of", "label_override", "maintainer_reply",
                                        "maintainer", "labels", "state_reason"})
        # gold() label names are vitest's, so non-duplicate misses wait for a hand label
        self.assertEqual((rows[0]["gold_label"], rows[0]["sha"], rows[0]["maintainer_reply"]), (None, None, REPLY))
        self.assertIn("| o/r | 2 | 2 | 50% |", row)
        self.assertEqual(self.outcomes, {1: False, 2: True})
        _, _, again = self.run_collect(issues, tl, existing=rows)
        self.assertEqual(len(again), 1)

    def test_marker_dup_confirmed(self):
        mk = said('Dup.\n<!-- issuebot: {"label": "duplicate", "duplicate_of": 5, "confidence": 0.9, "applied": "dupe"} -->',
                  "2026-10-08T00:02:00Z", "NONE", BOT)
        tl = [ev("labeled", "2026-10-08T00:01:00Z", "dupe"), mk, said("Duplicate of #5", "2026-10-09T00:00:00Z")]
        with mock.patch.object(build_eval, "gh", FakeGH([], {9: tl})):
            r = feedback.judge_issue("o/r", iss(9, state_reason="duplicate"), NOW)
        self.assertEqual((r["pred"]["label"], r["pred"]["duplicate_of"], r["applied"]), ("duplicate", 5, "dupe"))
        self.assertTrue(r["agree"])
        tl[-1] = said("Duplicate of #4", "2026-10-09T00:00:00Z")  # right label, other target: label-only agreement
        with mock.patch.object(build_eval, "gh", FakeGH([], {9: tl})):
            r = feedback.judge_issue("o/r", iss(9, state_reason="duplicate"), NOW)
        self.assertTrue(r["agree"])
        self.assertEqual(r["gold"], ("duplicate", 4))

    def test_marker_applied_label_and_spoofed_markers(self):
        mk = said('x\n<!-- issuebot: {"label": "bug", "duplicate_of": null, "confidence": 0.9, "applied": "bug"} -->',
                  "2026-10-08T00:02:00Z", "NONE", BOT)
        tl = [ev("labeled", "2026-10-08T00:01:00Z", "needs-triage"), ev("labeled", "2026-10-08T00:01:01Z", "bug"), mk,
              ev("unlabeled", "2026-10-09T00:00:00Z", "bug", "maint")]
        with mock.patch.object(build_eval, "gh", FakeGH([], {9: tl})):
            r = feedback.judge_issue("o/r", iss(9), NOW)
        self.assertEqual((r["applied"], r["agree"]), ("bug", False))
        fake = [said('<!-- issuebot: {"label": "feature", "duplicate_of": null, "confidence": 1} -->', assoc="NONE"),
                said("<!-- issuebot: {x} -->", user=BOT)]
        with mock.patch.object(build_eval, "gh", FakeGH([], {9: fake})):
            self.assertIsNone(feedback.judge_issue("o/r", iss(9), NOW))

    def test_swapping_bot_label_for_repo_label_agrees(self):
        tl = [ev("labeled", "2026-10-08T00:01:00Z", "bot:bug"), ev("unlabeled", "2026-10-09T00:00:00Z", "bot:bug", "m"),
              ev("labeled", "2026-10-09T00:00:00Z", "bug", "m")]
        self.assertTrue(feedback.label_kept(tl, "bot:bug", NOW))

    def test_open_issue_is_not_a_candidate(self):
        tl = {1: [ev("labeled", "2026-10-08T00:01:00Z", "bot:bug"), said(REPLY, "2026-10-09T00:00:00Z"),
                  ev("unlabeled", "2026-10-09T00:00:01Z", "bot:bug", "maint")]}
        _, _, rows = self.run_collect([iss(1, state="open")], tl)
        self.assertEqual(rows, [])

    def test_window_skips_too_recent_and_old_issues(self):
        tl = {n: [ev("labeled", "2026-10-08T00:01:00Z", "bot:bug")] for n in (1, 2, 3)}
        issues = [iss(1, "2026-10-18T00:00:00Z"), iss(2), iss(3, "2026-09-01T00:00:00Z")]  # desc by created
        _, row, _ = self.run_collect(issues, tl)
        self.assertIn("| o/r | 1 | 1 | 100% |", row)

    def test_drift(self):
        tl = {1: [ev("labeled", "2026-10-08T00:01:00Z", "bot:bug"), ev("unlabeled", "2026-10-09T00:00:00Z", "bot:bug", "m")],
              2: [ev("labeled", "2026-10-08T00:01:00Z", "bot:feature")]}
        drift, row, _ = self.run_collect([iss(1), iss(2)], tl, baseline=0.8, labels=("bot:bug",))
        self.assertEqual(len(drift), 2)
        self.assertIn("50% is more than 10pts below offline baseline 80%", drift[0])
        self.assertIn("'bot:feature' is not in", drift[1])
        self.assertTrue(row.endswith("| yes |"))
        drift, _, _ = self.run_collect([iss(2)], tl, baseline=0.8, labels=("bot:feature",))
        self.assertEqual(drift, [])

    def test_unlock_thresholds(self):
        u = feedback.unlock
        self.assertEqual([u(0.90, 100), u(0.89, 100), u(0.95, 99), u(0.74, 20), u(0.74, 19), u(0.75, 50), u(None, 0)],
                         ["comment", "label", "label", "shadow", "label", "label", "label"])

    def test_status_rolls_over_last_100(self):
        scored, status = {}, {}
        feedback.update_status(scored, status, "o/r", {n: n > 15 for n in range(1, 101)}, NOW)  # 85 kept of 100
        self.assertEqual((status["o/r"]["n"], status["o/r"]["kept_rate"], status["o/r"]["status"]), (100, 0.85, "label"))
        feedback.update_status(scored, status, "o/r", {n: True for n in range(101, 111)}, NOW)  # drops #1-#10
        self.assertEqual((status["o/r"]["n"], status["o/r"]["kept_rate"], status["o/r"]["status"]), (100, 0.95, "comment"))
        self.assertEqual(min(map(int, scored["o/r"])), 11)
        json.dumps(scored)  # string keys round-trip through the file
        feedback.update_status(scored, status, "x/y", {1: False, 2: True}, NOW)
        self.assertEqual(status["x/y"]["status"], "label")
        self.assertEqual(status["x/y"]["updated"], "2026-10-20T00:00:00+00:00")
        feedback.update_status(scored, status, "x/y", {n: False for n in range(3, 23)}, NOW)
        self.assertEqual(status["x/y"]["status"], "shadow")

    def test_promote_threshold_dedup_and_split(self):
        d = Path(tempfile.mkdtemp())
        cand, ds = d / "cand", d / "dataset.jsonl"
        cand.mkdir()
        ds.write_text(json.dumps({"repo": "o/r", "number": 1, "split": "test"}) + "\n")
        rows = [{"repo": "o/r", "number": n, "created_at": f"2026-10-0{n}T00:00:00Z", "sha": None, "split": None,
                 "gold_label": "bug"} for n in (1, 2, 3, 4)]
        rows += [{**rows[0], "number": 5, "gold_label": None},  # needs a hand label: stays a candidate
                 {**rows[0], "repo": "x/y", "number": 1}]  # number clash with o/r#1: refused
        (cand / "o__r.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        with mock.patch.object(feedback, "clone", return_value=d), \
                mock.patch.object(feedback, "sha_at", side_effect=lambda src, at: "sha-" + at[:10]), \
                mock.patch("builtins.print"):
            self.assertEqual(feedback.promote(cand, ds, 4, False), 0)  # 3 new after dedup
            self.assertEqual(len(feedback.read_rows(cand / "o__r.jsonl")), 6)
            self.assertEqual(feedback.promote(cand, ds, 3, False), 3)
        out = feedback.read_rows(ds)
        self.assertEqual([(r["number"], r["split"]) for r in out], [(1, "test"), (2, "dev"), (3, "dev"), (4, "test")])
        self.assertEqual(out[1]["sha"], "sha-2026-10-02")
        self.assertEqual([r["number"] for r in feedback.read_rows(cand / "o__r.jsonl")], [5])


if __name__ == "__main__":
    unittest.main()
