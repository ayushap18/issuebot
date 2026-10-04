import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from issuebot import tools

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def git(d, *args, date=None):
    env = {**GIT_ENV, **({"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date} if date else {})}
    return subprocess.run(["git", "-C", str(d), *args], env=env, check=True, capture_output=True, text=True).stdout.strip()


class GitRepoTest(unittest.TestCase):
    """Throwaway repo: Jan adds docs/guide.md, Feb adds feb.txt, Mar edits guide."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / "src"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "main")
        (self.repo / "docs").mkdir()
        (self.repo / "docs/guide.md").write_text("".join(f"line {i}\n" for i in range(1, 11)) + "use defineConfig here\n")
        (self.repo / ".env").write_text("SECRET=1\n")
        self.shas = [self.commit("jan", "2026-01-10T12:00:00Z")]
        (self.repo / "feb.txt").write_text("added in feb\n")
        self.shas.append(self.commit("feb", "2026-02-10T12:00:00Z"))
        (self.repo / "docs/guide.md").write_text("march rewrite\n")
        self.shas.append(self.commit("mar", "2026-03-10T12:00:00Z"))
        self.ctx = {"repo": "o/r", "dir": self.repo, "number": 5, "created_at": "2026-05-01T10:00:00Z"}

    def tearDown(self):
        self.tmp.cleanup()

    def commit(self, m, date):
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", m, date=date)
        return git(self.repo, "rev-parse", "HEAD")


class TestShaAndCheckout(GitRepoTest):
    def test_sha_at_picks_last_commit_before_date(self):
        self.assertEqual(tools.sha_at(self.repo, "2026-02-15T00:00:00Z", "main"), self.shas[1])
        self.assertEqual(tools.sha_at(self.repo, "2026-12-01T00:00:00Z", "main"), self.shas[2])
        with self.assertRaises(ValueError):
            tools.sha_at(self.repo, "2026-01-01T00:00:00Z", "main")

    def test_sha_at_out_of_order_committer_date(self):  # leakage: D (dated Feb 20) pushed after Mar
        (self.repo / "d.txt").write_text("d\n")
        self.commit("d", "2026-02-20T12:00:00Z")
        self.assertEqual(tools.sha_at(self.repo, "2026-03-01T00:00:00Z", "main"), self.shas[1])

    def test_checkout_hides_future_files(self):  # leakage
        wt = tools.checkout(self.repo, self.shas[0])
        self.assertFalse((wt / "feb.txt").exists())
        self.assertIn("defineConfig", (wt / "docs/guide.md").read_text())
        (wt / "junk.txt").write_text("x")
        wt2 = tools.checkout(self.repo, self.shas[2])  # reuse path: checkout -f + clean
        self.assertEqual(wt, wt2)
        self.assertTrue((wt / "feb.txt").exists())
        self.assertFalse((wt / "junk.txt").exists())


class TestRepoTools(GitRepoTest):
    def setUp(self):
        super().setUp()
        self.ctx["dir"] = tools.checkout(self.repo, self.shas[0])

    def test_grep(self):
        self.assertIn("docs/guide.md:11:use defineConfig here", tools.grep_repo(self.ctx, "defineConfig"))
        self.assertEqual(tools.grep_repo(self.ctx, "nothing_matches_this"), "no matches")
        self.assertEqual(len(tools.grep_repo(self.ctx, "line", "docs", 3).splitlines()), 3)
        self.assertEqual(tools.grep_repo(self.ctx, "SECRET"), "no matches")
        self.assertIn("docs/guide.md", tools.grep_repo(self.ctx, "defineConfig", "", 5))

    def test_grep_hides_nested_env(self):
        (self.repo / "sub").mkdir()
        (self.repo / "sub/.env").write_text("SECRET=1\n")
        self.ctx["dir"] = tools.checkout(self.repo, self.commit("env", "2026-04-01T00:00:00Z"))
        self.assertEqual(tools.grep_repo(self.ctx, "SECRET"), "no matches")

    def test_secrets_hidden_from_both_tools(self):
        (self.repo / "id.pem").write_text("PRIVATE KEY\n")
        self.ctx["dir"] = tools.checkout(self.repo, self.commit("pem", "2026-04-01T00:00:00Z"))
        (self.ctx["dir"] / "gha-creds-1.json").write_text('{"private_key": "SECRET"}')  # untracked
        self.assertEqual(tools.grep_repo(self.ctx, "PRIVATE"), "no matches")
        for bad in ["id.pem", "gha-creds-1.json"]:
            self.assertTrue(tools.call("read_file", {"path": bad, "start_line": 1, "end_line": 5}, self.ctx)[1], bad)

    def test_read_file_slice(self):
        self.assertEqual(tools.read_file(self.ctx, "docs/guide.md", 2, 3), "2: line 2\n3: line 3")

    def test_read_file_trust_boundary(self):
        (self.ctx["dir"] / "link").symlink_to("/etc/passwd")
        for bad in ["../../etc/passwd", ".git/config", ".env", "link", "/etc/passwd", "docs/../../x"]:
            out, err = tools.call("read_file", {"path": bad, "start_line": 1, "end_line": 5}, self.ctx)
            self.assertTrue(err, bad)
            self.assertTrue(out.startswith("error:"), bad)

    def test_list_docs(self):
        self.assertEqual(tools.list_docs(self.ctx, "docs"), "docs/guide.md")
        self.assertEqual(tools.list_docs(self.ctx, ""), "docs/guide.md")
        self.assertEqual(tools.list_docs(self.ctx, "./docs/"), "docs/guide.md")
        for bad in ["src", "docs/../src", "../docs"]:  # config `docs` is the allowlist
            self.assertIn("not a docs dir", tools.call("list_docs", {"subdir": bad}, self.ctx)[0])
        self.assertEqual(tools.list_docs({**self.ctx, "docs": ["site"]}, ""), "no docs")


class TestSearchIssues(unittest.TestCase):
    def setUp(self):
        tools._search.cache_clear()
        self.seen = []

        def handler(req):
            self.seen.append(req)
            return httpx.Response(200, json={"items": [
                {"number": 5, "created_at": "2026-05-01T10:00:00Z", "title": "SELF", "body": "me"},
                {"number": 8, "created_at": "2026-06-01T00:00:00Z", "title": "FUTURE", "body": "later"},
                {"number": 3, "created_at": "2026-03-02T00:00:00Z", "title": "Old crash", "body": "stack",
                 "state": "closed", "labels": [{"name": "p4-important"}], "closed_at": "2026-04-01T00:00:00Z",
                 "comments": 7}]})

        self.patch = mock.patch.object(tools, "_http", httpx.Client(transport=httpx.MockTransport(handler)))
        self.patch.start()
        self.ctx = {"repo": "o/r", "dir": Path("."), "number": 5, "created_at": "2026-05-01T10:00:00Z"}

    def tearDown(self):
        self.patch.stop()
        tools._search.cache_clear()

    def test_leakage_filter(self):
        out = tools.search_issues(self.ctx, "crash", 10)
        q = self.seen[0].url.params["q"]
        self.assertIn("created:<2026-05-01T10:00:00Z", q)
        self.assertIn("repo:o/r", q)
        self.assertIn("is:issue", q)
        self.assertIn("in:title,body", q)
        self.assertNotIn("FUTURE", out)
        self.assertIn("#3 (2026-03-02) Old crash", out)
        self.assertNotIn("SELF", out)
        self.assertNotIn("#5", out)
        for leak in ("closed", "p4-important", "2026-04-01"):
            self.assertNotIn(leak, out)

    def test_max_results_clamped(self):
        tools.search_issues(self.ctx, "x", 500)
        self.assertEqual(self.seen[0].url.params["per_page"], "21")


class TestGhAndCall(unittest.TestCase):
    def test_gh_retries_rate_limit(self):
        n = []

        def handler(req):
            n.append(1)
            if len(n) == 1:
                return httpx.Response(429, headers={"retry-after": "0"})
            return httpx.Response(200, json={"ok": True})

        with mock.patch.object(tools, "_http", httpx.Client(transport=httpx.MockTransport(handler))), \
                mock.patch.object(tools.time, "sleep") as sleep:
            self.assertEqual(tools.gh("/x"), {"ok": True})
        self.assertEqual(len(n), 2)
        sleep.assert_called_once()

    def test_call_errors(self):
        out, err = tools.call("nope", {}, {})
        self.assertTrue(err)
        with mock.patch.dict(tools.FNS, {"grep_repo": mock.Mock(side_effect=RuntimeError("boom"))}):
            out, err = tools.call("grep_repo", {"pattern": "x"}, {})
        self.assertTrue(err)
        self.assertIn("boom", out)

    def test_schemas_strict(self):
        for t in tools.TOOLS + [tools.SUBMIT]:
            s = t["input_schema"]
            self.assertTrue(t["strict"])
            self.assertFalse(s["additionalProperties"])
            self.assertEqual(set(s["required"]), set(s["properties"]))


if __name__ == "__main__":
    unittest.main()
