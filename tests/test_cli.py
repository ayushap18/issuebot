import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

from issuebot import agent, cli, judge, run_eval
from issuebot.tools import SUBMIT, TOOLS

REAL_RUN = subprocess.run


def claude_out(structured, cost=0.01):
    return {"is_error": False, "result": "", "structured_output": structured, "total_cost_usd": cost,
            "modelUsage": {"m": {"inputTokens": 900, "outputTokens": 40, "cacheReadInputTokens": 5,
                                 "cacheCreationInputTokens": 7, "costUSD": cost}}}


def calls(*pairs):
    return {"tool_calls": [{"name": n, "input_json": json.dumps(i)} for n, i in pairs]}


SUB = {"label": "bug", "duplicate_of": None, "reply": "Add a repro.", "confidence": 0.7}


class FakeCLI:
    """Stands in for subprocess.run on CLI commands; anything else (git in tools) runs for real."""

    def __init__(self, *outs):
        self.outs, self.cmds = list(outs), []

    def __call__(self, cmd, **kw):
        if cmd[0] not in ("claude", "agy", "codex"):
            return REAL_RUN(cmd, **kw)
        self.cmds.append({"cmd": cmd, **kw})
        out = self.outs.pop(0) if len(self.outs) > 1 else self.outs[0]
        if cmd[0] == "codex":
            Path(cmd[cmd.index("-o") + 1]).write_text(json.dumps(out))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, json.dumps(out), "")

    def arg(self, flag, i=-1):
        c = self.cmds[i]["cmd"]
        return c[c.index(flag) + 1]


def patched(fake):
    return mock.patch("issuebot.cli.subprocess.run", fake)


class TranscriptTest(unittest.TestCase):
    def test_renders_tool_use_and_result_skips_thinking(self):
        t = cli.transcript([
            {"role": "user", "content": "Issue text"},
            {"role": "assistant", "content": [NS(type="thinking", thinking="secret"),
                                              NS(type="tool_use", id="cli_0_0", name="grep_repo", input={"pattern": "x"})]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "cli_0_0", "content": "a.md:1:x",
                                          "is_error": True}]},
            {"role": "user", "content": "Call the submit tool now."}])
        self.assertIn("USER: Issue text", t)
        self.assertIn('called tool grep_repo (id cli_0_0) with {"pattern": "x"}', t)
        self.assertIn("TOOL RESULT for cli_0_0 (error):\na.md:1:x", t)
        self.assertIn("USER: Call the submit tool now.", t)
        self.assertNotIn("secret", t)


class CreateTest(unittest.TestCase):
    def create(self, out, tools=TOOLS + [SUBMIT], messages=None):
        fake = FakeCLI(out)
        with patched(fake):
            r = cli.CLIClient("claude-cli").messages.create(
                model="claude-sonnet-5-5", system="SYS", tools=tools, max_tokens=1,
                messages=messages or [{"role": "user", "content": "hi"}], cache_control={"type": "ephemeral"})
        return r, fake

    def test_tool_calls_become_tool_use_blocks(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": [NS(type="text", text="x")]},
                {"role": "user", "content": "go"}]
        r, fake = self.create(claude_out(calls(("grep_repo", {"pattern": "a", "path": ".", "max_results": 5}),
                                               ("list_docs", {"subdir": ""}))), messages=msgs)
        self.assertEqual(r.stop_reason, "tool_use")
        self.assertEqual([(b.type, b.id, b.name) for b in r.content],
                         [("tool_use", "cli_1_0", "grep_repo"), ("tool_use", "cli_1_1", "list_docs")])
        self.assertEqual(r.content[0].input["pattern"], "a")
        schema = json.loads(fake.arg("--json-schema"))
        self.assertEqual(schema["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"],
                         [t["name"] for t in TOOLS + [SUBMIT]])
        self.assertIn("input_schema", fake.arg("--system-prompt"))
        c = fake.cmds[0]["cmd"]
        for flag in ("--strict-mcp-config", "--no-session-persistence"):
            self.assertIn(flag, c)
        self.assertEqual(fake.arg("--tools"), "")
        self.assertEqual(fake.arg("--setting-sources"), "")
        self.assertEqual((r.usage.input_tokens, r.usage.output_tokens, r.usage.cache_read_input_tokens,
                          r.usage.cache_creation_input_tokens, r._cost), (900, 40, 5, 7, 0.01))

    def test_malformed_output_is_a_text_turn(self):
        for bad in ({"tool_calls": [{"name": "submit", "input_json": "{not json"}]},
                    calls(("submit", {"label": "bug"})),  # missing required keys
                    {"tool_calls": []}, None):
            r, _ = self.create(claude_out(bad))
            self.assertEqual((r.stop_reason, r.content[0].type), ("end_turn", "text"))

    def test_env_strips_api_key_and_runs_in_empty_dir(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-x", "KEEP": "1"}):
            _, fake = self.create(claude_out(calls(("submit", SUB))))
        env, cwd = fake.cmds[0]["env"], fake.cmds[0]["cwd"]
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertEqual(env["KEEP"], "1")
        self.assertFalse(Path(cwd).exists())  # temp dir cleaned up
        self.assertEqual(fake.cmds[0]["timeout"], 300)

    def test_errors_raise(self):
        with self.assertRaisesRegex(RuntimeError, "claude CLI error"):
            self.create({"is_error": True, "result": "quota"})
        boom = mock.Mock(side_effect=subprocess.TimeoutExpired("claude", 1))
        with mock.patch("issuebot.cli.subprocess.run", boom), self.assertRaisesRegex(RuntimeError, "timed out"):
            cli.CLIClient("claude-cli").create(model="m", system="s", messages=[{"role": "user", "content": "x"}])

    def test_agy_and_codex_refused_as_agent(self):
        for kind in ("agy", "codex"):
            with self.assertRaisesRegex(ValueError, "only be a judge"):
                cli.CLIClient(kind).create(model="m", system="s", tools=[SUBMIT],
                                           messages=[{"role": "user", "content": "x"}])
            with mock.patch.dict(os.environ, {"ISSUEBOT_BACKEND": kind}), self.assertRaisesRegex(ValueError, "JUDGE_BACKEND"):
                agent.make_client()


class JudgeTest(unittest.TestCase):
    ISSUE = {"title": "t", "body": "b"}
    GRADE = {"reason": "close", "score": 4, "wrong": False}

    def test_claude_structured_judge_uses_reported_cost(self):
        fake = FakeCLI(claude_out(self.GRADE, cost=0.003))
        with patched(fake):
            j = judge.judge(self.ISSUE, "maint", "draft", client=cli.CLIClient("claude-cli"))
        self.assertEqual((j["score"], j["cost"], j["backend"]), (4, 0.003, "claude-cli"))
        self.assertEqual(json.loads(fake.arg("--json-schema")), judge.SCHEMA)
        self.assertEqual(fake.arg("--model"), judge.JUDGE_MODEL)

    def test_agy_judge(self):
        out = {"status": "SUCCESS", "structured_output": self.GRADE, "response": "",
               "usage": {"input_tokens": 10, "output_tokens": 2, "thinking_tokens": 3, "cache_read_tokens": 1}}
        fake = FakeCLI(out)
        with patched(fake), mock.patch.dict(os.environ, {"ISSUEBOT_AGY_MODEL": "gem"}):
            j = judge.judge(self.ISSUE, "maint", "draft", client=cli.CLIClient("agy"))
        self.assertEqual((j["score"], j["usage"]["output"], j["cost"]), (4, 5, 0.0))
        self.assertEqual(fake.arg("--model"), "gem")
        self.assertTrue(fake.arg("-p").startswith(judge.RUBRIC))  # no system flag: prepended
        sent = json.loads(fake.arg("--json-schema"))["properties"]
        self.assertNotIn("enum", sent["score"])  # Gemini rejects integer enums
        with patched(FakeCLI({"status": "ERROR", "error": "INVALID_ARGUMENT"})), \
                self.assertRaisesRegex(RuntimeError, "agy status ERROR: INVALID_ARGUMENT"):
            judge.judge(self.ISSUE, "m", "d", client=cli.CLIClient("agy"))

    def test_codex_judge_reads_output_file(self):
        fake = FakeCLI(self.GRADE)
        with patched(fake):
            j = judge.judge(self.ISSUE, "maint", "draft", client=cli.CLIClient("codex"))
        self.assertEqual(j["score"], 4)
        self.assertNotIn("-m", fake.cmds[0]["cmd"])
        self.assertIn("--output-schema", fake.cmds[0]["cmd"])

    def test_judge_backend_env(self):
        with mock.patch.dict(os.environ, {"ISSUEBOT_BACKEND": "claude-cli", "ISSUEBOT_REPLAY": "off"}):
            os.environ.pop("ISSUEBOT_JUDGE_BACKEND", None)
            self.assertEqual(judge.judge_backend(), "claude-cli")
            os.environ["ISSUEBOT_JUDGE_BACKEND"] = "agy"
            self.assertEqual(judge.judge_backend(), "agy")
            self.assertEqual(agent.make_client(judge.judge_backend()).backend, "agy")

    def test_agreement(self):
        cs = [{"score": s, "wrong": False, "judge2_score": t, "judge2_wrong": False} for s, t in ((5, 5), (2, 3), (4, 4))]
        a = run_eval.agreement(cs + [{"score": 1, "wrong": False}])
        self.assertEqual(a["n"], 3)
        self.assertAlmostEqual(a["exact"], 2 / 3)


class AgentEndToEndTest(unittest.TestCase):
    def test_run_on_fake_claude_cli(self):
        fake = FakeCLI(claude_out(calls(("read_file", {"path": "README.md", "start_line": 1, "end_line": 5})), 0.02),
                       claude_out(calls(("submit", SUB)), 0.03))
        with tempfile.TemporaryDirectory() as d, patched(fake):
            ctx = {"repo": "o/r", "dir": d, "number": 5, "created_at": "2025-01-01T00:00:00Z", "docs": ["docs"]}
            rec = agent.run({"number": 5, "title": "t", "body": "b", "created_at": ctx["created_at"]}, ctx,
                            client=cli.CLIClient("claude-cli"), runs_dir=d)
            line = json.loads(next(Path(d).glob("*.jsonl")).read_text())
        self.assertEqual((rec["label"], rec["steps"], rec["error"], rec["backend"]), ("bug", 2, None, "claude-cli"))
        self.assertEqual([c["name"] for c in rec["tool_calls"]], ["read_file"])
        self.assertAlmostEqual(rec["cost"], 0.05)  # CLI-reported $, not the PRICES estimate
        self.assertEqual(line["backend"], "claude-cli")
        second = fake.arg("-p", 1)
        self.assertIn("called tool read_file (id cli_0_0)", second)
        self.assertIn("TOOL RESULT for cli_0_0", second)


class ReplayKeyTest(unittest.TestCase):
    def test_cache_key_includes_backend_and_api_keys_unchanged(self):
        inner = NS(messages=NS(create=lambda **kw: NS(content=[NS(type="text", text="x")], stop_reason="end_turn",
                                                      usage=NS(input_tokens=1, output_tokens=1), _cost=0.5)))
        with tempfile.TemporaryDirectory() as d:
            kw = dict(model="m", messages=[{"role": "user", "content": "x"}])
            agent.Replay("record", d, inner).create(**kw)
            r = agent.Replay("record", d, inner, backend="claude-cli").create(**kw)
            self.assertEqual(len(list(Path(d).iterdir())), 2)
            self.assertEqual(r._cost, 0.5)  # survives the cache round trip
            self.assertEqual(agent.Replay("replay", d, backend="claude-cli").create(**kw)._cost, 0.5)
        self.assertEqual(agent.cost("m", dict(input=1e6, output=0, cache_read=0, cache_write=0), 0.25), 0.25)


if __name__ == "__main__":
    unittest.main()
