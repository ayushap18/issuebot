"""Subscription-CLI model backend: client.messages.create on top of `claude -p` (agent + judge) or agy / codex (judge only)."""
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

KINDS = ("claude-cli", "agy", "codex")
JUDGE_ONLY = ("agy", "codex")  # they run their own tools (web search / shell) and no flag turns that off


def _g(b, k):
    return b.get(k) if isinstance(b, dict) else getattr(b, k, None)


def transcript(messages: list) -> str:
    """The conversation as one plain-text transcript. Thinking blocks are skipped."""
    out = []
    for m in messages:
        role, content = m["role"].upper(), m["content"]
        for b in [{"type": "text", "text": content}] if isinstance(content, str) else content:
            t = _g(b, "type")
            if t == "text":
                out.append(f"{role}: {_g(b, 'text')}")
            elif t == "tool_use":
                out.append(f"ASSISTANT called tool {_g(b, 'name')} (id {_g(b, 'id')}) with {json.dumps(_g(b, 'input'))}")
            elif t == "tool_result":
                err = " (error)" if _g(b, "is_error") else ""
                out.append(f"TOOL RESULT for {_g(b, 'tool_use_id')}{err}:\n{_g(b, 'content')}")
    return "\n\n".join(out)


def tool_schema(tools: list) -> dict:
    call = {"type": "object", "additionalProperties": False, "required": ["name", "input_json"],
            "properties": {"name": {"type": "string", "enum": [t["name"] for t in tools]},
                           "input_json": {"type": "string", "description": "the tool arguments as a JSON object string"}}}
    return {"type": "object", "additionalProperties": False, "required": ["tool_calls"],
            "properties": {"tool_calls": {"type": "array", "minItems": 1, "items": call}}}


def tool_prompt(tools: list) -> str:
    specs = "\n".join(f"- {t['name']}: {t.get('description', '')}\n  input_schema: {json.dumps(t['input_schema'])}"
                      for t in tools)
    return ("\n\nYou act only by calling these tools. Reply with tool_calls: one or more {name, input_json} where "
            f"input_json is a JSON object string matching the tool's input_schema.\nTools:\n{specs}\n\n"
            "Continue the conversation below with your next tool call(s).")


def parse_calls(out, tools: list, step: int) -> list | None:
    """tool_calls -> tool_use blocks, or None if anything is malformed (agent.run nudges on a text turn)."""
    by = {t["name"]: t for t in tools}
    try:
        blocks = []
        for i, c in enumerate(out["tool_calls"]):
            inp = json.loads(c["input_json"])
            if c["name"] not in by or not isinstance(inp, dict) or \
                    not set(by[c["name"]]["input_schema"].get("required", [])) <= inp.keys():
                return None
            blocks.append(NS(type="tool_use", id=f"cli_{step}_{i}", name=c["name"], input=inp))
        return blocks or None
    except (KeyError, TypeError, ValueError):
        return None


class CLIClient:
    """Duck-types anthropic.Anthropic().messages.create. Each call is one fresh, tool-less CLI run in an empty dir."""

    def __init__(self, kind: str):
        if kind not in KINDS:
            raise ValueError(f"CLI backend must be one of {', '.join(KINDS)}, got {kind!r}")
        self.kind = self.backend = kind
        self.messages = self

    def create(self, model, system, messages, tools=None, max_tokens=None, output_config=None, **ignored):
        if tools and self.kind in JUDGE_ONLY:
            raise ValueError(f"{self.kind} can only be a judge backend, not the agent: it can browse/run commands on its "
                             "own and could look up the real issue outcome. Use ISSUEBOT_BACKEND=api or claude-cli.")
        fmt = (output_config or {}).get("format") or {}
        schema = tool_schema(tools) if tools else fmt.get("schema")
        prompt = transcript(messages)
        if tools:
            system += tool_prompt(tools)
        out, usage, cost = self._run(model, system, prompt, schema)
        if schema and not tools and not isinstance(out, dict):
            raise RuntimeError(f"{self.kind} returned no structured output: {str(out)[:300]}")
        step = sum(m["role"] == "assistant" for m in messages)
        if tools and (blocks := parse_calls(out, tools, step)):
            content, stop = blocks, "tool_use"
        else:
            content, stop = [NS(type="text", text=out if isinstance(out, str) else json.dumps(out))], "end_turn"
        return NS(content=content, stop_reason=stop, usage=usage, _cost=cost, model=model)

    def _run(self, model, system, prompt, schema):
        timeout = float(os.environ.get("ISSUEBOT_CLI_TIMEOUT") or 300)
        # never hand the API key to the CLI: it must bill the subscription
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        with tempfile.TemporaryDirectory() as d:
            if self.kind == "claude-cli":
                cmd = ["claude", "-p", prompt, "--model", model, "--tools", "", "--system-prompt", system,
                       "--strict-mcp-config", "--setting-sources", "", "--output-format", "json", "--no-session-persistence"]
            else:  # no system-prompt flag: prepend it
                prompt = f"{system}\n\n{prompt}"
            if self.kind == "agy":
                cmd = ["agy", "-p", prompt, "--output-format", "json", "--print-timeout", f"{int(timeout)}s",
                       "--model", os.environ.get("ISSUEBOT_AGY_MODEL") or "gemini-3.8-flash-medium"]
            elif self.kind == "codex":  # ponytail: untested live (quota); codex reports no usage, so tokens and $ are 0
                out_f, m = Path(d) / "out.json", os.environ.get("ISSUEBOT_CODEX_MODEL")
                cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-o", str(out_f),
                       *(["-m", m] if m else []), prompt]
                if schema:
                    (Path(d) / "schema.json").write_text(json.dumps(schema))
                    cmd[-1:-1] = ["--output-schema", str(Path(d) / "schema.json")]
            if schema and self.kind != "codex":
                cmd += ["--json-schema", json.dumps(schema)]
            try:
                p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"{self.kind} timed out after {timeout:.0f}s (ISSUEBOT_CLI_TIMEOUT)") from None
            if p.returncode and (self.kind == "codex" or not p.stdout.strip()):
                raise RuntimeError(f"{self.kind} exited {p.returncode}: {(p.stderr or p.stdout)[-500:]}")
            if self.kind == "codex":
                text = out_f.read_text()
                return (json.loads(text) if schema else text), _usage(), 0.0
        r = json.loads(p.stdout)
        if self.kind == "claude-cli":
            if r.get("is_error"):
                raise RuntimeError(f"claude CLI error: {str(r.get('result'))[:500]}")
            mu = (r.get("modelUsage") or {}).values()
            s = lambda k: sum(x.get(k) or 0 for x in mu)
            return (r.get("structured_output") if schema else r.get("result")), \
                _usage(s("inputTokens"), s("outputTokens"), s("cacheReadInputTokens"), s("cacheCreationInputTokens")), \
                r.get("total_cost_usd")
        if r.get("status") != "SUCCESS":
            raise RuntimeError(f"agy status {r.get('status')}: {str(r.get('response'))[:500]}")
        u = r.get("usage") or {}
        # ponytail: agy reports no price, so $ is 0 (not a fake Haiku-priced number); add a Gemini price table if needed
        return (r.get("structured_output") if schema else r.get("response")), \
            _usage(u.get("input_tokens") or 0, (u.get("output_tokens") or 0) + (u.get("thinking_tokens") or 0),
                   u.get("cache_read_tokens") or 0), 0.0


def _usage(i=0, o=0, cr=0, cw=0):
    return NS(input_tokens=i, output_tokens=o, cache_read_input_tokens=cr, cache_creation_input_tokens=cw)
