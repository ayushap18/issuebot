"""Scripted stand-in for anthropic.Anthropic: no network, no API key."""
from types import SimpleNamespace as NS


def usage(i=100, o=50, cr=0, cw=0):
    return NS(input_tokens=i, output_tokens=o, cache_read_input_tokens=cr, cache_creation_input_tokens=cw)


def msg(*blocks, u=None, stop="tool_use"):
    return NS(content=list(blocks), usage=u or usage(), stop_reason=stop)


def tool(name, inp, id="t1"):
    return NS(type="tool_use", id=id, name=name, input=inp)


def text(t):
    return NS(type="text", text=t)


def submit(label="bug", dup=None, reply="Please add a repro.", conf=0.8, id="s1"):
    return tool("submit", {"label": label, "duplicate_of": dup, "reply": reply, "confidence": conf}, id)


class FakeClient:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []
        self.messages = self

    def create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})  # snapshot: the loop keeps appending
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
