"""OTLP JSON (GenAI semantic conventions) → Trace.

One trace = one agent run: a root ``invoke_agent`` span plus child ``chat`` /
``execute_tool`` spans. Content is read from the semconv content attributes
(``gen_ai.input.messages``, ``gen_ai.output.messages``,
``gen_ai.tool.call.arguments``, ``gen_ai.tool.call.result``) with the legacy
``gen_ai.prompt`` / ``gen_ai.completion`` as fallback. A trace with no content
is still loaded — the judge just sees structure only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class Span(BaseModel):
    model_config = ConfigDict(extra="forbid")
    span_id: str
    parent_id: str | None
    name: str
    start_ns: int
    attrs: dict[str, Any]


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    tool: str
    arguments: str
    result: str


class Trace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trace_id: str
    agent: str
    spans: list[Span]

    @property
    def root(self) -> Span | None:
        for s in self.spans:
            if s.attrs.get("gen_ai.operation.name") == "invoke_agent" or s.parent_id is None:
                return s
        return None

    @property
    def prompt(self) -> str:
        r = self.root
        if not r:
            return ""
        return _text(r.attrs.get("gen_ai.input.messages"), role="user") or str(r.attrs.get("gen_ai.prompt", ""))

    @property
    def answer(self) -> str:
        r = self.root
        if not r:
            return ""
        return _text(r.attrs.get("gen_ai.output.messages"), role="assistant") or str(r.attrs.get("gen_ai.completion", ""))

    @property
    def termination(self) -> str:
        r = self.root
        return str(r.attrs.get("faultgate.termination", "")) if r else ""

    @property
    def steps(self) -> list[Step]:
        out = []
        for s in self.spans:
            if s.attrs.get("gen_ai.operation.name") != "execute_tool":
                continue
            out.append(Step(
                index=len(out) + 1,
                tool=str(s.attrs.get("gen_ai.tool.name", s.name)),
                arguments=str(s.attrs.get("gen_ai.tool.call.arguments", "")),
                result=str(s.attrs.get("gen_ai.tool.call.result", "")),
            ))
        return out

    @property
    def has_content(self) -> bool:
        return bool(self.prompt or self.answer or any(st.arguments or st.result for st in self.steps))

    def render(self) -> str:
        """The state text a judge reads. Mirrors FAULTLINE P5's trace_context."""
        steps = self.steps
        lines = [f"Agent: {self.agent}", f"Prompt: {self.prompt}", f"Steps used: {len(steps)}"]
        if self.termination:
            lines.append(f"Termination: {self.termination}")
        for st in steps:
            lines.append(f"Step {st.index}: {st.tool} {st.arguments} -> {st.result}")
        lines.append(f"Final answer: {self.answer}")
        return "\n".join(lines)


def _text(messages: Any, role: str) -> str:
    """Pull the text parts for ``role`` out of a gen_ai.*.messages value."""
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except ValueError:
            return messages
    if not isinstance(messages, list):
        return ""
    parts = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != role:
            continue
        for p in m.get("parts", []):
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("content", "")))
        if "content" in m and not m.get("parts"):
            parts.append(str(m["content"]))
    return "\n".join(parts)


def _value(v: dict[str, Any]) -> Any:
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "arrayValue" in v:
        return [_value(x) for x in v["arrayValue"].get("values", [])]
    return v


def load_otlp(path: str | Path) -> list[Trace]:
    """Parse an OTLP/JSON export into traces, ordered by first span start."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    by_trace: dict[str, list[Span]] = {}
    agents: dict[str, str] = {}
    for rs in doc.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for s in ss.get("spans", []):
                attrs = {a["key"]: _value(a["value"]) for a in s.get("attributes", [])}
                span = Span(
                    span_id=s["spanId"],
                    parent_id=s.get("parentSpanId"),
                    name=s.get("name", ""),
                    start_ns=int(s.get("startTimeUnixNano", 0)),
                    attrs=attrs,
                )
                by_trace.setdefault(s["traceId"], []).append(span)
                if "gen_ai.agent.name" in attrs:
                    agents[s["traceId"]] = str(attrs["gen_ai.agent.name"])
    traces = [
        Trace(trace_id=tid, agent=agents.get(tid, "agent"), spans=sorted(spans, key=lambda x: x.start_ns))
        for tid, spans in by_trace.items()
    ]
    traces.sort(key=lambda t: t.spans[0].start_ns if t.spans else 0)
    return traces
