"""OTLP JSON (GenAI semantic conventions) → Trace.

One trace = one agent run: a root ``invoke_agent`` span plus child ``chat`` /
``execute_tool`` spans. Content is read from the semconv content attributes
(``gen_ai.input.messages``, ``gen_ai.output.messages``,
``gen_ai.tool.call.arguments``, ``gen_ai.tool.call.result``) with the legacy
``gen_ai.prompt`` / ``gen_ai.completion`` as fallback. A trace with no content
is still loaded — the judge just sees structure only.
"""
from __future__ import annotations

import hashlib
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
    status_code: int = 0


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    tool: str
    arguments: str
    result: str
    span_id: str = ""


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
        p = (_text(r.attrs.get("gen_ai.input.messages"), role="user") or str(r.attrs.get("gen_ai.prompt", ""))) if r else ""
        if p.strip():
            return p
        for s in self.spans:
            p = _text(s.attrs.get("gen_ai.input.messages"), role="user") or str(s.attrs.get("gen_ai.prompt", ""))
            if p.strip():
                return p
        return ""

    @property
    def answer(self) -> str:
        r = self.root
        a = (_text(r.attrs.get("gen_ai.output.messages"), role="assistant") or str(r.attrs.get("gen_ai.completion", ""))) if r else ""
        if a.strip():
            return a
        for s in reversed(self.spans):
            a = _text(s.attrs.get("gen_ai.output.messages"), role="assistant") or str(s.attrs.get("gen_ai.completion", ""))
            if a.strip():
                return a
        return ""

    @property
    def key(self) -> str:
        """Stable scenario key across runs: faultgate.scenario, else gen_ai.agent.id, else a prompt hash."""
        r = self.root
        if r:
            for k in ("faultgate.scenario", "gen_ai.agent.id"):
                if r.attrs.get(k):
                    return str(r.attrs[k])
        return "p-" + hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:12]

    @property
    def termination(self) -> str:
        r = self.root
        return str(r.attrs.get("faultgate.termination", "")) if r else ""

    @property
    def steps(self) -> list[Step]:
        exec_spans = [s for s in self.spans if s.attrs.get("gen_ai.operation.name") == "execute_tool"]
        exec_steps = [
            Step(
                index=i + 1,
                tool=str(s.attrs.get("gen_ai.tool.name", s.name)),
                arguments=str(s.attrs.get("gen_ai.tool.call.arguments", "")),
                result=str(s.attrs.get("gen_ai.tool.call.result", "")),
                span_id=s.span_id,
            )
            for i, s in enumerate(exec_spans)
        ]
        msg_steps = _steps_from_messages(self.spans)
        if not exec_spans:
            return [st for _, st in msg_steps]
        exec_ids = {str(s.attrs["gen_ai.tool.call.id"]) for s in exec_spans if s.attrs.get("gen_ai.tool.call.id")}
        combined = exec_steps + [st for cid, st in msg_steps if cid not in exec_ids]
        span_starts = {s.span_id: s.start_ns for s in self.spans}
        combined.sort(key=lambda st: span_starts.get(st.span_id, 0))
        return [
            Step(index=i + 1, tool=st.tool, arguments=st.arguments, result=st.result, span_id=st.span_id)
            for i, st in enumerate(combined)
        ]

    @property
    def has_content(self) -> bool:
        return bool(self.prompt or self.answer or any(st.arguments or st.result for st in self.steps))

    def render(self, budget: int | None = None) -> str:
        """The state text a judge reads.

        Diagnostic facts first (answer, termination, step count), then the steps.
        With a character ``budget`` (small local judges have a 512-token window),
        steps are compacted and the middle is elided — the first two and the last
        ones survive, because loops are visible at the tail and setup at the head.
        """
        steps = self.steps
        head = [
            f"Agent: {self.agent}",
            f"Prompt: {_trim(self.prompt, 600 if budget else None)}",
            f"Steps used: {len(steps)}",
            f"Termination: {self.termination or 'unknown'}",
            f"Final answer: {_trim(self.answer, 400 if budget else None) or '(none)'}",
        ]
        lines = [f"Step {st.index}: {st.tool} {_trim(st.arguments, 160 if budget else None)} -> {_trim(st.result, 160 if budget else None)}" for st in steps]
        if budget is None:
            return "\n".join(head + lines)
        room = budget - sum(len(h) + 1 for h in head)
        keep_head = lines[:2]
        room -= sum(len(x) + 1 for x in keep_head)
        tail: list[str] = []
        for line in reversed(lines[2:]):
            if room - len(line) - 1 < 24:  # leave space for the elision marker
                break
            tail.insert(0, line)
            room -= len(line) + 1
        omitted = len(lines) - len(keep_head) - len(tail)
        middle = [f"... {omitted} steps omitted ..."] if omitted > 0 else []
        return "\n".join(head + keep_head + middle + tail)


def _trim(s: str, n: int | None) -> str:
    return s if n is None or len(s) <= n else s[: n - 1] + "…"


def _messages(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    return [m for m in raw if isinstance(m, dict)] if isinstance(raw, list) else []


def _format_payload(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                parsed = json.loads(s)
                return _format_payload(parsed)
            except Exception:
                pass
        return val
    if isinstance(val, list):
        if val and all(isinstance(x, dict) and "text" in x for x in val):
            return " ".join(str(x["text"]) for x in val)
        return json.dumps(val, ensure_ascii=False)
    if isinstance(val, dict):
        if val.get("type") == "text" and "text" in val:
            return str(val["text"])
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def _steps_from_messages(spans: list[Span]) -> list[tuple[str, Step]]:
    responses: dict[str, str] = {}
    for s in spans:
        for m in _messages(s.attrs.get("gen_ai.input.messages")):
            for p in m.get("parts", []):
                if isinstance(p, dict) and p.get("type") == "tool_call_response" and p.get("id"):
                    res = _format_payload(p.get("result"))
                    is_err = (
                        p.get("is_error") is True
                        or bool(p.get("error"))
                        or str(p.get("status", "")).lower() in ("error", "failed")
                    )
                    if is_err and not res.startswith("Error: "):
                        res = "Error: " + res
                    responses[str(p["id"])] = res

    steps: list[tuple[str, Step]] = []
    for s in spans:
        for m in _messages(s.attrs.get("gen_ai.output.messages")):
            for p in m.get("parts", []):
                if isinstance(p, dict) and p.get("type") == "tool_call":
                    cid = str(p.get("id", ""))
                    steps.append((
                        cid,
                        Step(
                            index=len(steps) + 1,
                            tool=str(p.get("name", "unknown")),
                            arguments=_format_payload(p.get("arguments")),
                            result=responses.get(cid, ""),
                            span_id=s.span_id,
                        ),
                    ))
    return steps


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
                parts.append(str(p.get("content", p.get("text", ""))))
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


def _status_code(raw: Any) -> int:
    if isinstance(raw, int):
        return raw
    s = str(raw)
    if s == "STATUS_CODE_ERROR":
        return 2
    if s == "STATUS_CODE_OK":
        return 1
    return int(s) if s.isdigit() else 0


def load_otlp(path: str | Path) -> list[Trace]:
    """Parse an OTLP/JSON export into traces, ordered by first span start."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    by_trace: dict[str, list[Span]] = {}
    agents: dict[str, str] = {}
    for rs in doc.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for s in ss.get("spans", []):
                attrs = {a["key"]: _value(a["value"]) for a in s.get("attributes", [])}
                st = s.get("status") or {}
                status_code = _status_code(st.get("code", 0))
                span = Span(
                    span_id=s["spanId"],
                    parent_id=s.get("parentSpanId"),
                    name=s.get("name", ""),
                    start_ns=int(s.get("startTimeUnixNano", 0)),
                    attrs=attrs,
                    status_code=status_code,
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
