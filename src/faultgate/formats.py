"""Trace formats → the OTel GenAI span model every check reads.

``otlp`` is the native input (``trace.load_otlp``). The others are mapped onto the
same attributes, so every rule, policy and judge runs unchanged:

- ``openinference`` (Arize Phoenix / OpenInference): OTLP whose spans carry
  ``openinference.span.kind``; ``input.value`` / ``output.value`` / ``tool.*``, and the
  flattened ``llm.input_messages.N.message.*`` / ``llm.output_messages.N.message.*``
  (text, tool calls and tool results).
- ``langfuse``: trace objects with ``observations``, or observation rows grouped by
  trace (``GET /api/public/observations`` pages in camelCase, blob exports in
  snake_case). ``TOOL`` observations are steps.
- ``langsmith``: a list of runs (``run_type`` ``chain`` / ``llm`` / ``tool``).
- ``braintrust``: a list of logged spans (``span_attributes.type`` ``task`` / ``llm`` / ``tool``).

Mapped onto: ``gen_ai.operation.name`` (``invoke_agent`` / ``chat`` / ``execute_tool``),
``gen_ai.input.messages`` / ``gen_ai.output.messages``,
``gen_ai.tool.name`` / ``gen_ai.tool.call.arguments`` / ``gen_ai.tool.call.result``,
``gen_ai.response.finish_reasons``, ``gen_ai.usage.input_tokens`` / ``output_tokens``,
span status (any error → code 2 + message; a tool's result also gets ``Error: ``),
``enduser.id`` and ``faultgate.scenario``.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from faultgate.trace import Span, Trace, is_tool_span, load_otlp

FORMATS = ("auto", "otlp", "openinference", "langfuse", "langsmith", "braintrust")
USER_KEYS = ("enduser.id", "user.id", "user_id", "userId")
LANGFUSE_TYPES = {"SPAN", "GENERATION", "EVENT", "TOOL", "AGENT", "CHAIN", "RETRIEVER", "EVALUATOR", "EMBEDDING", "GUARDRAIL"}


# ---------------------------------------------------------------- detection

def _items(doc: Any) -> list[Any]:
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        for k in ("data", "runs", "events", "traces", "spans", "observations"):
            if isinstance(doc.get(k), list):
                return doc[k]
        return [doc]
    return []


def _g(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """First present key: lets one mapping read camelCase and snake_case exports."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _is_langfuse_row(x: dict[str, Any]) -> bool:
    return bool(_g(x, "traceId", "trace_id")) and str(x.get("type", "")).upper() in LANGFUSE_TYPES and "run_type" not in x


def detect(doc: Any) -> str:
    if isinstance(doc, dict) and "resourceSpans" in doc:
        for rs in doc["resourceSpans"]:
            for ss in rs.get("scopeSpans", []):
                for s in ss.get("spans", []):
                    if any(a.get("key") == "openinference.span.kind" for a in s.get("attributes", [])):
                        return "openinference"
        return "otlp"
    first = next((x for x in _items(doc) if isinstance(x, dict)), {})
    if "observations" in first or _is_langfuse_row(first):
        return "langfuse"
    if "run_type" in first:
        return "langsmith"
    if "span_attributes" in first or "root_span_id" in first:
        return "braintrust"
    raise ValueError("unrecognised trace format; expected OTLP/JSON, OpenInference, Langfuse, LangSmith or Braintrust "
                     "(pass --format to force one)")


def load(path: str | Path, fmt: str = "auto") -> list[Trace]:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; available: {', '.join(FORMATS)}")
    if fmt == "otlp":
        return load_otlp(path)
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if fmt == "auto":
        fmt = detect(doc)
    if fmt == "otlp":
        return load_otlp(path)
    if fmt == "openinference":
        return [_openinference(t) for t in load_otlp(path)]
    items = [x for x in _items(doc) if isinstance(x, dict)]
    ok = {"langfuse": lambda x: "observations" in x or _is_langfuse_row(x),
          "langsmith": lambda x: "run_type" in x,
          "braintrust": lambda x: "span_attributes" in x or "root_span_id" in x}[fmt]
    if not items or not ok(items[0]):
        raise ValueError(f"not a valid {fmt} export: the records do not have its fields")
    try:
        traces = {"langfuse": _langfuse, "langsmith": _langsmith, "braintrust": _braintrust}[fmt](items)
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"not a valid {fmt} export: {e!r}") from e
    if not traces:
        raise ValueError(f"no {fmt} traces found")
    return traces


# ---------------------------------------------------------------- helpers

def _text(v: Any, role: str = "user") -> str:
    """Plain text from a prompt/answer payload: a string, a {input|output|...: str} wrapper, or chat messages."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("input", "output", "query", "question", "prompt", "answer", "content", "text", "result"):
            if isinstance(v.get(k), str):
                return v[k]
        if isinstance(v.get("messages"), list):
            return _text(v["messages"], role)
        if len(v) == 1:
            return _text(next(iter(v.values())), role)
    if isinstance(v, list) and v and all(isinstance(m, dict) for m in v):
        texts = [m.get("content") for m in v if isinstance(m.get("content"), str)]
        roles = ("user", "human") if role == "user" else ("assistant", "ai")
        mine = [m["content"] for m in v if m.get("role") in roles and isinstance(m.get("content"), str)]
        return (mine or texts or [""])[-1]
    return json.dumps(v, ensure_ascii=False)


def _payload(v: Any) -> str:
    """Tool arguments / results as the string a step carries (JSON for structures)."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict) and len(v) == 1 and next(iter(v)) in ("input", "output"):
        return _payload(next(iter(v.values())))
    return json.dumps(v, ensure_ascii=False)


def _msgs(role: str, v: Any) -> str:
    t = _text(v, role)
    return json.dumps([{"role": role, "content": t}], ensure_ascii=False) if t else ""


NUMERIC = re.compile(r"^\s*\d+(?:\.\d+)?\s*$")


def _ns(t: Any) -> int:
    """ISO-8601 string, or epoch seconds / ms / µs / ns as a number or numeric string, → epoch ns. Missing → 0."""
    if t is None or t == "":
        return 0
    if isinstance(t, str) and NUMERIC.match(t):
        t = float(t) if "." in t else int(t)
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        x = float(t)
        for limit, scale in ((1e17, 1), (1e14, 1e3), (1e11, 1e6)):
            if x > limit:
                return int(t) * int(scale) if isinstance(t, int) else int(x * scale)
        return int(round(x * 1e9))
    d = datetime.fromisoformat(str(t).strip().replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return int(round(d.timestamp() * 1e9))


def _root_attrs(name: str, prompt: Any, answer: Any, metadata: Any, extra_user: Any = None) -> dict[str, Any]:
    a: dict[str, Any] = {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": name}
    if _msgs("user", prompt):
        a["gen_ai.input.messages"] = _msgs("user", prompt)
    if _msgs("assistant", answer):
        a["gen_ai.output.messages"] = _msgs("assistant", answer)
    md = metadata if isinstance(metadata, dict) else {}
    if md.get("scenario"):
        a["faultgate.scenario"] = str(md["scenario"])
    user = extra_user or next((md[k] for k in USER_KEYS if md.get(k)), None)
    if user:
        a["enduser.id"] = str(user)
    return a


def _tool_attrs(name: str, args: Any, result: Any, error: Any) -> dict[str, Any]:
    res = _payload(result)
    if error:
        res = "Error: " + (str(error) if not res else f"{error} {res}")
    return {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name,
            "gen_ai.tool.call.arguments": _payload(args), "gen_ai.tool.call.result": res}


def _usage(a: dict[str, Any], inp: Any, out: Any) -> None:
    if isinstance(inp, (int, float)) and not isinstance(inp, bool):
        a["gen_ai.usage.input_tokens"] = int(inp)
    if isinstance(out, (int, float)) and not isinstance(out, bool):
        a["gen_ai.usage.output_tokens"] = int(out)


def _trace(tid: str, spans: list[Span]) -> Trace:
    """Order spans root first. Root = the parentless agent/invoke span with content, else the earliest parentless span."""
    roots = [s for s in spans if s.parent_id is None]

    def rank(s: Span) -> tuple:
        agentish = s.attrs.get("gen_ai.operation.name") == "invoke_agent"
        content = bool(s.attrs.get("gen_ai.input.messages") or s.attrs.get("gen_ai.output.messages"))
        return (not agentish, not content, s.start_ns, s.span_id)

    root = min(roots, key=rank) if roots else None
    spans = sorted(spans, key=lambda s: (s is not root, s.start_ns, s.span_id))
    agent = str(root.attrs.get("gen_ai.agent.name", root.name)) if root else "agent"
    return Trace(trace_id=tid, agent=agent, spans=spans)


# ---------------------------------------------------------------- OpenInference

OI_OP = {"AGENT": "invoke_agent", "CHAIN": "invoke_agent", "LLM": "chat", "TOOL": "execute_tool"}
OI_MSG = re.compile(r"^llm\.(input|output)_messages\.(\d+)\.message\.(.+)$")
OI_CALL = re.compile(r"^tool_calls\.(\d+)\.tool_call\.(.+)$")


def _oi_messages(attrs: dict[str, Any], direction: str) -> list[dict[str, Any]]:
    """Rebuild flattened llm.<direction>_messages.N.message.* attributes into GenAI messages with parts."""
    msgs: dict[int, dict[str, Any]] = {}
    for k, v in attrs.items():
        m = OI_MSG.match(k)
        if not m or m.group(1) != direction:
            continue
        msg = msgs.setdefault(int(m.group(2)), {"calls": {}})
        rest = m.group(3)
        c = OI_CALL.match(rest)
        if c:
            msg["calls"].setdefault(int(c.group(1)), {})[c.group(2)] = v
        else:
            msg[rest] = v
    out = []
    for _, msg in sorted(msgs.items()):
        role = str(msg.get("role", "assistant"))
        parts: list[dict[str, Any]] = []
        if role == "tool" and msg.get("tool_call_id"):
            parts.append({"type": "tool_call_response", "id": str(msg["tool_call_id"]), "result": msg.get("content", "")})
        elif msg.get("content"):
            parts.append({"type": "text", "content": str(msg["content"])})
        for _, call in sorted(msg["calls"].items()):
            parts.append({"type": "tool_call", "id": str(call.get("id", "")), "name": str(call.get("function.name", "unknown")),
                          "arguments": call.get("function.arguments", "")})
        out.append({"role": role, "parts": parts})
    return out


def _openinference(t: Trace) -> Trace:
    # native OTel execute_tool spans next to OpenInference LLM spans (an app that traces its own tools and
    # instruments the LLM client) are tool spans too: the calls in the LLM messages are the same calls
    has_tool_spans = any(str(s.attrs.get("openinference.span.kind", "")).upper() == "TOOL" or is_tool_span(s) for s in t.spans)
    spans = []
    for s in t.spans:
        a = dict(s.attrs)
        kind = str(a.get("openinference.span.kind", "")).upper()
        if s.parent_id is None:
            md = a.get("metadata")
            try:
                md = json.loads(md) if isinstance(md, str) else md
            except ValueError:
                md = None
            # a native GenAI root keeps its own attributes; the mapping only fills what is missing
            a = {**_root_attrs(s.name, a.get("input.value"), a.get("output.value"), md, a.get("user.id")), **a}
        elif kind == "TOOL":
            args = a.get("tool.parameters", a.get("input.value"))
            a.update(_tool_attrs(str(a.get("tool.name", s.name)), args, a.get("output.value"), s.status_message or None))
        elif kind in OI_OP:
            a["gen_ai.operation.name"] = OI_OP[kind]
        if kind == "LLM":
            ins, outs = _oi_messages(a, "input"), _oi_messages(a, "output")
            if has_tool_spans:  # tool spans are the steps; keep only text so calls are not counted twice
                ins = [{**m, "parts": [p for p in m["parts"] if p["type"] == "text"]} for m in ins]
                outs = [{**m, "parts": [p for p in m["parts"] if p["type"] == "text"]} for m in outs]
            if ins:
                a["gen_ai.input.messages"] = json.dumps(ins, ensure_ascii=False)
            if outs:
                a["gen_ai.output.messages"] = json.dumps(outs, ensure_ascii=False)
        _usage(a, a.get("llm.token_count.prompt"), a.get("llm.token_count.completion"))
        if a.get("llm.finish_reason"):
            a["gen_ai.response.finish_reasons"] = [str(a["llm.finish_reason"])]
        spans.append(s.model_copy(update={"attrs": a}))
    return _trace(t.trace_id, spans)


# ---------------------------------------------------------------- Langfuse

def _lf_observation(o: dict[str, Any], root_id: str) -> Span:
    typ = str(o.get("type", "SPAN")).upper()
    level = str(o.get("level", "")).upper()
    err = _g(o, "statusMessage", "status_message") if level == "ERROR" else None
    err = err or ("error" if level == "ERROR" else None)
    if typ == "TOOL":
        a = _tool_attrs(str(o.get("name", "tool")), o.get("input"), o.get("output"), err)
    else:
        a = {"gen_ai.operation.name": "chat" if typ == "GENERATION" else typ.lower()}
        if typ == "GENERATION":
            if _msgs("user", o.get("input")):
                a["gen_ai.input.messages"] = _msgs("user", o.get("input"))
            if _msgs("assistant", o.get("output")):
                a["gen_ai.output.messages"] = _msgs("assistant", o.get("output"))
    u = _g(o, "usageDetails", "usage_details", "usage", default={}) or {}
    _usage(a, _g(u, "input", "promptTokens", "input_tokens"), _g(u, "output", "completionTokens", "output_tokens"))
    md = o.get("metadata") if isinstance(o.get("metadata"), dict) else {}
    if md.get("finish_reason"):
        a["gen_ai.response.finish_reasons"] = [str(md["finish_reason"])]
    cost = _g(o, "costDetails", "cost_details", default={}) or {}
    if isinstance(cost.get("total"), (int, float)):
        a["faultgate.cost_usd"] = float(cost["total"])
    return Span(span_id=str(o["id"]), parent_id=str(_g(o, "parentObservationId", "parent_observation_id") or root_id),
                name=str(o.get("name", typ.lower())), start_ns=_ns(_g(o, "startTime", "start_time")),
                end_ns=_ns(_g(o, "endTime", "end_time")), attrs=a, status_code=2 if err else 0, status_message=str(err or ""))


def _lf_trace(tid: str, tr: dict[str, Any], obs: list[dict[str, Any]]) -> Trace:
    root_id = f"{tid}:root"
    starts = [_ns(_g(o, "startTime", "start_time")) for o in obs if _g(o, "startTime", "start_time")]
    ends = [_ns(_g(o, "endTime", "end_time")) for o in obs if _g(o, "endTime", "end_time")]
    t0 = _ns(tr.get("timestamp")) or min(starts, default=0)
    latency = tr.get("latency")
    end = t0 + int(latency * 1e9) if isinstance(latency, (int, float)) and not isinstance(latency, bool) else max(ends, default=0)
    gens = sorted((o for o in obs if str(o.get("type", "")).upper() == "GENERATION"), key=lambda o: _ns(_g(o, "startTime", "start_time")))
    prompt = tr.get("input") if tr.get("input") is not None else (gens[0].get("input") if gens else None)
    answer = tr.get("output") if tr.get("output") is not None else (gens[-1].get("output") if gens else None)
    name = str(tr.get("name") or "agent")
    root = Span(span_id=root_id, parent_id=None, name=name, start_ns=t0, end_ns=end,
                attrs=_root_attrs(name, prompt, answer, tr.get("metadata"), _g(tr, "userId", "user_id")))
    return _trace(tid, [root] + [_lf_observation(o, root_id) for o in obs])


def _langfuse(items: list[dict[str, Any]]) -> list[Trace]:
    if "observations" in items[0]:
        return [_lf_trace(str(tr["id"]), tr, tr.get("observations") or []) for tr in items]
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for o in items:
        by_trace.setdefault(str(_g(o, "traceId", "trace_id")), []).append(o)
    return [_lf_trace(tid, {}, obs) for tid, obs in by_trace.items()]


# ---------------------------------------------------------------- LangSmith

def _ls_finish(outputs: Any) -> str | None:
    try:
        g = outputs["generations"][0][0]
    except (KeyError, IndexError, TypeError):
        return None
    info = g.get("generation_info") or {}
    meta = ((g.get("message") or {}).get("kwargs") or {}).get("response_metadata") or {}
    return info.get("finish_reason") or meta.get("finish_reason") or meta.get("stop_reason")


def _langsmith(items: list[dict[str, Any]]) -> list[Trace]:
    by_trace: dict[str, list[Span]] = {}
    for r in items:
        rid = str(r["id"])
        tid = str(r.get("trace_id") or rid)
        rt = str(r.get("run_type", "chain"))
        err = r.get("error")
        if r.get("parent_run_id") is None and rt != "llm":
            md = ((r.get("extra") or {}).get("metadata")) or {}
            a = _root_attrs(str(r.get("name", "agent")), r.get("inputs"), r.get("outputs"), md)
        elif rt == "tool":
            a = _tool_attrs(str(r.get("name", "tool")), r.get("inputs"), r.get("outputs"), err)
        else:
            a = {"gen_ai.operation.name": "chat" if rt == "llm" else rt}
            fr = _ls_finish(r.get("outputs"))
            if fr:
                a["gen_ai.response.finish_reasons"] = [str(fr)]
        _usage(a, r.get("prompt_tokens"), r.get("completion_tokens"))
        if isinstance(r.get("total_cost"), (int, float)):
            a["faultgate.cost_usd"] = float(r["total_cost"])
        by_trace.setdefault(tid, []).append(Span(
            span_id=rid, parent_id=str(r["parent_run_id"]) if r.get("parent_run_id") else None,
            name=str(r.get("name", rt)), start_ns=_ns(r.get("start_time")), end_ns=_ns(r.get("end_time")),
            attrs=a, status_code=2 if err else 0, status_message=str(err or "")))
    return [_trace(tid, spans) for tid, spans in by_trace.items()]


# ---------------------------------------------------------------- Braintrust

def _braintrust(items: list[dict[str, Any]]) -> list[Trace]:
    by_trace: dict[str, list[Span]] = {}
    for e in items:
        sid = str(e.get("span_id") or e["id"])
        rid = str(e.get("root_span_id") or sid)
        sa = e.get("span_attributes") or {}
        typ = str(sa.get("type", "")).lower()
        name = str(sa.get("name", typ or "span"))
        parents = e.get("span_parents") or []
        m = e.get("metrics") or {}
        md = e.get("metadata") if isinstance(e.get("metadata"), dict) else {}
        err = e.get("error")
        if sid == rid or not parents:
            a = _root_attrs(name, e.get("input"), e.get("output"), md)
        elif typ == "tool":
            a = _tool_attrs(name, e.get("input"), e.get("output"), err)
        else:
            a = {"gen_ai.operation.name": "chat" if typ == "llm" else (typ or "span")}
            if md.get("finish_reason"):
                a["gen_ai.response.finish_reasons"] = [str(md["finish_reason"])]
        _usage(a, m.get("prompt_tokens"), m.get("completion_tokens"))
        by_trace.setdefault(rid, []).append(Span(
            span_id=sid, parent_id=str(parents[0]) if parents and sid != rid else None, name=name,
            start_ns=_ns(m.get("start")), end_ns=_ns(m.get("end")), attrs=a,
            status_code=2 if err else 0, status_message=str(err or "")))
    return [_trace(tid, spans) for tid, spans in by_trace.items()]
