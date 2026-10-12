"""Trace formats → the OTel GenAI span model every check reads.

``otlp`` is the native input (``trace.load_otlp``). The others are mapped onto the
same attributes, so every rule, policy and judge runs unchanged:

- ``openinference`` (Arize Phoenix / OpenInference): OTLP whose spans carry
  ``openinference.span.kind``; ``input.value`` / ``output.value`` / ``tool.*``.
- ``langfuse``: a trace export with ``observations`` (``TOOL`` observations are steps).
- ``langsmith``: a list of runs (``run_type`` ``chain`` / ``llm`` / ``tool``).
- ``braintrust``: a list of logged spans (``span_attributes.type`` ``task`` / ``llm`` / ``tool``).

Mapped onto: ``gen_ai.operation.name`` (``invoke_agent`` / ``chat`` / ``execute_tool``),
``gen_ai.input.messages`` / ``gen_ai.output.messages`` on the root,
``gen_ai.tool.name`` / ``gen_ai.tool.call.arguments`` / ``gen_ai.tool.call.result``,
``gen_ai.response.finish_reasons``, ``gen_ai.usage.input_tokens`` / ``output_tokens``,
span status (error → code 2 + message), ``enduser.id`` and ``faultgate.scenario``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from faultgate.trace import Span, Trace, load_otlp

FORMATS = ("auto", "otlp", "openinference", "langfuse", "langsmith", "braintrust")
USER_KEYS = ("enduser.id", "user.id", "user_id", "userId")


# ---------------------------------------------------------------- detection

def _items(doc: Any) -> list[Any]:
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        for k in ("data", "runs", "events", "traces", "spans"):
            if isinstance(doc.get(k), list):
                return doc[k]
        return [doc]
    return []


def detect(doc: Any) -> str:
    if isinstance(doc, dict) and "resourceSpans" in doc:
        for rs in doc["resourceSpans"]:
            for ss in rs.get("scopeSpans", []):
                for s in ss.get("spans", []):
                    if any(a.get("key") == "openinference.span.kind" for a in s.get("attributes", [])):
                        return "openinference"
        return "otlp"
    first = next((x for x in _items(doc) if isinstance(x, dict)), {})
    if "observations" in first:
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
    marker = {"langfuse": ("observations",), "langsmith": ("run_type",), "braintrust": ("span_attributes", "root_span_id")}[fmt]
    if not items or not any(k in items[0] for k in marker):
        raise ValueError(f"not a valid {fmt} export: records lack {' / '.join(marker)}")
    try:
        traces = {"langfuse": _langfuse, "langsmith": _langsmith, "braintrust": _braintrust}[fmt](items)
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"not a valid {fmt} export: {e!r}") from e
    if not traces:
        raise ValueError(f"no {fmt} traces found")
    return traces


# ---------------------------------------------------------------- helpers

def _text(v: Any) -> str:
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
            return _text(v["messages"])
        if len(v) == 1:
            return _text(next(iter(v.values())))
    if isinstance(v, list) and v and all(isinstance(m, dict) for m in v):
        texts = [m.get("content") for m in v if isinstance(m.get("content"), str)]
        user = [m["content"] for m in v if m.get("role") in ("user", "human") and isinstance(m.get("content"), str)]
        return (user or texts or [""])[-1]
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
    t = _text(v)
    return json.dumps([{"role": role, "content": t}], ensure_ascii=False) if t else ""


def _ns(t: Any) -> int:
    """ISO-8601 string, or epoch seconds / ms / µs / ns as a number, → epoch ns. Missing → 0."""
    if t is None or t == "":
        return 0
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        x = float(t)
        for limit, scale in ((1e17, 1), (1e14, 1e3), (1e11, 1e6)):
            if x > limit:
                return int(x * scale)
        return int(round(x * 1e9))
    s = str(t).strip().replace("Z", "+00:00")
    d = datetime.fromisoformat(s)
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
    roots = [s for s in spans if s.parent_id is None]
    root = roots[0] if roots else None
    spans = sorted(spans, key=lambda s: (s is not root, s.start_ns))
    agent = str(root.attrs.get("gen_ai.agent.name", "agent")) if root else "agent"
    return Trace(trace_id=tid, agent=agent, spans=spans)


# ---------------------------------------------------------------- OpenInference

OI_OP = {"AGENT": "invoke_agent", "CHAIN": "invoke_agent", "LLM": "chat", "TOOL": "execute_tool"}


def _openinference(t: Trace) -> Trace:
    spans = []
    for s in t.spans:
        a = dict(s.attrs)
        kind = str(a.get("openinference.span.kind", "")).upper()
        is_root = s.parent_id is None
        if is_root:
            md = a.get("metadata")
            try:
                md = json.loads(md) if isinstance(md, str) else md
            except ValueError:
                md = None
            a.update(_root_attrs(s.name, a.get("input.value"), a.get("output.value"), md, a.get("user.id")))
        elif kind == "TOOL":
            args = a.get("tool.parameters", a.get("input.value"))
            a.update(_tool_attrs(str(a.get("tool.name", s.name)), args, a.get("output.value"), None))
        elif kind in OI_OP:
            a["gen_ai.operation.name"] = OI_OP[kind]
        _usage(a, a.get("llm.token_count.prompt"), a.get("llm.token_count.completion"))
        if a.get("llm.finish_reason"):
            a["gen_ai.response.finish_reasons"] = [str(a["llm.finish_reason"])]
        spans.append(s.model_copy(update={"attrs": a}))
    return _trace(t.trace_id, spans)


# ---------------------------------------------------------------- Langfuse

def _langfuse(items: list[dict[str, Any]]) -> list[Trace]:
    traces = []
    for tr in items:
        tid = str(tr["id"])
        root_id = f"{tid}:root"
        obs = tr.get("observations") or []
        starts = [_ns(o.get("startTime")) for o in obs if o.get("startTime")]
        ends = [_ns(o.get("endTime")) for o in obs if o.get("endTime")]
        root = Span(span_id=root_id, parent_id=None, name=str(tr.get("name") or "trace"),
                    start_ns=(t0 := _ns(tr.get("timestamp")) or min(starts, default=0)),
                    end_ns=t0 + int(tr["latency"] * 1e9) if isinstance(tr.get("latency"), (int, float)) else max(ends, default=0),
                    attrs=_root_attrs(str(tr.get("name") or "agent"), tr.get("input"), tr.get("output"),
                                      tr.get("metadata"), tr.get("userId")))
        spans = [root]
        for o in obs:
            typ = str(o.get("type", "SPAN")).upper()
            err = o.get("statusMessage") if str(o.get("level", "")).upper() == "ERROR" else None
            if typ == "TOOL":
                a = _tool_attrs(str(o.get("name", "tool")), o.get("input"), o.get("output"), err)
            else:
                a = {"gen_ai.operation.name": "chat" if typ == "GENERATION" else typ.lower()}
            u = o.get("usageDetails") or o.get("usage") or {}
            _usage(a, u.get("input"), u.get("output"))
            fr = (o.get("metadata") or {}).get("finish_reason") if isinstance(o.get("metadata"), dict) else None
            if fr:
                a["gen_ai.response.finish_reasons"] = [str(fr)]
            if isinstance((o.get("costDetails") or {}).get("total"), (int, float)):
                a["faultgate.cost_usd"] = float(o["costDetails"]["total"])
            spans.append(Span(span_id=str(o["id"]), parent_id=str(o.get("parentObservationId") or root_id),
                              name=str(o.get("name", typ.lower())), start_ns=_ns(o.get("startTime")),
                              end_ns=_ns(o.get("endTime")), attrs=a, status_code=2 if err else 0,
                              status_message=str(err or "")))
        traces.append(_trace(tid, spans))
    return traces


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
        if r.get("parent_run_id") is None:
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
        span_err = err if (err and rt != "tool") else None
        by_trace.setdefault(tid, []).append(Span(
            span_id=rid, parent_id=str(r["parent_run_id"]) if r.get("parent_run_id") else None,
            name=str(r.get("name", rt)), start_ns=_ns(r.get("start_time")), end_ns=_ns(r.get("end_time")),
            attrs=a, status_code=2 if span_err else 0, status_message=str(span_err or "")))
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
        span_err = err if (err and typ != "tool") else None
        by_trace.setdefault(rid, []).append(Span(
            span_id=sid, parent_id=str(parents[0]) if parents and sid != rid else None, name=name,
            start_ns=_ns(m.get("start")), end_ns=_ns(m.get("end")), attrs=a,
            status_code=2 if span_err else 0, status_message=str(span_err or "")))
    return [_trace(tid, spans) for tid, spans in by_trace.items()]
