import json

import pytest

from faultgate import cli
from faultgate.checks import Policy, context_overflow_rule, policy_rule, unrecovered_tool_error_rule
from faultgate.formats import FORMATS, detect, load
from tests.conftest import EXAMPLES

FMT = EXAMPLES.parent / "formats"
NAMES = ["openinference", "langfuse", "langsmith", "braintrust"]


@pytest.mark.parametrize("name", NAMES)
def test_each_format_normalises_to_the_same_run(name):
    """The four example exports describe one run; every format must yield the same prompt, answer and steps."""
    path = FMT / f"{name}.json"
    assert detect(json.loads(path.read_text(encoding="utf-8"))) == name
    (t,) = load(path)
    assert t.prompt == "Refund order o-7 for customer C-123"
    assert t.answer == "Refunded $120.00 to C-123."
    assert t.key == "refund-o7"
    assert [(s.tool, json.loads(s.arguments)) for s in t.steps] == [
        ("lookup_order", {"order_id": "o-7"}),
        ("issue_refund", {"customer_id": "C-123", "amount": 120}),
    ]
    assert json.loads(t.steps[0].result) == {"total": 120}
    assert t.root.end_ns - t.root.start_ns == 4_500_000_000
    assert t.usage == {"input_tokens": 812, "output_tokens": 40}


@pytest.mark.parametrize("name", NAMES)
def test_rules_and_policy_run_on_every_format(name):
    (t,) = load(FMT / f"{name}.json")
    pol = Policy.load(EXAMPLES.parent / "policy_support.json")
    # compliant run: allowlisted tools, refund under the limit, lookup before refund, and the vendor's
    # user id mapped to enduser.id (identity would fail closed without it)
    v = policy_rule(t, pol)
    assert not v.detected, v.reason
    assert t.root.attrs["enduser.id"] == "C-123"


def test_otlp_is_unchanged():
    assert detect(json.loads(EXAMPLES.read_text(encoding="utf-8"))) == "otlp"
    assert len(load(EXAMPLES)) == 5


def test_errors_and_cutoffs_map_to_span_status_and_finish_reasons(tmp_path):
    runs = [
        {"id": "a", "trace_id": "t", "parent_run_id": None, "run_type": "chain", "name": "agent",
         "inputs": {"input": "q"}, "outputs": None, "start_time": "2026-10-01T10:00:00", "end_time": "2026-10-01T10:00:09"},
        {"id": "b", "trace_id": "t", "parent_run_id": "a", "run_type": "tool", "name": "fetch",
         "inputs": {"url": "x"}, "outputs": None, "error": "TimeoutError: upstream timed out",
         "start_time": "2026-10-01T10:00:01", "end_time": "2026-10-01T10:00:02"},
        {"id": "c", "trace_id": "t", "parent_run_id": "a", "run_type": "llm", "name": "llm",
         "outputs": {"generations": [[{"generation_info": {"finish_reason": "length"}}]]},
         "start_time": "2026-10-01T10:00:03", "end_time": "2026-10-01T10:00:08"},
    ]
    p = tmp_path / "ls.json"
    p.write_text(json.dumps(runs))
    (t,) = load(p)
    assert t.steps[0].result.startswith("Error: ") and "timed out" in t.steps[0].result
    assert unrecovered_tool_error_rule(t).detected
    assert context_overflow_rule(t).detected


def test_unknown_format_and_forced_format(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps([{"foo": 1}]))
    with pytest.raises(ValueError, match="unrecognised trace format"):
        load(p)
    with pytest.raises(ValueError, match="unknown format"):
        load(p, "zipkin")
    assert set(FORMATS) == {"auto", "otlp", "openinference", "langfuse", "langsmith", "braintrust"}


def test_cli_check_reads_other_formats(capsys):
    for name in NAMES:
        assert cli.main(["check", str(FMT / f"{name}.json"), "--check", "repeated_call"]) == 0
    assert cli.main(["check", str(FMT / "langsmith.json"), "--format", "langfuse", "--check", "repeated_call"]) == 2


# --- precheck ---

def test_precheck_clean_files(capsys):
    for f in [EXAMPLES, *(FMT / f"{n}.json" for n in NAMES)]:
        assert cli.main(["precheck", str(f)]) == 0, f
    out = capsys.readouterr().out
    assert "precheck: OK" in out and "format braintrust" in out


def _otlp(spans):
    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}


def _span(sid, parent=None, start="10", end="20", **kw):
    d = {"traceId": "a" * 32, "spanId": sid, "name": kw.pop("name", "s"), "startTimeUnixNano": start, "endTimeUnixNano": end,
         "attributes": kw.pop("attributes", [])}
    if parent:
        d["parentSpanId"] = parent
    return d


def test_precheck_finds_structural_errors(tmp_path, capsys):
    bad = _otlp([
        _span("1" * 16),
        _span("2" * 16, parent="1" * 16, start="0"),                  # missing start
        _span("3" * 16, parent="1" * 16, start="50", end="40"),       # ends before it starts
        _span("3" * 16, parent="1" * 16, start="60", end="70"),       # duplicate id
        _span("4" * 16, parent="f" * 16, start="80", end="90"),       # parent not in trace (warning)
    ])
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    assert cli.main(["precheck", str(p)]) == 2
    out = capsys.readouterr().out
    for msg in ("missing start time", "ends before it starts", "duplicate span id", "parent", "no prompt"):
        assert msg in out, msg
    assert "precheck: FAILED" in out


def test_precheck_no_root_and_parse_errors(tmp_path, capsys):
    # real exports often omit the root: every span hangs off one parent that is not in the file.
    # Checks still run on such traces, so it is a warning, reported once per trace.
    p = tmp_path / "noroot.json"
    p.write_text(json.dumps(_otlp([_span("1" * 16, parent="9" * 16), _span("2" * 16, parent="9" * 16, start="30", end="40")])))
    assert cli.main(["precheck", str(p)]) == 0
    out = capsys.readouterr().out
    assert "no root span in the export" in out and out.count("not in the export") == 1
    p.write_text("{not json")
    assert cli.main(["precheck", str(p)]) == 2
    assert "cannot parse" in capsys.readouterr().out


def test_precheck_sample_limits_traces(tmp_path, capsys):
    assert cli.main(["precheck", str(EXAMPLES), "--sample", "2"]) == 0
    assert "checked 2 of 5 traces" in capsys.readouterr().out


# --- review round 1 (Codex): real vendor shapes beyond the happy path ---

def _write(tmp_path, doc, name="x.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


def test_langfuse_observation_rows_api_page_and_blob_export(tmp_path):
    """GET /api/public/observations pages (camelCase) and blob exports (snake_case) are rows grouped by trace."""
    api_page = {"data": [
        {"id": "g1", "traceId": "T1", "type": "GENERATION", "name": "chat", "parentObservationId": None,
         "startTime": "2026-10-01T10:00:00Z", "endTime": "2026-10-01T10:00:01Z",
         "input": [{"role": "user", "content": "Where is order o-7?"}], "output": {"role": "assistant", "content": "Shipped."},
         "usageDetails": {"input": 10, "output": 3}},
        {"id": "t1", "traceId": "T1", "type": "TOOL", "name": "lookup_order", "parentObservationId": None,
         "startTime": "2026-10-01T10:00:00.5Z", "endTime": "2026-10-01T10:00:00.7Z",
         "input": {"order_id": "o-7"}, "output": {"status": "shipped"}, "level": "ERROR", "statusMessage": "timeout"},
    ], "meta": {"page": 1, "totalPages": 1}}
    blob = [
        {"id": "g1", "trace_id": "T2", "type": "GENERATION", "name": "chat", "parent_observation_id": None,
         "start_time": "2026-10-01T10:00:00Z", "end_time": "2026-10-01T10:00:01Z",
         "input": [{"role": "user", "content": "Where is order o-8?"}], "output": "Delivered.",
         "usage_details": {"input": 5, "output": 2}},
        {"id": "t1", "trace_id": "T2", "type": "TOOL", "name": "lookup_order", "parent_observation_id": None,
         "start_time": "2026-10-01T10:00:00.5Z", "end_time": "2026-10-01T10:00:00.7Z",
         "input": {"order_id": "o-8"}, "output": {"status": "delivered"}},
    ]
    for doc, prompt, answer, usage in ((api_page, "Where is order o-7?", "Shipped.", {"input_tokens": 10, "output_tokens": 3}),
                                       (blob, "Where is order o-8?", "Delivered.", {"input_tokens": 5, "output_tokens": 2})):
        p = _write(tmp_path, doc)
        assert detect(doc) == "langfuse"
        (t,) = load(p)
        assert (t.prompt, t.answer, t.usage) == (prompt, answer, usage)
        assert [s.tool for s in t.steps] == ["lookup_order"]
    (t,) = load(_write(tmp_path, api_page))
    tool = next(s for s in t.spans if s.name == "lookup_order")
    assert tool.status_code == 2 and tool.status_message == "timeout" and t.steps[0].result.startswith("Error: timeout")


def _oi_span(sid, parent, kind, attrs, start):
    a = [{"key": "openinference.span.kind", "value": {"stringValue": kind}}]
    a += [{"key": k, "value": ({"intValue": str(v)} if isinstance(v, int) else {"stringValue": v})} for k, v in attrs.items()]
    d = {"traceId": "e" * 32, "spanId": sid, "name": kind.lower(), "startTimeUnixNano": str(start),
         "endTimeUnixNano": str(start + 5), "attributes": a}
    if parent:
        d["parentSpanId"] = parent
    return d


def test_openinference_llm_messages_and_embedded_tool_calls(tmp_path):
    """Root without input.value: prompt/answer come from flattened llm.*_messages; tool calls inside them become steps."""
    root = "a" * 16
    doc = {"resourceSpans": [{"scopeSpans": [{"spans": [
        _oi_span(root, None, "AGENT", {}, 100),
        _oi_span("b" * 16, root, "LLM", {
            "llm.input_messages.0.message.role": "system", "llm.input_messages.0.message.content": "Be brief.",
            "llm.input_messages.1.message.role": "user", "llm.input_messages.1.message.content": "Weather in Paris?",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_1",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"city": "Paris"}',
            "llm.finish_reason": "tool_calls"}, 110),
        _oi_span("c" * 16, root, "LLM", {
            "llm.input_messages.0.message.role": "tool", "llm.input_messages.0.message.tool_call_id": "call_1",
            "llm.input_messages.0.message.content": '{"temp_c": 18}',
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "18 °C and clear."}, 120),
    ]}]}]}
    (t,) = load(_write(tmp_path, doc))
    assert t.prompt == "Weather in Paris?" and t.answer == "18 °C and clear."
    assert [(s.tool, json.loads(s.arguments), json.loads(s.result)) for s in t.steps] == [("get_weather", {"city": "Paris"}, {"temp_c": 18})]


def test_root_selection_is_order_independent(tmp_path):
    runs = [
        {"id": "orphan", "trace_id": "t", "parent_run_id": None, "run_type": "llm", "name": "stray",
         "start_time": "2026-10-01T10:00:05", "end_time": "2026-10-01T10:00:06"},
        {"id": "root", "trace_id": "t", "parent_run_id": None, "run_type": "chain", "name": "agent",
         "inputs": {"input": "q"}, "outputs": {"output": "a"}, "start_time": "2026-10-01T10:00:00", "end_time": "2026-10-01T10:00:09"},
    ]
    for order in (runs, runs[::-1]):
        (t,) = load(_write(tmp_path, order))
        assert t.agent == "agent" and t.root.span_id == "root" and t.prompt == "q"


@pytest.mark.parametrize("value", [1790848800, 1790848800.0, "1790848800", "1790848800000", "1790848800000000",
                                   "1790848800000000000", 1790848800000, "2026-10-01T10:00:00Z", "2026-10-01T10:00:00"])
def test_timestamps_numbers_numeric_strings_and_iso(value):
    from faultgate.formats import _ns

    assert abs(_ns(value) - (1790848800_000000000 if "T" not in str(value) else _ns("2026-10-01T10:00:00+00:00"))) < 1_000


def test_tool_errors_set_span_status_in_langsmith_and_braintrust(tmp_path):
    ls = [{"id": "r", "trace_id": "t", "parent_run_id": None, "run_type": "chain", "name": "a", "inputs": {"input": "q"},
           "start_time": "2026-10-01T10:00:00", "end_time": "2026-10-01T10:00:09"},
          {"id": "x", "trace_id": "t", "parent_run_id": "r", "run_type": "tool", "name": "fetch", "inputs": {"u": 1},
           "error": "boom", "start_time": "2026-10-01T10:00:01", "end_time": "2026-10-01T10:00:02"}]
    bt = [{"span_id": "r", "root_span_id": "r", "span_parents": [], "span_attributes": {"name": "a", "type": "task"},
           "input": "q", "metrics": {"start": 1790848800, "end": 1790848809}},
          {"span_id": "x", "root_span_id": "r", "span_parents": ["r"], "span_attributes": {"name": "fetch", "type": "tool"},
           "input": {"u": 1}, "error": "boom", "metrics": {"start": 1790848801, "end": 1790848802}}]
    for doc in (ls, bt):
        (t,) = load(_write(tmp_path, doc))
        tool = next(s for s in t.spans if s.name == "fetch")
        assert tool.status_code == 2 and tool.status_message == "boom" and t.steps[0].result == "Error: boom"


# --- client test, 2026-10-13: what precheck passed was not what check read ---

def _attr(k, v):
    return {"key": k, "value": {"stringValue": v}}


def _names_only_doc():
    """invoke_agent / execute_tool named by span name, with the README's attributes but no gen_ai.operation.name."""
    root = _span("1" * 16, name="invoke_agent", start="10", end="90",
                 attributes=[_attr("gen_ai.prompt", "cancel #W1"), _attr("gen_ai.completion", "done")])
    tools = [_span(f"{i + 2:016d}", parent="1" * 16, name="execute_tool", start=str(20 + i), end=str(21 + i),
                   attributes=[_attr("gen_ai.tool.name", "get_order_details"), _attr("gen_ai.tool.call.arguments", '{"order_id": "#W1"}'),
                               _attr("gen_ai.tool.call.result", '{"status": "pending"}')]) for i in range(3)]
    return _otlp([root, *tools])


def test_tool_spans_named_by_span_name_are_read(tmp_path, capsys):
    p = tmp_path / "names.json"
    p.write_text(json.dumps(_names_only_doc()))
    (t,) = load(p)
    assert [s.tool for s in t.steps] == ["get_order_details"] * 3
    from faultgate.checks import repeated_call_rule
    assert repeated_call_rule(t).detected  # was 0 steps: every structural check was blind
    assert cli.main(["precheck", str(p)]) == 0
    out = capsys.readouterr().out
    assert "3 tool steps read" in out and "recognised by span name only" in out


def test_precheck_warns_on_tool_attributes_it_does_not_read(tmp_path, capsys):
    doc = _names_only_doc()
    for s in doc["resourceSpans"][0]["scopeSpans"][0]["spans"][1:]:
        s["name"] = "tool:get_order_details"
    p = tmp_path / "unread.json"
    p.write_text(json.dumps(doc))
    assert cli.main(["precheck", str(p)]) == 0
    out = capsys.readouterr().out
    assert "0 tool steps read" in out and "3 span(s) carry tool attributes but are not read" in out


def test_mixed_native_and_openinference_export_counts_each_call_once(tmp_path, capsys):
    """An app that traces its own tools (native GenAI spans) and instruments the LLM client with OpenInference.
    Auto-detect picks openinference; the LLM's tool-call messages are the same calls as the tool spans."""
    root = _span("1" * 16, name="invoke_agent", start="10", end="90",
                 attributes=[_attr("gen_ai.operation.name", "invoke_agent"), _attr("gen_ai.agent.name", "retail"),
                             _attr("gen_ai.prompt", "cancel #W1"), _attr("gen_ai.completion", "done")])
    llm = _span("2" * 16, parent="1" * 16, name="completion", start="20", end="30", attributes=[
        _attr("openinference.span.kind", "LLM"),
        _attr("llm.output_messages.0.message.role", "assistant"),
        _attr("llm.output_messages.0.message.tool_calls.0.tool_call.id", "call_1"),
        _attr("llm.output_messages.0.message.tool_calls.0.tool_call.function.name", "get_order_details"),
        _attr("llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments", '{"order_id": "#W1"}')])
    tool = _span("3" * 16, parent="1" * 16, name="execute_tool", start="40", end="50", attributes=[
        _attr("gen_ai.operation.name", "execute_tool"), _attr("gen_ai.tool.name", "get_order_details"),
        _attr("gen_ai.tool.call.arguments", '{"order_id": "#W1"}'), _attr("gen_ai.tool.call.result", '{"status": "pending"}')])
    p = tmp_path / "mixed.json"
    p.write_text(json.dumps(_otlp([root, llm, tool])))
    assert detect(json.loads(p.read_text())) == "openinference"
    (auto,), (otlp,) = load(p), load(p, "otlp")
    assert [(s.tool, s.arguments, s.result) for s in auto.steps] == [(s.tool, s.arguments, s.result) for s in otlp.steps]
    assert len(auto.steps) == 1 and auto.agent == "retail"
    assert cli.main(["precheck", str(p)]) == 0
    assert "mixed export" in capsys.readouterr().out
