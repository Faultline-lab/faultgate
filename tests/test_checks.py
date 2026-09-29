from faultgate.checks import CHECKS, _empty_result, search_loop_rule
from faultgate.trace import load_otlp
from tests.conftest import EXAMPLES


def test_empty_result_forms():
    assert all(_empty_result(s) for s in ("", "[]", "{}", '{"n_candidates": 0, "candidates": []}', '{"results": []}', '{"count": 0}'))
    assert not any(_empty_result(s) for s in ("hit", '{"n_candidates": 2}', '["a"]', '{"results": ["x"]}', "not json"))


def test_search_loop_rule_on_examples():
    traces = load_otlp(EXAMPLES)
    fired = [search_loop_rule(t).detected for t in traces]
    assert fired == [True, False, False, False, False]  # only s-0074 (12 steps, no answer, step cap)
    assert "cap" in search_loop_rule(traces[0]).reason


def test_rule_needs_no_judge(fake_judge):
    from faultgate.checks import select
    from faultgate.runner import run

    judge = fake_judge()
    report = run(load_otlp(EXAMPLES), select("search_loop"), judge)
    assert judge.calls == [] and report.fired == {"search_loop": 1}


def test_kinds():
    assert [c.kind for c in CHECKS.values()] == ["rule", "rule", "rule", "rule", "rule", "golden", "golden", "judge"]


def test_abstention_rule_on_examples():
    from faultgate.checks import abstention_rule

    fired = [abstention_rule(t).detected for t in load_otlp(EXAMPLES)]
    assert fired == [False, True, False, False, False]  # only s-0020 "I couldn’t verify …"


def _make_trace(steps: list[tuple[str, str, str]], span_attrs: dict | None = None):
    from faultgate.trace import Span, Trace

    spans = [
        Span(
            span_id="root",
            parent_id=None,
            name="invoke_agent",
            start_ns=0,
            attrs={"gen_ai.operation.name": "invoke_agent"},
        )
    ]
    for i, (tool, args, res) in enumerate(steps):
        spans.append(
            Span(
                span_id=f"span_{i}",
                parent_id="root",
                name=f"execute_tool {tool}",
                start_ns=(i + 1) * 100,
                attrs={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": tool,
                    "gen_ai.tool.call.arguments": args,
                    "gen_ai.tool.call.result": res,
                },
            )
        )
    if span_attrs:
        spans.append(
            Span(
                span_id="last",
                parent_id="root",
                name="chat",
                start_ns=len(steps) * 100 + 200,
                attrs=span_attrs,
            )
        )
    return Trace(trace_id="t1", agent="agent", spans=spans)


def test_repeated_call_rule():
    from faultgate.checks import repeated_call_rule

    # Fires on 3 identical call+result (including JSON sort_keys normalization)
    t_fired = _make_trace([
        ("search", '{"q": "cats", "limit": 10}', "found 3 items"),
        ("search", '{"limit": 10, "q": "cats"}', "found 3 items"),
        ("search", '{"q": "cats", "limit": 10}', "found 3 items"),
    ])
    v_fired = repeated_call_rule(t_fired)
    assert v_fired.detected
    assert "Tool 'search'" in v_fired.reason

    # Does not fire on 3 identical calls with different results
    t_diff_res = _make_trace([
        ("search", '{"q": "cats"}', "page 1 results"),
        ("search", '{"q": "cats"}', "page 2 results"),
        ("search", '{"q": "cats"}', "page 3 results"),
    ])
    v_diff = repeated_call_rule(t_diff_res)
    assert not v_diff.detected

    # Does not fire on fewer than 3 calls
    t_few = _make_trace([
        ("search", '{"q": "cats"}', "found 3 items"),
        ("search", '{"q": "cats"}', "found 3 items"),
    ])
    assert not repeated_call_rule(t_few).detected


def test_unrecovered_tool_error_rule():
    from faultgate.checks import unrecovered_tool_error_rule

    # Fires on {"error": ...} with no later success
    t_unrec = _make_trace([
        ("search", '{"q": "cats"}', '{"error": "rate limit exceeded"}'),
    ])
    v_unrec = unrecovered_tool_error_rule(t_unrec)
    assert v_unrec.detected
    assert "Tool 'search' failed" in v_unrec.reason

    # Does not fire when a later same-tool call succeeds
    t_rec = _make_trace([
        ("search", '{"q": "cats"}', '{"error": "rate limit exceeded"}'),
        ("search", '{"q": "cats"}', '{"results": ["cat1", "cat2"]}'),
    ])
    assert not unrecovered_tool_error_rule(t_rec).detected

    # Does not fire on normal output containing the word 'error' (e.g. 'raise ValueError')
    t_word_error = _make_trace([
        ("read_file", '{"file": "foo.py"}', 'def check(x):\n    if not x: raise ValueError("empty")'),
    ])
    assert not unrecovered_tool_error_rule(t_word_error).detected

    # Other structural error variants: non-zero exit_code, <tool_use_error>, starts with Error:
    t_exit = _make_trace([("bash", '{"cmd": "pytest"}', '{"exit_code": 1}')])
    assert unrecovered_tool_error_rule(t_exit).detected

    t_tag = _make_trace([("calc", "{}", "<tool_use_error>division by zero</tool_use_error>")])
    assert unrecovered_tool_error_rule(t_tag).detected

    t_pfx = _make_trace([("calc", "{}", "Traceback (most recent call last):\nZeroDivisionError")])
    assert unrecovered_tool_error_rule(t_pfx).detected

    # Span-level error on last span with no later successful step
    t_span = _make_trace([], span_attrs={"error.type": "ConnectionError"})
    assert unrecovered_tool_error_rule(t_span).detected


def test_unrecovered_tool_error_span_status_code():
    from faultgate.checks import unrecovered_tool_error_rule
    from faultgate.trace import Span, Trace

    t = Trace(
        trace_id="t1",
        agent="agent",
        spans=[
            Span(
                span_id="root",
                parent_id=None,
                name="invoke_agent",
                start_ns=0,
                attrs={"gen_ai.operation.name": "invoke_agent"},
            ),
            Span(
                span_id="tool_span",
                parent_id="root",
                name="execute_tool bash",
                start_ns=100,
                status_code=2,
                attrs={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "bash",
                    "gen_ai.tool.call.arguments": "pytest",
                    "gen_ai.tool.call.result": "failed",
                },
            ),
        ],
    )
    v = unrecovered_tool_error_rule(t)
    assert v.detected
    assert "Span status error: Code 2" in v.reason


def test_unrecovered_tool_error_chat_embedded_is_error(tmp_path):
    import json
    from faultgate.checks import unrecovered_tool_error_rule

    doc = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "a" * 32,
                                "spanId": "1" * 16,
                                "name": "invoke_agent",
                                "startTimeUnixNano": "10",
                                "attributes": [{"key": "gen_ai.operation.name", "value": {"stringValue": "invoke_agent"}}],
                            },
                            {
                                "traceId": "a" * 32,
                                "spanId": "2" * 16,
                                "parentSpanId": "1" * 16,
                                "name": "chat",
                                "startTimeUnixNano": "100",
                                "attributes": [
                                    {
                                        "key": "gen_ai.output.messages",
                                        "value": {"stringValue": json.dumps([{"role": "assistant", "parts": [
                                            {"type": "tool_call", "id": "c1", "name": "fetch", "arguments": "https://example.com"},
                                        ]}])},
                                    },
                                ],
                            },
                            {
                                "traceId": "a" * 32,
                                "spanId": "3" * 16,
                                "parentSpanId": "1" * 16,
                                "name": "chat",
                                "startTimeUnixNano": "200",
                                "attributes": [
                                    {
                                        "key": "gen_ai.input.messages",
                                        "value": {"stringValue": json.dumps([{"role": "user", "parts": [
                                            {"type": "tool_call_response", "id": "c1", "result": "timeout", "is_error": True},
                                        ]}])},
                                    },
                                ],
                            },
                        ]
                    }
                ]
            }
        ]
    }
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc))
    (t,) = load_otlp(p)
    v = unrecovered_tool_error_rule(t)
    assert v.detected
    assert "Tool 'fetch' failed" in v.reason


def test_unrecovered_span_error_middle_unrecovered():
    from faultgate.checks import unrecovered_tool_error_rule
    from faultgate.trace import Span, Trace

    t = Trace(
        trace_id="t1",
        agent="agent",
        spans=[
            Span(span_id="s1", parent_id=None, name="invoke_agent", start_ns=0, attrs={"gen_ai.operation.name": "invoke_agent"}),
            Span(span_id="s2", parent_id="s1", name="chat", start_ns=100, attrs={"error.type": "RateLimitError"}),
            Span(span_id="s3", parent_id="s1", name="chat", start_ns=200, attrs={}),
        ],
    )
    v = unrecovered_tool_error_rule(t)
    assert v.detected
    assert "Unrecovered span error at span 2 of 3: Span error: RateLimitError" in v.reason


def test_unrecovered_span_error_middle_recovered():
    from faultgate.checks import unrecovered_tool_error_rule
    from faultgate.trace import Span, Trace

    t = Trace(
        trace_id="t1",
        agent="agent",
        spans=[
            Span(span_id="s1", parent_id=None, name="invoke_agent", start_ns=0, attrs={"gen_ai.operation.name": "invoke_agent"}),
            Span(span_id="s2", parent_id="s1", name="chat", start_ns=100, attrs={"error.type": "RateLimitError"}),
            Span(
                span_id="s3",
                parent_id="s1",
                name="execute_tool search",
                start_ns=200,
                attrs={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "search",
                    "gen_ai.tool.call.arguments": "cats",
                    "gen_ai.tool.call.result": "found cats",
                },
            ),
        ],
    )
    v = unrecovered_tool_error_rule(t)
    assert not v.detected
