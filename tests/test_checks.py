import json

import pytest

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
    assert [c.kind for c in CHECKS.values()] == ["rule", "rule", "rule", "rule", "rule", "rule", "golden", "golden", "judge"]


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


def test_context_overflow_finish_reason():
    from faultgate.checks import context_overflow_rule

    assert not context_overflow_rule(_make_trace([("search", "{}", "ok")])).detected
    for reasons in (["length"], ["stop", "max_tokens"], "length", ["MAX_TOKENS"]):
        v = context_overflow_rule(_make_trace([("search", "{}", "ok")], {"gen_ai.response.finish_reasons": reasons}))
        assert v.detected, reasons
    v = context_overflow_rule(_make_trace([], {"gen_ai.response.finish_reasons": ["length"]}))
    assert v.reason == "span 2 (chat) stopped on finish reason 'length': output cut at the token limit"
    assert not context_overflow_rule(_make_trace([], {"gen_ai.response.finish_reasons": ["stop", "tool_calls"]})).detected


def test_context_overflow_error_text():
    from faultgate.checks import context_overflow_rule

    msgs = [
        "This model's maximum context length is 128000 tokens. However, your messages resulted in 130211 tokens.",
        "Error code: 400 - context_length_exceeded",
        "prompt is too long: 210432 tokens > 200000 maximum",
        "input exceeds the context window",
    ]
    for m in msgs:
        assert context_overflow_rule(_make_trace([("search", "{}", "Error: " + m)])).detected, m
        assert context_overflow_rule(_make_trace([], {"error.type": m})).detected, m
    t = _make_trace([("search", "{}", "ok")])
    t.spans[0].status_message = "context_length_exceeded"
    assert context_overflow_rule(t).detected
    t = _make_trace([])
    t.spans[0].attrs["faultgate.termination"] = "context window exceeded"
    assert context_overflow_rule(t).detected
    # talking about tokens is not an overflow
    benign = [("search", '{"q": "max tokens setting"}', "the context window of GPT-4 is 128k tokens")]
    assert not context_overflow_rule(_make_trace(benign)).detected


def test_context_overflow_zero_on_shipped_traces():
    from pathlib import Path

    from faultgate.checks import context_overflow_rule

    files = [EXAMPLES, *sorted((EXAMPLES.parent / "gate").glob("*_*.json"))]
    traces = [t for f in files if f.name not in ("golden.json", "policy.json") for t in load_otlp(f)]
    assert len(traces) > 100 and not any(context_overflow_rule(t).detected for t in traces)


def test_status_message_and_end_time_are_kept(tmp_path):
    import json

    doc = {"resourceSpans": [{"scopeSpans": [{"spans": [{
        "traceId": "a" * 32, "spanId": "1" * 16, "name": "invoke_agent",
        "startTimeUnixNano": "10", "endTimeUnixNano": "2000000010",
        "status": {"code": 2, "message": "context_length_exceeded"}, "attributes": []}]}]}]}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc))
    (t,) = load_otlp(p)
    assert t.spans[0].status_message == "context_length_exceeded" and t.spans[0].end_ns == 2000000010


def test_context_overflow_provider_forms_and_benign_results():
    from faultgate.checks import context_overflow_rule

    provider_errors = [
        "The input token count (1048577) exceeds the maximum number of tokens allowed (1048576).",  # Gemini
        "Input is too long for requested model.",  # Bedrock
        "This model's maximum context length is 8192 tokens.",  # OpenAI / Azure OpenAI
        "prompt is too long: 210432 tokens > 200000 maximum",  # Anthropic
    ]
    for m in provider_errors:
        assert context_overflow_rule(_make_trace([("llm_call", "{}", "Error: " + m)])).detected, m
        assert context_overflow_rule(_make_trace([], {"error.type": m})).detected, m
    # a normal tool result that quotes the error text is documentation, not an overflow
    benign = [("search", '{"q": "openai errors"}', "maximum context length errors happen when the prompt is too long")]
    assert not context_overflow_rule(_make_trace(benign)).detected
    assert not context_overflow_rule(_make_trace([("read", "{}", '{"text": "context_length_exceeded is returned when"}')])).detected


def test_context_overflow_odd_finish_reason_values():
    from faultgate.checks import context_overflow_rule

    for raw in (123, {"reason": "MAX_TOKENS"}, ["stop", 123], None, []):
        context_overflow_rule(_make_trace([], {"gen_ai.response.finish_reasons": raw}))  # never raises
    assert context_overflow_rule(_make_trace([], {"gen_ai.response.finish_reasons": ["stop", 123, "length"]})).detected


# Client test, 2026-10-13 (tau2 retail): real final answers the old "cannot / unable to" list flagged.
# A support agent reminding the customer of a rule, refusing an unauthenticated request or explaining what
# a tool cannot do has not abstained. Frozen from the client's runs.
SUPPORT_NOT_ABSTAINING = [
    "Both changes are complete. Both orders remain pending, and their item changes cannot be modified again.",
    "I’m unable to verify an account using a username and ZIP code. Please provide the email address on the account.",
    "The order remains pending, but its items can’t be modified again or the order cancelled.",
    "The order status changed to return requested, so I couldn’t submit the exchanges through the available process.",
    "Note: the order details returned by the system show inconsistent item information, so I can’t reliably confirm the updated item options here.",
    "The order cannot be canceled because it has already been marked for return.",
    "I am unable to directly check the return window or escalate to a specialist.",
    "I couldn’t submit the return with PayPal because the return system requires refunds to go to the original payment method.",
]
ABSTAINING = [
    "I couldn’t verify the external auditor for Aster Labs from the available documents—the searches returned no matching source.",
    "It appears that I couldn't locate the specific credit card payment method to process the refund.",
    "I could not find any record of that order.",
    "I was unable to determine who signed the contract.",
    "No relevant documents mention the merger.",
    "The requested information is not available in the knowledge base.",
    "I don't know.",
]


@pytest.mark.parametrize("answer", SUPPORT_NOT_ABSTAINING)
def test_abstention_ignores_support_reminders_and_refusals(answer):
    from faultgate.checks import abstention_rule
    from faultgate.trace import Span, Trace

    root = Span(span_id="r", parent_id=None, name="invoke_agent", start_ns=1,
                attrs={"gen_ai.operation.name": "invoke_agent", "gen_ai.completion": answer})
    assert not abstention_rule(Trace(trace_id="t", agent="a", spans=[root])).detected


@pytest.mark.parametrize("answer", ABSTAINING)
def test_abstention_catches_information_not_found(answer):
    from faultgate.checks import abstention_rule
    from faultgate.trace import Span, Trace

    root = Span(span_id="r", parent_id=None, name="invoke_agent", start_ns=1,
                attrs={"gen_ai.operation.name": "invoke_agent", "gen_ai.completion": answer})
    assert abstention_rule(Trace(trace_id="t", agent="a", spans=[root])).detected


# --- 0.4: outcomes from the user's own evaluator ---

def _keys(path=EXAMPLES):
    return [t.key for t in load_otlp(path)]


def test_outcome_check_is_opt_in_and_reads_the_outcomes_file(tmp_path):
    from faultgate.checks import Outcomes, outcome_rule, select

    assert "outcome" not in [c.name for c in select(None)]  # no file: not in the default set, report unchanged
    p = tmp_path / "outcomes.json"
    keys = _keys()
    p.write_text(json.dumps({keys[0]: True, keys[1]: False}))
    outcomes = Outcomes.load(p)
    checks = select(None, outcomes=outcomes)
    assert checks[-1].name == "outcome" and checks[-1].kind == "outcome"
    traces = load_otlp(EXAMPLES)
    assert not outcome_rule(traces[0], outcomes).detected
    v = outcome_rule(traces[1], outcomes)
    assert v.detected and v.reason == "outcome false in the outcomes file"
    v = outcome_rule(traces[2], outcomes)
    assert not v.detected and v.confidence == 0.0 and v.reason == "no outcome for this scenario"
    assert [c.name for c in select("outcome,policy", outcomes=outcomes)] == ["outcome", "policy"]


@pytest.mark.parametrize("bad", ['["a"]', '{"a": "yes"}', '{"a": 1}', "{not json"])
def test_outcomes_file_must_map_keys_to_booleans(tmp_path, bad):
    from faultgate.checks import Outcomes

    p = tmp_path / "outcomes.json"
    p.write_text(bad)
    with pytest.raises(ValueError):
        Outcomes.load(p)
