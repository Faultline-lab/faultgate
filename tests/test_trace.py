import json

from faultgate.trace import load_otlp
from tests.conftest import EXAMPLES


def test_load_examples():
    traces = load_otlp(EXAMPLES)
    assert len(traces) == 5
    assert all(t.has_content for t in traces)
    loop = traces[0]
    assert loop.agent == "faultline-agent"
    assert len(loop.steps) == 12
    assert loop.steps[0].tool == "search"
    assert loop.answer == ""
    assert "step" in loop.termination.lower() or loop.termination


def test_render_shape():
    t = load_otlp(EXAMPLES)[2]
    text = t.render()
    assert text.startswith("Agent: faultline-agent\nPrompt: ")
    assert "Steps used: 2" in text  # 3 P03 steps, 2 of them tool calls
    assert "Step 1: search {" in text
    assert "Final answer: The internal codename of Girona Foundry is **Zinc-4524**." in text


def test_structure_only_trace(tmp_path):
    doc = {"resourceSpans": [{"scopeSpans": [{"spans": [
        {"traceId": "a" * 32, "spanId": "b" * 16, "name": "invoke_agent", "startTimeUnixNano": "1",
         "attributes": [{"key": "gen_ai.operation.name", "value": {"stringValue": "invoke_agent"}}]},
        {"traceId": "a" * 32, "spanId": "c" * 16, "parentSpanId": "b" * 16, "name": "execute_tool search",
         "startTimeUnixNano": "2", "attributes": [
             {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
             {"key": "gen_ai.tool.name", "value": {"stringValue": "search"}},
             {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "41"}}]},
    ]}]}]}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc))
    (t,) = load_otlp(p)
    assert not t.has_content
    assert t.steps[0].tool == "search" and t.steps[0].arguments == ""
    assert t.spans[1].attrs["gen_ai.usage.input_tokens"] == 41


def test_legacy_prompt_attrs(tmp_path):
    doc = {"resourceSpans": [{"scopeSpans": [{"spans": [
        {"traceId": "a" * 32, "spanId": "b" * 16, "name": "invoke_agent", "startTimeUnixNano": "1",
         "attributes": [{"key": "gen_ai.prompt", "value": {"stringValue": "who?"}},
                        {"key": "gen_ai.completion", "value": {"stringValue": "him"}}]}]}]}]}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc))
    (t,) = load_otlp(p)
    assert t.prompt == "who?" and t.answer == "him"


def test_render_budget_keeps_answer_and_tail():
    t = load_otlp(EXAMPLES.parent / "gate" / "baseline_glm_run1.json")[0]  # 24 steps, ~4.7k chars
    full, small = t.render(), t.render(1800)
    assert len(full) > 4000 and len(small) <= 1800
    assert "Final answer:" in small.split("\nStep 1: search")[0]  # answer is in the header, before any step
    assert "steps omitted" in small
    assert "Step 24:" in small and "Step 1:" in small and "Step 2:" in small
    assert "Step 12:" not in small


def test_render_budget_noop_when_it_fits():
    t = load_otlp(EXAMPLES)[2]
    assert "omitted" not in t.render(1800) and t.render(1800).count("Step ") == 2


def _chat_trace_doc():
    """Synthetic P18/Exgentic-style chat trace with message-embedded tool calls and responses."""
    return {
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
                                        "key": "gen_ai.input.messages",
                                        "value": {"stringValue": json.dumps([{"role": "user", "parts": [{"type": "text", "content": "What is the secret?"}]}])},
                                    },
                                    {
                                        "key": "gen_ai.output.messages",
                                        "value": {"stringValue": json.dumps([{"role": "assistant", "parts": [
                                            {"type": "tool_call", "id": "c1", "name": "search", "arguments": {"query": "secret"}},
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
                                            {"type": "tool_call_response", "id": "c1", "result": '[{"type": "text", "text": "secret is 42"}]'},
                                        ]}])},
                                    },
                                    {
                                        "key": "gen_ai.output.messages",
                                        "value": {"stringValue": json.dumps([{"role": "assistant", "parts": [
                                            {"type": "tool_call", "id": "c2", "name": "verify", "arguments": "42"},
                                        ]}])},
                                    },
                                ],
                            },
                            {
                                "traceId": "a" * 32,
                                "spanId": "4" * 16,
                                "parentSpanId": "1" * 16,
                                "name": "chat",
                                "startTimeUnixNano": "300",
                                "attributes": [
                                    {
                                        "key": "gen_ai.input.messages",
                                        "value": {"stringValue": json.dumps([{"role": "user", "parts": [
                                            {"type": "tool_call_response", "id": "c2", "result": "verified"},
                                        ]}])},
                                    },
                                    {
                                        "key": "gen_ai.output.messages",
                                        "value": {"stringValue": json.dumps([{"role": "assistant", "parts": [
                                            {"type": "text", "content": "The answer is 42."},
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


def test_chat_trace_prompt_and_answer(tmp_path):
    p = tmp_path / "chat_trace.json"
    p.write_text(json.dumps(_chat_trace_doc()))
    (t,) = load_otlp(p)
    assert t.prompt == "What is the secret?"
    assert t.answer == "The answer is 42."


def test_chat_trace_steps_and_responses(tmp_path):
    p = tmp_path / "chat_trace.json"
    p.write_text(json.dumps(_chat_trace_doc()))
    (t,) = load_otlp(p)
    assert len(t.steps) == 2
    assert t.steps[0].index == 1
    assert t.steps[0].tool == "search"
    assert t.steps[0].arguments == '{"query": "secret"}'
    assert t.steps[0].result == "secret is 42"
    assert t.steps[1].index == 2
    assert t.steps[1].tool == "verify"
    assert t.steps[1].arguments == "42"
    assert t.steps[1].result == "verified"


def test_chat_trace_missing_response(tmp_path):
    doc = _chat_trace_doc()
    # Remove span 4 so c2 has no response
    doc["resourceSpans"][0]["scopeSpans"][0]["spans"].pop()
    p = tmp_path / "chat_unanswered.json"
    p.write_text(json.dumps(doc))
    (t,) = load_otlp(p)
    assert len(t.steps) == 2
    assert t.steps[1].tool == "verify"
    assert t.steps[1].result == ""


def test_hybrid_trace_keeps_embedded_calls_without_duplicate(tmp_path):
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
                                "name": "execute_tool bash",
                                "startTimeUnixNano": "100",
                                "attributes": [
                                    {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
                                    {"key": "gen_ai.tool.name", "value": {"stringValue": "bash"}},
                                    {"key": "gen_ai.tool.call.id", "value": {"stringValue": "c1"}},
                                    {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": "echo span_c1"}},
                                    {"key": "gen_ai.tool.call.result", "value": {"stringValue": "span_res_c1"}},
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
                                        "key": "gen_ai.output.messages",
                                        "value": {"stringValue": json.dumps([
                                            {"role": "assistant", "parts": [
                                                {"type": "tool_call", "id": "c1", "name": "bash", "arguments": "echo msg_c1"},
                                                {"type": "tool_call", "id": "c2", "name": "read", "arguments": "file.txt"},
                                            ]}
                                        ])},
                                    },
                                    {
                                        "key": "gen_ai.input.messages",
                                        "value": {"stringValue": json.dumps([
                                            {"role": "user", "parts": [
                                                {"type": "tool_call_response", "id": "c1", "result": "msg_res_c1"},
                                                {"type": "tool_call_response", "id": "c2", "result": "file_contents"},
                                            ]}
                                        ])},
                                    },
                                ],
                            },
                        ]
                    }
                ]
            }
        ]
    }
    p = tmp_path / "hybrid.json"
    p.write_text(json.dumps(doc))
    (t,) = load_otlp(p)
    assert len(t.steps) == 2
    assert t.steps[0].index == 1
    assert t.steps[0].tool == "bash"
    assert t.steps[0].arguments == "echo span_c1"
    assert t.steps[0].result == "span_res_c1"
    assert t.steps[1].index == 2
    assert t.steps[1].tool == "read"
    assert t.steps[1].arguments == "file.txt"
    assert t.steps[1].result == "file_contents"


def test_hybrid_chronological_ordering(tmp_path):
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
                                        "value": {"stringValue": json.dumps([
                                            {"role": "assistant", "parts": [
                                                {"type": "tool_call", "id": "c1", "name": "search", "arguments": "cats"},
                                            ]}
                                        ])},
                                    },
                                    {
                                        "key": "gen_ai.input.messages",
                                        "value": {"stringValue": json.dumps([
                                            {"role": "user", "parts": [
                                                {"type": "tool_call_response", "id": "c1", "result": "found cats"},
                                            ]}
                                        ])},
                                    },
                                ],
                            },
                            {
                                "traceId": "a" * 32,
                                "spanId": "3" * 16,
                                "parentSpanId": "1" * 16,
                                "name": "execute_tool write",
                                "startTimeUnixNano": "200",
                                "attributes": [
                                    {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
                                    {"key": "gen_ai.tool.name", "value": {"stringValue": "write"}},
                                    {"key": "gen_ai.tool.call.id", "value": {"stringValue": "c2"}},
                                    {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": "cats.txt"}},
                                    {"key": "gen_ai.tool.call.result", "value": {"stringValue": "written"}},
                                ],
                            },
                        ]
                    }
                ]
            }
        ]
    }
    p = tmp_path / "hybrid_chrono.json"
    p.write_text(json.dumps(doc))
    (t,) = load_otlp(p)
    assert len(t.steps) == 2
    assert [st.tool for st in t.steps] == ["search", "write"]
    assert t.steps[0].index == 1
    assert t.steps[0].tool == "search"
    assert t.steps[0].arguments == "cats"
    assert t.steps[1].index == 2
    assert t.steps[1].tool == "write"
    assert t.steps[1].arguments == "cats.txt"
