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
