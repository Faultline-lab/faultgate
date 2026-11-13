from faultgate import formats
from faultgate.checks import select
from faultgate.runner import run
from faultgate.trace import Span, Trace
from tests.conftest import EXAMPLES


def _span(span_id: str, parent_id: str | None, name: str, attrs: dict) -> Span:
    return Span(span_id=span_id, parent_id=parent_id, name=name, start_ns=0, attrs=attrs)


def test_trace_model_root_request_model():
    # Root span's gen_ai.request.model takes priority
    t = Trace(
        trace_id="t1",
        agent="agent",
        spans=[
            _span("s1", None, "invoke_agent", {"gen_ai.operation.name": "invoke_agent", "gen_ai.request.model": "gpt-4o"}),
            _span("s2", "s1", "chat", {"gen_ai.response.model": "gpt-4o-mini"}),
        ],
    )
    assert t.model == "gpt-4o"


def test_trace_model_fallback_response_model():
    # Root has no model; falls back to first span with gen_ai.response.model
    # (gen_ai.response.model checked before gen_ai.request.model on that span)
    t = Trace(
        trace_id="t2",
        agent="agent",
        spans=[
            _span("s1", None, "invoke_agent", {"gen_ai.operation.name": "invoke_agent"}),
            _span("s2", "s1", "chat", {"gen_ai.response.model": "claude-3-5-sonnet", "gen_ai.request.model": "claude-3-haiku"}),
        ],
    )
    assert t.model == "claude-3-5-sonnet"


def test_trace_model_fallback_request_model():
    # Root has no model; falls back to first span with request model when no response model
    t = Trace(
        trace_id="t3",
        agent="agent",
        spans=[
            _span("s1", None, "invoke_agent", {"gen_ai.operation.name": "invoke_agent"}),
            _span("s2", "s1", "execute_tool", {"gen_ai.operation.name": "execute_tool"}),
            _span("s3", "s1", "chat", {"gen_ai.request.model": "deepseek-chat"}),
        ],
    )
    assert t.model == "deepseek-chat"


def test_trace_model_none_when_unset():
    # Returns None when no span has a model attribute
    t = Trace(
        trace_id="t4",
        agent="agent",
        spans=[
            _span("s1", None, "invoke_agent", {"gen_ai.operation.name": "invoke_agent"}),
            _span("s2", "s1", "execute_tool", {"gen_ai.tool.name": "search"}),
        ],
    )
    assert t.model is None

    empty = Trace(trace_id="t_empty", agent="agent", spans=[])
    assert empty.model is None


def test_runner_measures_steps_and_model():
    traces = formats.load(EXAMPLES)
    rule_checks = [c for c in select(None) if c.rule]
    report = run(traces, rule_checks)

    for t in traces:
        assert t.trace_id in report.measures
        m = report.measures[t.trace_id]
        # Existing keys remain present
        assert "tokens" in m
        assert "latency_ms" in m
        assert "cost_usd" in m
        # New keys
        assert "steps" in m
        assert isinstance(m["steps"], int)
        assert m["steps"] == len(t.steps)
        assert "model" in m
        assert m["model"] == t.model
        assert m["model"] == "openai/gpt-5.6-luna"


def test_runner_measures_model_none_and_steps_zero():
    trace = Trace(
        trace_id="t_none",
        agent="agent",
        spans=[_span("s1", None, "invoke_agent", {"gen_ai.operation.name": "invoke_agent"})],
    )
    rule_checks = [c for c in select(None) if c.rule]
    report = run([trace], rule_checks)

    m = report.measures["t_none"]
    assert m["steps"] == 0
    assert isinstance(m["steps"], int)
    assert m["model"] is None
    assert "tokens" in m
    assert "latency_ms" in m
    assert "cost_usd" in m
