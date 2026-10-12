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
