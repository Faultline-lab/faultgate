from faultgate.checks import select
from faultgate.runner import run
from faultgate.trace import load_otlp
from tests.conftest import EXAMPLES


def test_report_counts(fake_judge):
    traces = load_otlp(EXAMPLES)
    judge = fake_judge(fire_on={"ABSTENTION": "couldn’t verify", "SEARCH_LOOP": "Steps used: 12"})
    report = run(traces, select(None), judge)
    assert report.judge == "fake"
    assert len(report.results) == 5 * 3
    assert report.fired == {"search_loop": 1, "abstention": 1, "wrong_direction": 0}
    assert len(report.failed_traces) == 2 and not report.passed
    d = report.to_dict()
    assert d["traces"] == 5 and len(d["results"]) == 15 and d["failed_traces"] == report.failed_traces


def test_check_subset(fake_judge):
    traces = load_otlp(EXAMPLES)[:1]
    report = run(traces, select("abstention"), fake_judge())
    assert report.checks == ["abstention"] and report.passed
