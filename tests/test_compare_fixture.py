"""tests/fixtures/make_compare_runs.py: the seeded synthetic runs behave as compare calibration assumes."""
from __future__ import annotations

import json
import statistics

import pytest

from faultgate import formats, runner
from faultgate.checks import Policy, select
from tests.fixtures.make_compare_runs import POLICY, make_run, write_side


def _detected(rep, tid):
    return [r.check for r in rep.results if r.trace_id == tid and r.verdict.detected]


def test_shape_clean_rate_and_one_failure(tmp_path):
    paths = write_side(tmp_path, "base", calls=6.0, latency_ms=5000.0, cost_usd=0.04)
    traces = formats.load(paths[0])
    assert [t.key for t in traces] == [f"s{i:02d}" for i in range(30)]
    assert {t.root.attrs["gen_ai.request.model"] for t in traces} == {"model-a"}
    clean = 0
    for p in paths:
        traces = formats.load(p)
        rep = runner.run(traces, select(None))
        assert len(rep.trace_ids) == 30
        clean += sum(rep.clean_by_scenario.values())
        for tid in rep.failed_traces:  # a failed run fires exactly one check
            assert _detected(rep, tid) == ["unrecovered_tool_error"]
    assert clean / 90 == pytest.approx(0.9, abs=0.08)


def test_population_means(tmp_path):
    # 30 scenarios from a [0.5, 2] spread land ~25% off by chance; the knobs set the population mean
    traces = formats.load(write_side(tmp_path, "wide", runs=1, n=200)[0])
    rep = runner.run(traces, select(None))
    means = {k: statistics.mean(rep.measures[t.trace_id][k] for t in traces)
             for k in ("latency_ms", "cost_usd", "tokens")}
    assert statistics.mean(len(t.steps) for t in traces) == pytest.approx(6.0, rel=0.2)
    assert means["latency_ms"] == pytest.approx(5000.0, rel=0.2)
    assert means["cost_usd"] == pytest.approx(0.04, rel=0.2)
    assert means["tokens"] == pytest.approx(2500.0, rel=0.2)


def test_scales_with_the_asked_value(tmp_path):
    a = runner.run(formats.load(write_side(tmp_path, "a", runs=1, n=10)[0]), select(None)).measures
    b = runner.run(formats.load(write_side(tmp_path, "b", runs=1, n=10, cost_usd=0.04 * 1.3,
                                          latency_ms=5000.0 * 1.3)[0]), select(None)).measures
    assert list(a) == list(b)  # same seeds, same trace ids
    for tid in a:
        assert b[tid]["cost_usd"] / a[tid]["cost_usd"] == pytest.approx(1.3)
        assert b[tid]["latency_ms"] / a[tid]["latency_ms"] == pytest.approx(1.3)


def test_zero_calls(tmp_path):
    paths = write_side(tmp_path, "z", runs=1, n=8, p_clean=0.5, zero_calls=True)
    traces = formats.load(paths[0])
    assert [len(t.steps) for t in traces] == [0] * 8
    rep = runner.run(traces, select(None))
    for tid in rep.failed_traces:
        assert _detected(rep, tid) == ["unrecovered_tool_error"]
    assert len(rep.failed_traces) < 8  # at least one failed zero-call trace, so the loop above is not vacuous


def test_policy_violation_fires_only_policy(tmp_path):
    policy = Policy.from_dict(POLICY)
    bad = formats.load(write_side(tmp_path, "v", runs=1, p_clean=1.0, policy_violation=True)[0])
    good = formats.load(write_side(tmp_path, "g", runs=1, p_clean=1.0)[0])
    assert [len(t.steps) for t in bad] == [len(t.steps) for t in good]  # call count unchanged
    assert runner.run(good, select(None, policy=policy)).fired_on("policy") == []
    rep = runner.run(bad, select(None, policy=policy))
    assert [rep.scenario_of[t] for t in rep.fired_on("policy")] == ["s00"]
    assert rep.failed_traces == rep.fired_on("policy")  # nothing else fires


def test_no_cost_attribute(tmp_path):
    rep = runner.run(formats.load(write_side(tmp_path, "t", runs=1, n=5, cost_usd=None)[0]), select(None))
    assert {rep.measures[tid]["cost_usd"] for tid in rep.trace_ids} == {None}
    assert all(rep.measures[tid]["tokens"] for tid in rep.trace_ids)  # tokens only


def test_deterministic():
    assert json.dumps(make_run(3, 7)) == json.dumps(make_run(3, 7))
    assert json.dumps(make_run(3, 7)) != json.dumps(make_run(3, 8))
    assert json.dumps(make_run(3, 7)) != json.dumps(make_run(4, 7))
