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


# ---------------------------------------------------------------- M2/M3: compare.py

import json

import pytest

from faultgate.compare import load_budget, load_prices


def _json_file(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_load_budget_valid(tmp_path):
    b = load_budget(_json_file(tmp_path, "budget.json",
                               {"correctness": {"budget": 0.0},
                                "speed": {"budget": 0.25, "gate": True}}))
    assert b["correctness"]["budget"] == 0.0
    assert b["speed"] == {"budget": 0.25, "gate": True}


@pytest.mark.parametrize("doc", [
    {"latency": {"budget": 0.1}},
    {"cost": {"budget": -0.1}},
    {"cost": {"budget": float("nan")}},
    {"cost": {"budget": float("inf")}},
    {"cost": {"budget": True}},
    {"cost": {"gate": "yes"}},
])
def test_load_budget_rejects(tmp_path, doc):
    with pytest.raises(ValueError, match=r"^budget\.json:"):
        load_budget(_json_file(tmp_path, "budget.json", doc))


def test_load_prices_valid(tmp_path):
    p = load_prices(_json_file(tmp_path, "prices.json",
                               {"a": {"input_per_m": 3.0, "output_per_m": 15.0},
                                "b": {"gpu_hour_usd": 2.4, "concurrency": 16}}))
    assert p["a"]["input_per_m"] == 3.0
    assert p["b"]["concurrency"] == 16


@pytest.mark.parametrize("entry", [
    {"input_per_m": 1.0},
    {"gpu_hour_usd": 1.0},
    {"input_per_m": 1.0, "output_per_m": 2.0, "extra": 1},
    {"gpu_hour_usd": 1.0, "concurrency": 0},
    {"input_per_m": float("nan"), "output_per_m": 1.0},
    {"gpu_hour_usd": True, "concurrency": 1},
])
def test_load_prices_rejects(tmp_path, entry):
    with pytest.raises(ValueError, match=r"^prices\.json:"):
        load_prices(_json_file(tmp_path, "prices.json", {"m": entry}))


def test_upper_bound_95():
    from faultgate.compare import upper_bound_95

    assert upper_bound_95(0, 100) == 0.0295
    assert upper_bound_95(0, 1) == 0.95
    assert upper_bound_95(0, 0) is None
    assert upper_bound_95(1, 0) is None
    assert upper_bound_95(1, 100) is None
    assert upper_bound_95(3, 5) is None


# ---------------------------------------------------------------- compare(): M2 + M3

from pathlib import Path

from faultgate import runner
from faultgate.checks import Policy
from faultgate.compare import DIMENSIONS, compare
from tests.fixtures.make_compare_runs import POLICY, make_run, write_side

CHECKS = select(None, Policy.from_dict(POLICY))


def _run_side(tmp_path, prefix, runs=3, **kw):
    out = []
    for p in write_side(tmp_path, prefix, runs=runs, **kw):
        traces = formats.load(p)
        out.append((runner.run(traces, CHECKS), traces))
    return out


def _base(tmp_path):
    return _run_side(tmp_path, "base", world_seed=1, run_seed=1)


def _worse(res):
    return {d for d in DIMENSIONS if res["dimensions"][d]["label"] == "WORSE"}


def _edit_scenarios(path, fn):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    spans = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
    drop = set()
    for s in spans:
        for a in s.get("attributes", []):
            if a["key"] == "faultgate.scenario":
                old = a["value"]["stringValue"]
                new = fn(old)
                if new is None:
                    drop.add(s["traceId"])
                else:
                    a["value"]["stringValue"] = new
    doc["resourceSpans"][0]["scopeSpans"][0]["spans"] = [s for s in spans if s["traceId"] not in drop]
    Path(path).write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")


def _load_paths(paths):
    out = []
    for p in paths:
        traces = formats.load(p)
        out.append((runner.run(traces, CHECKS), traces))
    return out


def test_compare_same_distribution_no_dimension_worse(tmp_path):
    base = _base(tmp_path)
    cand = _run_side(tmp_path, "cand", world_seed=1, run_seed=2)
    res = compare(base, cand, resamples=400, seed=0)
    assert _worse(res) == set()
    assert res["verdict"] in ("SHIP", "REVIEW")
    assert res["exit_code"] == 0
    assert res["ci_level"] == pytest.approx(0.99)
    assert res["n"] == 30
    assert res["runs"] == {"baseline": 3, "candidate": 3}


@pytest.mark.parametrize("dimension,kw", [
    ("cost", {"cost_usd": 0.04 * 1.3}),
    ("speed", {"latency_ms": 5000.0 * 1.3}),
    ("approach", {"calls": 6.0 * 1.3}),
])
def test_compare_injected_regression_labels_only_that_dimension(tmp_path, dimension, kw):
    base = _base(tmp_path)
    cand = _run_side(tmp_path, "cand", world_seed=1, run_seed=1, **kw)
    res = compare(base, cand, resamples=400, seed=0)
    assert res["dimensions"][dimension]["label"] == "WORSE"
    assert _worse(res) == {dimension}
    assert res["verdict"] == "BLOCK"
    assert res["exit_code"] == 1


def test_compare_correctness_drop_blocks(tmp_path):
    base = _base(tmp_path)
    cand = _run_side(tmp_path, "cand", world_seed=1, run_seed=1, p_clean=0.6)
    res = compare(base, cand, resamples=400, seed=0)
    assert res["dimensions"]["correctness"]["label"] == "WORSE"
    assert _worse(res) <= {"correctness", "consistency"}
    assert res["verdict"] == "BLOCK"
    assert res["exit_code"] == 1


def test_compare_cheaper_is_better(tmp_path):
    base = _base(tmp_path)
    cand = _run_side(tmp_path, "cand", world_seed=1, run_seed=1, cost_usd=0.04 * 0.7)
    res = compare(base, cand, resamples=400, seed=0)
    assert res["dimensions"]["cost"]["label"] == "BETTER"
    assert res["verdict"] in ("SHIP", "REVIEW")
    assert res["exit_code"] == 0


def test_compare_one_policy_violation_blocks(tmp_path):
    base = _base(tmp_path)
    paths = write_side(tmp_path, "cand", runs=3, world_seed=1, run_seed=1)
    (tmp_path / "cand_run0.json").write_text(
        json.dumps(make_run(1, 1000, policy_violation=True), sort_keys=True), encoding="utf-8")
    cand = _load_paths(paths)
    res = compare(base, cand, resamples=400, seed=0)
    assert res["safety"] == {"violations": 1, "runs": 90, "upper_bound_95": None, "label": "BLOCK"}
    assert res["verdict"] == "BLOCK"
    assert res["exit_code"] == 1
    assert res["policy_violations"] == [{"run": 1, "scenario": "s00"}]


def test_compare_zero_baseline_uses_absolute_change(tmp_path):
    base = _run_side(tmp_path, "base", world_seed=1, run_seed=1, zero_calls=True)
    cand = _run_side(tmp_path, "cand", world_seed=1, run_seed=1)
    res = compare(base, cand, resamples=400, seed=0)
    approach = res["dimensions"]["approach"]
    assert "absolute change" in approach["note"]
    assert approach["baseline"] == 0.0
    assert approach["delta"] is not None


def test_compare_cost_units_mismatch_is_na_with_warning(tmp_path):
    base = _base(tmp_path)
    cand = _run_side(tmp_path, "cand", world_seed=1, run_seed=1, cost_usd=None)
    res = compare(base, cand, resamples=400, seed=0)
    assert res["dimensions"]["cost"]["label"] == "n/a"
    assert res["dimensions"]["cost"]["gates"] is False
    assert any("cost n/a" in w and "candidate" in w for w in res["warnings"])
    assert res["verdict"] in ("SHIP", "REVIEW")
    assert res["exit_code"] == 0


def test_compare_tokens_only_warns_and_prices_resolve_to_usd(tmp_path):
    base = _run_side(tmp_path, "base", world_seed=1, run_seed=1, cost_usd=None, model="model-a")
    cand = _run_side(tmp_path, "cand", world_seed=1, run_seed=1, cost_usd=None, model="model-b")
    res = compare(base, cand, resamples=400, seed=0)
    assert res["dimensions"]["cost"]["unit"] == "tokens (no price)"
    assert any("tokenizer" in w and "model-a" in w and "model-b" in w for w in res["warnings"])

    prices = load_prices(_json_file(tmp_path, "prices.json",
                                    {"model-a": {"input_per_m": 1.0, "output_per_m": 2.0},
                                     "model-b": {"input_per_m": 1.0, "output_per_m": 2.0}}))
    res = compare(base, cand, prices=prices, resamples=400, seed=0)
    assert res["dimensions"]["cost"]["unit"] == "usd"
    assert res["dimensions"]["cost"]["label"] in ("SAME", "BETTER", "INCONCLUSIVE")


def test_compare_missing_scenario_counts_failed(tmp_path):
    # the candidate files are byte-identical to the baseline runs, then s05 is dropped from one run,
    # so the only possible difference is s05 counted as failed (0) in that candidate run
    base = _base(tmp_path)
    paths = write_side(tmp_path, "cand", runs=3, world_seed=1, run_seed=1)
    _edit_scenarios(paths[0], lambda s: None if s == "s05" else s)
    cand = _load_paths(paths)
    res = compare(base, cand, resamples=400, seed=0)
    s05 = [r.clean_by_scenario["s05"] for r, _ in base]
    assert res["missing"] == [{"run": 1, "scenario": "s05"}]
    assert any("counted as failed" in w for w in res["warnings"])
    assert res["dimensions"]["correctness"]["candidate"] == pytest.approx(
        res["dimensions"]["correctness"]["baseline"] - (1.0 if s05[0] else 0.0) / 90)
    assert res["dimensions"]["consistency"]["candidate"] == pytest.approx(
        res["dimensions"]["consistency"]["baseline"] - (1.0 if all(s05) else 0.0) / 30)


def test_compare_rejects_bad_input(tmp_path):
    base = _base(tmp_path)
    cand = _run_side(tmp_path, "cand", world_seed=1, run_seed=1)
    with pytest.raises(ValueError):
        compare(base[:1], cand, resamples=400)
    with pytest.raises(ValueError):
        compare(base, cand[:1], resamples=400)
    with pytest.raises(ValueError):
        compare(base, cand, resamples=50)
    paths = write_side(tmp_path, "other", runs=3, world_seed=1, run_seed=1)
    for p in paths:
        _edit_scenarios(p, lambda s: "x" + s)
    with pytest.raises(ValueError):
        compare(base, _load_paths(paths), resamples=400)


def test_compare_is_reproducible_and_ci_level_follows_g(tmp_path):
    base = _base(tmp_path)
    cand = _run_side(tmp_path, "cand", world_seed=1, run_seed=2)
    r1 = compare(base, cand, resamples=400, seed=7)
    r2 = compare(base, cand, resamples=400, seed=7)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    r3 = compare(base, cand, budget={"approach": {"gate": False}}, resamples=400, seed=7)
    assert r3["ci_level"] == 0.9875
    assert r3["dimensions"]["approach"]["gates"] is False


def test_compare_result_shape_rounding_and_extras(tmp_path):
    base = _base(tmp_path)
    same = _run_side(tmp_path, "same", world_seed=1, run_seed=1)
    res = compare(base, same, resamples=400, seed=0)
    assert set(res) == {"verdict", "exit_code", "seed", "resamples", "ci_level", "n", "runs", "dimensions",
                        "safety", "cost_per_solved", "trajectory_similarity", "new_tools", "regressions",
                        "missing", "policy_violations", "check_deltas", "warnings", "faultgate_version"}
    assert set(res["dimensions"]) == set(DIMENSIONS)
    dim_keys = {"baseline", "candidate", "delta", "ci", "budget", "gates", "label", "scenarios_needed",
                "unit", "note"}
    for name, entry in res["dimensions"].items():
        assert set(entry) == dim_keys | ({"p50", "p95"} if name == "speed" else set())
    assert res["verdict"] in ("SHIP", "REVIEW")
    assert res["exit_code"] == 0
    assert 0.0 < res["trajectory_similarity"] <= 1.0
    assert res["new_tools"] == [] and res["regressions"] == [] and res["missing"] == []
    assert set(res["check_deltas"]) == set(base[0][0].checks)
    assert all(v["delta"] == 0 for v in res["check_deltas"].values())

    def rounded(x):
        if isinstance(x, dict):
            return all(rounded(v) for v in x.values())
        if isinstance(x, list):
            return all(rounded(v) for v in x)
        return not isinstance(x, float) or x == round(x, 6)

    assert rounded(res)
    blob = json.dumps(res, sort_keys=True)
    assert not any(k in blob for k in ("prompt", "answer", "input_tokens", "tool_output"))
    assert json.loads(blob) == res

    dropped = _run_side(tmp_path, "drop", world_seed=1, run_seed=2, p_clean=0.6)
    regressions = compare(base, dropped, resamples=400, seed=0)["regressions"]
    assert regressions and set(regressions) <= {f"s{i:02d}" for i in range(30)}


def test_compare_speed_p95_is_gated_point_estimate(tmp_path):
    base = _base(tmp_path)
    cand = _run_side(tmp_path, "cand", world_seed=1, run_seed=1, latency_ms=5000.0 * 1.3)
    res = compare(base, cand, budget={"speed_p95": {"gate": True}}, resamples=400, seed=0)
    p95 = res["dimensions"]["speed_p95"]
    assert p95["ci"] is None
    assert p95["note"] == "low confidence: pooled p95, point estimate, no CI"
    assert p95["delta"] == pytest.approx(0.3)
    assert p95["label"] == "WORSE"
    assert res["ci_level"] == pytest.approx(0.99)  # speed_p95 is not counted in g
    assert res["verdict"] == "BLOCK"


def test_compare_repeated_trace_ids_across_runs_costs_not_overwritten(tmp_path):
    base = _base(tmp_path)
    r0 = make_run(1, 1000, cost_usd=0.02)
    r1 = make_run(1, 1000, cost_usd=0.06)  # identical trace_ids as r0
    p0 = tmp_path / "cand_run0.json"
    p1 = tmp_path / "cand_run1.json"
    p0.write_text(json.dumps(r0, sort_keys=True), encoding="utf-8")
    p1.write_text(json.dumps(r1, sort_keys=True), encoding="utf-8")
    cand = _load_paths([p0, p1])
    res = compare(base, cand, resamples=400, seed=0)
    cand_mean = res["dimensions"]["cost"]["candidate"]
    r0_only = compare(base, _load_paths([p0, p0]), resamples=400, seed=0)["dimensions"]["cost"]["candidate"]
    r1_only = compare(base, _load_paths([p1, p1]), resamples=400, seed=0)["dimensions"]["cost"]["candidate"]
    assert cand_mean == pytest.approx((r0_only + r1_only) / 2, abs=2e-6)


def test_compare_cost_missing_unit_none_warning_and_cost_per_solved_none(tmp_path):
    base = _base(tmp_path)
    paths = write_side(tmp_path, "cand", runs=2, world_seed=1, run_seed=1)
    for p in paths:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
        for s in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]:
            s["attributes"] = [a for a in s.get("attributes", [])
                               if a["key"] not in ("faultgate.cost_usd", "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens")]
        Path(p).write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    cand = _load_paths(paths)
    res = compare(base, cand, resamples=400, seed=0)
    assert res["dimensions"]["cost"]["label"] == "n/a"
    assert "cost n/a: missing on candidate" in res["warnings"]
    assert not any("is in USD" in w for w in res["warnings"])
    assert res["cost_per_solved"]["baseline"] is None
    assert res["cost_per_solved"]["candidate"] is None
    assert res["cost_per_solved"]["delta_pct"] is None


def test_compare_cost_units_agree_but_no_shared_value_warns(tmp_path):
    # the candidate has a usd cost, but only in a run whose scenarios are not shared: the unit matches
    # the baseline (usd) while no shared scenario has a cost on either side
    base = _base(tmp_path)
    priced = write_side(tmp_path / "priced", "cand", runs=1, world_seed=1, run_seed=1)[0]
    _edit_scenarios(priced, lambda s: "x" + s[1:])
    unpriced = write_side(tmp_path / "unpriced", "cand", runs=1, world_seed=1, run_seed=2, cost_usd=None)[0]
    doc = json.loads(unpriced.read_text(encoding="utf-8"))
    for s in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]:
        s["attributes"] = [a for a in s["attributes"]
                           if a["key"] not in ("gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens")]
    unpriced.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    res = compare(base, _load_paths([priced, unpriced]), resamples=400, seed=0)
    assert res["dimensions"]["cost"]["label"] == "n/a"
    assert "cost n/a: missing on one side (every value is None), not gated" in res["warnings"]
    assert not any("is in USD" in w or "missing on candidate" in w for w in res["warnings"])


def test_compare_cost_usd_vs_tokens_warning_and_cost_per_solved_none(tmp_path):
    base = _base(tmp_path)
    cand = _run_side(tmp_path, "cand", runs=2, world_seed=1, run_seed=1, cost_usd=None)
    res = compare(base, cand, resamples=400, seed=0)
    assert res["dimensions"]["cost"]["label"] == "n/a"
    assert any("cost n/a: baseline is in USD, candidate only has tokens" in w for w in res["warnings"])
    assert not any("missing on candidate" in w for w in res["warnings"])
    assert res["cost_per_solved"]["baseline"] is None
    assert res["cost_per_solved"]["candidate"] is None
    assert res["cost_per_solved"]["delta_pct"] is None


def test_compare_tokenizer_warning_unrelated_prices_do_not_silence(tmp_path):
    base = _run_side(tmp_path, "base", runs=2, world_seed=1, run_seed=1, cost_usd=None, model="model-a")
    cand = _run_side(tmp_path, "cand", runs=2, world_seed=1, run_seed=1, cost_usd=None, model="model-b")
    unrelated_prices = load_prices(_json_file(tmp_path, "prices.json",
                                              {"unrelated-model": {"input_per_m": 1.0, "output_per_m": 2.0}}))
    res = compare(base, cand, prices=unrelated_prices, resamples=400, seed=0)
    assert res["dimensions"]["cost"]["unit"] == "tokens (no price)"
    assert any("tokens are not comparable across tokenizers" in w and "model-a" in w and "model-b" in w
               for w in res["warnings"])


def test_compare_speed_p95_gated_no_latency_warns(tmp_path):
    base = _base(tmp_path)
    paths = write_side(tmp_path, "cand", runs=2, world_seed=1, run_seed=1)
    for p in paths:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
        for s in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]:
            s.pop("startTimeUnixNano", None)
            s.pop("endTimeUnixNano", None)
        Path(p).write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    cand = _load_paths(paths)
    res = compare(base, cand, budget={"speed_p95": {"gate": True}}, resamples=400, seed=0)
    p95 = res["dimensions"]["speed_p95"]
    assert p95["label"] == "n/a"
    assert p95["gates"] is False
    assert "speed_p95 n/a: missing on one side, not gated" in res["warnings"]
