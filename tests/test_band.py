import json

import pytest

from faultgate import band, cli
from faultgate.checks import Policy, select
from faultgate.runner import run
from faultgate.trace import load_otlp
from tests.conftest import EXAMPLES, FakeJudge

GATE = EXAMPLES.parent / "gate"


JUDGE_ONLY = "wrong_direction"  # the fake judge controls every verdict; the other checks are real rules


def _report(path, fire_on):
    return run(load_otlp(path), select(JUDGE_ONLY), FakeJudge(fire_on=fire_on))


def test_scenario_key_is_stable_across_runs():
    a = {t.key for t in load_otlp(GATE / "baseline_glm_run1.json")}
    b = {t.key for t in load_otlp(GATE / "baseline_glm_run2.json")}
    assert a == b and len(a) == 30 and "r-0336" in a


def test_prompt_hash_key_when_no_id():
    t = load_otlp(EXAMPLES)[0]  # examples/traces.json has no gen_ai.agent.id
    assert t.key.startswith("p-") and len(t.key) == 14


def test_build_and_evaluate():
    # fake judge: abstention fires on "couldn’t"; baseline runs differ a little
    r1 = _report(GATE / "baseline_glm_run1.json", {"DIRECTION": "couldn’t"})
    r2 = _report(GATE / "baseline_glm_run2.json", {"DIRECTION": "couldn’t"})
    b = band.build([r1, r2])
    assert b["n"] == 30 and b["runs"] == 2 and 0 <= b["clean_rate"]["min"] <= b["clean_rate"]["max"] <= 1
    # candidate identical to a baseline run -> PASS, no regressions
    g = band.evaluate(r1, b)
    assert g["verdict"] == "PASS" and g["regressions"] == [] and g["missing"] == []
    # candidate where everything fires -> FAIL, regressions = every scenario clean in all runs
    bad = _report(GATE / "candidate_gpt.json", {"DIRECTION": "Steps used"})
    g = band.evaluate(bad, b)
    assert g["verdict"] == "FAIL" and g["clean_rate"] == 0.0
    assert set(g["regressions"]) == set(b["clean_in_all_runs"])


def test_pass_inconclusive_fail_zones():
    r1 = _report(GATE / "baseline_glm_run1.json", {})
    r2 = _report(GATE / "baseline_glm_run2.json", {})
    b = band.build([r1, r2])  # every scenario clean: min 1.0, n=30 -> PASS >= 29/30, FAIL < 27/30
    prompts = [t.prompt for t in load_otlp(GATE / "baseline_glm_run1.json")]
    # k=3 (0.90) used to be WARN; the one-sided test now calls it (p ~ 0.034 < 0.05)
    for k, expected in ((1, "PASS"), (2, "INCONCLUSIVE"), (3, "FAIL"), (4, "FAIL")):
        g = band.evaluate(_report(GATE / "baseline_glm_run1.json", {"DIRECTION": prompts[:k]}), b)
        assert g["verdict"] == expected, (k, g)
        assert len(g["regressions"]) == k


def test_inconclusive_reports_required_n():
    r1 = _report(GATE / "baseline_glm_run1.json", {})
    b = band.build([r1, _report(GATE / "baseline_glm_run2.json", {})])
    prompts = [t.prompt for t in load_otlp(GATE / "baseline_glm_run1.json")]
    g = band.evaluate(_report(GATE / "baseline_glm_run1.json", {"DIRECTION": prompts[:2]}), b)
    s = g["stats"]
    assert g["verdict"] == "INCONCLUSIVE"
    assert s["alpha"] == 0.05 and s["power"] == 0.80
    assert 0.05 < s["p_value"] < 0.10
    # two-proportion sample size, p_b = 1.0 vs p_c = 28/30: 89 scenarios, 59 more than the 30 run
    assert s["required_n"] == 89 and s["additional_n"] == 59
    for v in ("PASS", "FAIL"):
        k = 0 if v == "PASS" else 4
        g = band.evaluate(_report(GATE / "baseline_glm_run1.json", {"DIRECTION": prompts[:k]}), b)
        assert g["verdict"] == v and g["stats"]["required_n"] is None


def test_cli_inconclusive_exits_0_and_says_how_many(monkeypatch, tmp_path, capsys):
    prompts = [t.prompt for t in load_otlp(GATE / "baseline_glm_run1.json")]
    out = tmp_path / "band.json"
    monkeypatch.setattr(cli, "get_judge", lambda spec: FakeJudge())
    assert cli.main(["baseline", str(GATE / "baseline_glm_run1.json"), str(GATE / "baseline_glm_run2.json"),
                     "--check", JUDGE_ONLY, "--judge", "fake", "--out", str(out)]) == 0
    assert "use 3 or more" in capsys.readouterr().err
    monkeypatch.setattr(cli, "get_judge", lambda spec: FakeJudge(fire_on={"DIRECTION": prompts[:2]}))
    code = cli.main(["check", str(GATE / "baseline_glm_run1.json"), "--check", JUDGE_ONLY, "--judge", "fake", "--band", str(out)])
    text = capsys.readouterr().out
    assert code == 0 and "gate: INCONCLUSIVE" in text and "needs 89 scenarios" in text and "run 59 more" in text


def test_noise_widens_the_test():
    """Run-to-run spread in the baseline makes the same drop less significant."""
    tight = {"judge": "none", "checks": [], "runs": 3, "n": 30, "scenarios": [f"s{i}" for i in range(30)],
             "clean_rate": {"min": 0.9, "max": 0.9, "mean": 0.9, "per_run": [0.9, 0.9, 0.9]}, "clean_in_all_runs": []}
    noisy = {**tight, "clean_rate": {"min": 0.8, "max": 1.0, "mean": 0.9, "per_run": [0.8, 0.9, 1.0]}}
    assert band.stats(0.8, 30, tight)["p_value"] < band.stats(0.8, 30, noisy)["p_value"]


def test_band_from_fewer_than_three_runs_warns():
    r1 = _report(GATE / "baseline_glm_run1.json", {})
    b = band.build([r1, _report(GATE / "baseline_glm_run2.json", {})])
    assert any("3 or more" in w for w in band.evaluate(r1, b)["warnings"])
    b3 = band.build([r1, r1, r1])
    assert not any("3 or more" in w for w in band.evaluate(r1, b3)["warnings"])


def test_build_rejects_bad_input():
    r1 = _report(GATE / "baseline_glm_run1.json", {})
    with pytest.raises(ValueError):
        band.build([r1])
    other = run(load_otlp(EXAMPLES), select("abstention"), FakeJudge())  # different check set
    with pytest.raises(ValueError):
        band.build([r1, other])


def test_cli_baseline_then_gate(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "get_judge", lambda spec: FakeJudge(fire_on={"DIRECTION": "couldn’t"}))
    out = tmp_path / "band.json"
    assert cli.main(["baseline", str(GATE / "baseline_glm_run1.json"), str(GATE / "baseline_glm_run2.json"), "--check", JUDGE_ONLY, "--judge", "fake", "--out", str(out)]) == 0
    b = json.loads(out.read_text())
    assert b["runs"] == 2
    rep = tmp_path / "r.json"
    code = cli.main(["check", str(GATE / "baseline_glm_run2.json"), "--check", JUDGE_ONLY, "--judge", "fake", "--band", str(out), "--json", str(rep)])
    text = capsys.readouterr().out
    assert code == 0 and "gate: PASS" in text and " · p=" in text
    assert json.loads(rep.read_text())["gate"]["verdict"] == "PASS"
    assert cli.main(["baseline", str(GATE / "baseline_glm_run1.json"), "--judge", "fake", "--out", str(out)]) == 2


def test_policy_violation_is_a_hard_block():
    r1 = _report(GATE / "baseline_glm_run1.json", {})
    b = band.build([r1, _report(GATE / "baseline_glm_run2.json", {})])  # min 1.0, n=30: PASS >= 29/30
    cand = run(load_otlp(GATE / "baseline_glm_run1.json"), select(JUDGE_ONLY + ",policy", Policy(max_calls=0)), FakeJudge())
    first = cand.trace_ids[0]
    cand.results = [r for r in cand.results if r.check != "policy" or r.trace_id == first]  # one violation: 29/30 clean
    g = band.evaluate(cand, b)
    assert g["verdict"] == "FAIL" and g["policy_violations"] == [first]
    g = band.evaluate(r1, b)
    assert g["verdict"] == "PASS" and g["policy_violations"] == []
