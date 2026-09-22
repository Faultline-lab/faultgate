import json

import pytest

from faultgate import band, cli
from faultgate.checks import select
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


def test_pass_warn_fail_zones():
    r1 = _report(GATE / "baseline_glm_run1.json", {})
    r2 = _report(GATE / "baseline_glm_run2.json", {})
    b = band.build([r1, r2])  # every scenario clean: min 1.0, n=30 -> PASS >= 29/30, FAIL < 27/30
    prompts = [t.prompt for t in load_otlp(GATE / "baseline_glm_run1.json")]
    for k, expected in ((1, "PASS"), (2, "WARN"), (4, "FAIL")):
        g = band.evaluate(_report(GATE / "baseline_glm_run1.json", {"DIRECTION": prompts[:k]}), b)
        assert g["verdict"] == expected, (k, g)
        assert len(g["regressions"]) == k


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
    assert code == 0 and "gate: PASS" in text
    assert json.loads(rep.read_text())["gate"]["verdict"] == "PASS"
    assert cli.main(["baseline", str(GATE / "baseline_glm_run1.json"), "--judge", "fake", "--out", str(out)]) == 2
