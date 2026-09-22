import json

import pytest

from faultgate import cli
from faultgate.checks import Golden, select, truncation_rule, wrong_answer_rule
from faultgate.runner import run
from faultgate.trace import load_otlp
from tests.conftest import EXAMPLES

GOLDEN = EXAMPLES.parent / "golden.json"


def test_golden_keyed_by_prompt_or_scenario():
    g = Golden.load(GOLDEN)
    traces = load_otlp(EXAMPLES)
    assert all(g.expected(t) for t in traces)  # keyed by prompt text
    g2 = Golden.load(EXAMPLES.parent / "gate" / "golden.json")
    assert g2.expected(load_otlp(EXAMPLES.parent / "gate" / "baseline_glm_run1.json")[0])  # keyed by gen_ai.agent.id


def test_wrong_answer_and_truncation_on_examples():
    g = Golden.load(GOLDEN)
    traces = load_otlp(EXAMPLES)  # loop(no answer), abstention, Zinc-4524 ok, Kelvin-4384 ok, "Onyx district" for Onyx-4413
    assert [wrong_answer_rule(t, g).detected for t in traces] == [False, True, False, False, True]
    assert [truncation_rule(t, g).detected for t in traces] == [False, False, False, False, True]
    assert "Onyx" in truncation_rule(traces[4], g).reason


def test_unbound_golden_checks_are_skipped(capsys):
    report = run(load_otlp(EXAMPLES), select("wrong_answer,search_loop"))
    assert report.checks == ["search_loop"]
    assert cli.main(["check", str(EXAMPLES), "--check", "wrong_answer"]) == 2
    assert "need --golden" in capsys.readouterr().err


def test_cli_with_golden(tmp_path):
    out = tmp_path / "r.json"
    assert cli.main(["check", str(EXAMPLES), "--golden", str(GOLDEN), "--json", str(out)]) == 1
    fired = json.loads(out.read_text())["fired"]
    assert fired == {"search_loop": 1, "abstention": 1, "policy": 0, "wrong_answer": 2, "truncation": 1}


def test_golden_load_rejects_bad_shape(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps(["a"]))
    with pytest.raises(ValueError):
        Golden.load(p)
