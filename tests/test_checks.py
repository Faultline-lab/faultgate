from faultgate.checks import CHECKS, _empty_result, search_loop_rule
from faultgate.trace import load_otlp
from tests.conftest import EXAMPLES


def test_empty_result_forms():
    assert all(_empty_result(s) for s in ("", "[]", "{}", '{"n_candidates": 0, "candidates": []}', '{"results": []}', '{"count": 0}'))
    assert not any(_empty_result(s) for s in ("hit", '{"n_candidates": 2}', '["a"]', '{"results": ["x"]}', "not json"))


def test_search_loop_rule_on_examples():
    traces = load_otlp(EXAMPLES)
    fired = [search_loop_rule(t).detected for t in traces]
    assert fired == [True, False, False, False, False]  # only s-0074 (12 steps, no answer, step cap)
    assert "cap" in search_loop_rule(traces[0]).reason


def test_rule_needs_no_judge(fake_judge):
    from faultgate.checks import select
    from faultgate.runner import run

    judge = fake_judge()
    report = run(load_otlp(EXAMPLES), select("search_loop"), judge)
    assert judge.calls == [] and report.fired == {"search_loop": 1}


def test_kinds():
    assert [c.kind for c in CHECKS.values()] == ["rule", "rule", "rule", "judge"]


def test_abstention_rule_on_examples():
    from faultgate.checks import abstention_rule

    fired = [abstention_rule(t).detected for t in load_otlp(EXAMPLES)]
    assert fired == [False, True, False, False, False]  # only s-0020 "I couldn’t verify …"
