import json

import pytest

from faultgate import cli
from tests.conftest import EXAMPLES, FakeJudge


@pytest.fixture
def patched(monkeypatch):
    judge = FakeJudge(fire_on={"DIRECTION": "couldn’t verify"})
    monkeypatch.setattr(cli, "get_judge", lambda spec: judge)
    return judge


def test_exit_1_when_fired(patched, capsys, tmp_path):
    out = tmp_path / "r.json"
    assert cli.main(["check", str(EXAMPLES), "--judge", "fake", "--json", str(out)]) == 1
    text = capsys.readouterr().out
    assert "2/5 traces failed" in text and "FAIL" in text  # rule catches the loop, fake judge the abstention
    assert json.loads(out.read_text())["fired"] == {
        "search_loop": 1,
        "abstention": 1,
        "policy": 0,
        "repeated_call": 0,
        "unrecovered_tool_error": 0,
        "context_overflow": 0,
        "wrong_direction": 1,
    }


def test_exit_0_when_clean(monkeypatch):
    monkeypatch.setattr(cli, "get_judge", lambda spec: FakeJudge())
    assert cli.main(["check", str(EXAMPLES), "--check", "wrong_direction", "--judge", "fake"]) == 0


def test_rules_only_by_default(capsys):
    assert cli.main(["check", str(EXAMPLES)]) == 1  # loop + abstention rules fire; no model loaded
    err = capsys.readouterr().err
    assert "wrong_direction skipped" in err
    assert cli.main(["check", str(EXAMPLES), "--check", "wrong_direction"]) == 2  # nothing runnable without a judge


def test_exit_2_on_bad_input(patched, capsys):
    assert cli.main(["check", "does-not-exist.json"]) == 2
    assert cli.main(["check", str(EXAMPLES), "--check", "nope"]) == 2
    assert "unknown check" in capsys.readouterr().err


def test_listing(capsys):
    assert cli.main(["checks"]) == 0
    assert cli.main(["judges"]) == 0
    out = capsys.readouterr().out
    assert "search_loop" in out and "litellm:<model>" in out


def test_exit_2_on_malformed_policy(tmp_path, capsys):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({"limits": [{"tool": "issue_refund"}]}))
    assert cli.main(["check", str(EXAMPLES), "--policy", str(p)]) == 2
    assert "policy: limits[0]" in capsys.readouterr().err


def test_band_prints_hard_block(tmp_path, capsys):
    gate = EXAMPLES.parent / "gate"
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({"max_calls": 0}))
    code = cli.main(["check", str(gate / "baseline_glm_run2.json"), "--band", str(gate / "band.json"),
                     "--golden", str(gate / "golden.json"), "--policy", str(p)])
    out = capsys.readouterr().out
    assert code == 1 and "gate: FAIL · policy violation in 30 trace(s) (hard block)" in out


def test_output_parent_directories_are_created(tmp_path, capsys):
    out = tmp_path / "not" / "yet" / "report.json"
    md = tmp_path / "md" / "pr.md"
    assert cli.main(["check", str(EXAMPLES), "--check", "repeated_call", "--json", str(out), "--markdown", str(md)]) == 0
    assert json.loads(out.read_text())["fired"] == {"repeated_call": 0} and md.is_file()
    band = tmp_path / "bands" / "band.json"
    assert cli.main(["baseline", str(EXAMPLES), str(EXAMPLES), "--check", "repeated_call", "--out", str(band)]) == 0
    assert band.is_file()


def test_unwritable_output_exits_2_not_1(tmp_path, capsys):
    """A clean run whose report cannot be written must not exit 1: in CI that reads as a gate FAIL."""
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert cli.main(["check", str(EXAMPLES), "--check", "repeated_call", "--json", str(blocker / "report.json")]) == 2
    captured = capsys.readouterr()
    assert "0/5 traces failed" in captured.out and "cannot write output" in captured.err


# --- 0.4: --outcomes and precheck --policy ---

GATE = EXAMPLES.parent / "gate"


def _outcomes(tmp_path, path, false_every=4, drop=0):
    from faultgate.trace import load_otlp

    keys = [t.key for t in load_otlp(path)]
    keys = keys[: len(keys) - drop]
    p = tmp_path / f"outcomes-{path.stem}.json"
    p.write_text(json.dumps({k: i % false_every != 0 for i, k in enumerate(keys)}))
    return p


def test_check_with_outcomes_reports_both_rates(tmp_path, capsys):
    oc = _outcomes(tmp_path, EXAMPLES, false_every=5, drop=1)  # 4 judged (1 false), 1 without an outcome
    out, md = tmp_path / "r.json", tmp_path / "pr.md"
    assert cli.main(["check", str(EXAMPLES), "--outcomes", str(oc), "--json", str(out), "--markdown", str(md)]) == 1
    text = capsys.readouterr().out
    assert "outcome: 3/4 true (0.75) · 1 scenario(s) without an outcome, not judged · structural clean rate 0.60" in text
    rep = json.loads(out.read_text())
    assert rep["checks"][-1] == "outcome" and rep["fired"]["outcome"] == 1
    assert rep["outcome"] == {"judged": 4, "missing": 1, "true": 3, "outcome_rate": 0.75, "structural_clean_rate": 0.6,
                              "clean_rate": 0.6}
    assert "**Outcome:** 3/4 true (0.75) · 1 without an outcome · structural clean rate 0.60" in md.read_text(encoding="utf-8")


def test_without_outcomes_the_report_is_unchanged(tmp_path):
    out = tmp_path / "r.json"
    cli.main(["check", str(EXAMPLES), "--json", str(out)])
    rep = json.loads(out.read_text())
    assert "outcome" not in rep and "outcome" not in rep["checks"]


def test_outcomes_join_the_band_and_gate(tmp_path, capsys):
    runs = [GATE / "baseline_glm_run1.json", GATE / "baseline_glm_run2.json"]
    band = tmp_path / "band.json"
    args = ["baseline", *map(str, runs), "--outcomes", str(_outcomes(tmp_path, runs[0], false_every=100)), "--out", str(band)]
    assert cli.main(args) == 0
    assert json.loads(band.read_text())["checks"][-1] == "outcome"
    # the same run, now with most outcomes false: structurally identical, outcome-wise a regression
    bad = _outcomes(tmp_path, runs[1], false_every=2)
    assert cli.main(["check", str(runs[1]), "--outcomes", str(bad), "--band", str(band)]) == 1
    assert "gate: FAIL" in capsys.readouterr().out


def test_malformed_outcomes_exit_2(tmp_path, capsys):
    p = tmp_path / "o.json"
    p.write_text('{"a": "yes"}')
    assert cli.main(["check", str(EXAMPLES), "--outcomes", str(p)]) == 2
    assert "outcomes" in capsys.readouterr().err


def test_precheck_reports_policy_keys_and_missing_json_results(tmp_path, capsys):
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"allow": ["search"], "require_complete": True,
                               "requires": [{"tool": "lookup", "after": "search", "latest": True,
                                             "result": [{"path": "status", "equals": "ok"}]}]}))
    assert cli.main(["precheck", str(EXAMPLES), "--policy", str(pol)]) == 0
    out = capsys.readouterr().out
    assert "policy keys: allow, require_complete, requires (after, latest, result, tool)" in out
    assert "no tool result in the checked traces is JSON" not in out  # the examples carry JSON results
    # traces whose tool results are plain text cannot satisfy a result predicate
    doc = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    for s in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]:
        for a in s["attributes"]:
            if a["key"] == "gen_ai.tool.call.result":
                a["value"] = {"stringValue": "plain text"}
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps(doc))
    assert cli.main(["precheck", str(plain), "--policy", str(pol)]) == 0
    assert "no tool result in the checked traces is JSON" in capsys.readouterr().out
    pol.write_text('{"requires": [{"tool": "x"}]}')
    assert cli.main(["precheck", str(EXAMPLES), "--policy", str(pol)]) == 2
