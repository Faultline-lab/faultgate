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
