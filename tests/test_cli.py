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
    assert json.loads(out.read_text())["fired"] == {"search_loop": 1, "abstention": 1, "wrong_direction": 1}


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
