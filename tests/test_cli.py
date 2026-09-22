import json

import pytest

from faultgate import cli
from tests.conftest import EXAMPLES, FakeJudge


@pytest.fixture
def patched(monkeypatch):
    judge = FakeJudge(fire_on={"ABSTENTION": "couldn’t verify"})
    monkeypatch.setattr(cli, "get_judge", lambda spec: judge)
    return judge


def test_exit_1_when_fired(patched, capsys, tmp_path):
    out = tmp_path / "r.json"
    assert cli.main(["check", str(EXAMPLES), "--json", str(out)]) == 1
    text = capsys.readouterr().out
    assert "1/5 traces failed" in text and "FAIL" in text
    assert json.loads(out.read_text())["fired"]["abstention"] == 1


def test_exit_0_when_clean(patched):
    assert cli.main(["check", str(EXAMPLES), "--check", "search_loop"]) == 0


def test_exit_2_on_bad_input(patched, capsys):
    assert cli.main(["check", "does-not-exist.json"]) == 2
    assert cli.main(["check", str(EXAMPLES), "--check", "nope"]) == 2
    assert "unknown check" in capsys.readouterr().err


def test_listing(capsys):
    assert cli.main(["checks"]) == 0
    assert cli.main(["judges"]) == 0
    out = capsys.readouterr().out
    assert "search_loop" in out and "litellm:<model>" in out
