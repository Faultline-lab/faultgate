"""`faultgate compare` from the command line: verdicts, bad input, outputs, offline."""
import json
import socket
from pathlib import Path

import pytest

from faultgate import cli
from tests.fixtures.make_compare_runs import POLICY, make_run, write_side

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "examples" / "gate"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """compare calls no model and loads no price list from the network: any socket use fails the test."""
    def blocked(*_a, **_k):
        raise OSError("network disabled in tests")

    monkeypatch.setattr(socket, "socket", blocked)


def _run(argv: list[str], capsys) -> tuple[int, str, str]:
    code = cli.main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def _examples() -> list[str]:
    return [str(GATE / "baseline_glm_run1.json"), str(GATE / "baseline_glm_run2.json")]


def _verdict(out: str) -> str:
    lines = [line for line in out.splitlines() if line.startswith("verdict:")]
    assert len(lines) == 1, out
    return lines[0]


def test_real_examples_no_false_alarm(capsys):
    code, out, err = _run(["compare", "--baseline", *_examples(), "--candidate", *_examples()], capsys)
    assert code == 0
    assert out.startswith("compare: 30 scenarios · baseline 2 runs · candidate 2 runs · seed 0 · CI 98.75% (4 gated dimensions)")
    assert "WORSE" not in out
    assert _verdict(out).split(": ")[1].split(" ")[0] in ("SHIP", "REVIEW")
    assert out.splitlines()[-1] == _verdict(out)  # the verdict is the last line
    assert "95% upper bound on violation rate" in out
    assert all(line.startswith(("note: ", "warning: ")) for line in err.splitlines()), err


def test_policy_violation_blocks(tmp_path, capsys):
    base = write_side(tmp_path, "base", runs=3)
    cand = write_side(tmp_path, "cand", runs=3, policy_violation=True)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(POLICY))
    code, out, _ = _run(["compare", "--baseline", *map(str, base), "--candidate", *map(str, cand),
                         "--policy", str(policy), "--resamples", "400"], capsys)
    assert code == 1
    assert _verdict(out).startswith("verdict: BLOCK · worse on safety")
    assert "policy violations" in out and "BLOCK" in out


def test_verdict_line_says_one_policy_violation(tmp_path, capsys):
    base = write_side(tmp_path, "base", runs=2)
    cand = write_side(tmp_path, "cand", runs=2)
    (tmp_path / "cand_run0.json").write_text(json.dumps(make_run(0, 0, policy_violation=True), sort_keys=True),
                                              encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(POLICY))
    _code, out, _ = _run(["compare", "--baseline", *map(str, base), "--candidate", *map(str, cand),
                          "--policy", str(policy), "--resamples", "400"], capsys)
    assert "safety (1 policy violation)" in _verdict(out)


def test_markdown_absolute_change_is_not_a_percent(tmp_path, capsys):
    # a baseline with zero calls makes `approach` fall back to an absolute change
    base = write_side(tmp_path, "base", runs=2, world_seed=1, run_seed=1, zero_calls=True)
    cand = write_side(tmp_path, "cand", runs=2, world_seed=1, run_seed=1)
    md = tmp_path / "abs.md"
    _run(["compare", "--baseline", *map(str, base), "--candidate", *map(str, cand),
          "--resamples", "400", "--markdown", str(md)], capsys)
    text = md.read_text()
    assert "approach: absolute change used because the baseline mean is 0" in text
    row = next(line for line in text.splitlines() if line.startswith("| approach |"))
    assert "%" not in row.split("|")[4]  # the change cell


def test_strict_turns_review_into_exit_1(capsys, tmp_path):
    out_json = tmp_path / "strict.json"
    args = ["compare", "--baseline", *_examples(), "--candidate", *_examples(), "--json", str(out_json)]
    code, out, _ = _run(args, capsys)
    if "verdict: REVIEW" not in out:  # pragma: no cover - depends on the shipped examples
        pytest.skip("the shipped examples no longer produce REVIEW")
    assert code == 0 and json.loads(out_json.read_text())["exit_code"] == 0
    assert _run([*args, "--strict"], capsys)[0] == 1
    assert json.loads(out_json.read_text())["exit_code"] == 1  # the shell code is written out


def test_json_is_byte_identical(capsys, tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    args = ["compare", "--baseline", *_examples(), "--candidate", *_examples(), "--seed", "7"]
    assert _run([*args, "--json", str(first)], capsys)[0] == 0
    assert _run([*args, "--json", str(second)], capsys)[0] == 0
    assert first.read_bytes() == second.read_bytes()
    doc = json.loads(first.read_text())
    assert doc["seed"] == 7 and doc["resamples"] == 2000 and doc["faultgate_version"] == "0.5.0"
    assert str(tmp_path) not in first.read_text()  # no absolute paths


def test_markdown_has_table_and_reproduce_command(capsys, tmp_path):
    md = tmp_path / "pr.md"
    _code, out, _ = _run(["compare", "--baseline", *_examples(), "--candidate", *_examples(), "--markdown", str(md)], capsys)
    text = md.read_text()
    assert text.startswith("<!-- faultgate -->")
    assert "| dimension | baseline | candidate | change | budget | result |" in text
    assert "pip install 'faultgate==" in text
    assert "faultgate compare --baseline examples/gate/baseline_glm_run1.json" in text
    assert _verdict(out)


def test_audit_files(tmp_path, capsys):
    base = write_side(tmp_path, "base", runs=2)
    cand = write_side(tmp_path, "cand", runs=2)
    audit = tmp_path / "audit"
    code, _, _ = _run(["compare", "--baseline", *map(str, base), "--candidate", *map(str, cand),
                       "--seed", "3", "--resamples", "400", "--audit", str(audit)], capsys)
    assert code == 0
    assert sorted(p.name for p in audit.iterdir()) == ["failing_traces.json", "repro.sh", "verdict.json"]
    doc = json.loads((audit / "verdict.json").read_text())
    assert doc["seed"] == 3 and doc["resamples"] == 400 and doc["verdict"] == doc["compare"]["verdict"]
    assert {i["role"] for i in doc["inputs"]} == {"baseline", "candidate"}
    assert all(len(i["sha256"]) == 64 for i in doc["inputs"])
    text = "".join((audit / name).read_text() for name in ("verdict.json", "failing_traces.json", "repro.sh"))
    assert "scenario s0" not in text  # prompts are never written out
    failing = json.loads((audit / "failing_traces.json").read_text())
    assert [f["run"] for f in failing] == list(range(1, len(cand) + 1))


def test_formats_compared_against_themselves(capsys):
    for name in ("braintrust", "langfuse", "langsmith", "openinference"):
        path = str(REPO / "examples" / "formats" / f"{name}.json")
        code, out, _ = _run(["compare", "--baseline", path, path, "--candidate", path, path,
                             "--resamples", "400"], capsys)
        assert code == 0, name
        assert "WORSE" not in out, name


def _cases(tmp_path: Path):
    base = write_side(tmp_path, "base", runs=2)
    cand = write_side(tmp_path, "cand", runs=2)
    both = ["--baseline", *map(str, base), "--candidate", *map(str, cand)]
    budget = tmp_path / "budget.json"
    neg = tmp_path / "neg.json"
    prices = tmp_path / "prices.json"
    budget.write_text(json.dumps({"latency": {"budget": 0.1}}))
    neg.write_text(json.dumps({"cost": {"budget": -0.1}}))
    prices.write_text(json.dumps({"model-a": {"input_per_m": 1.0}}))
    return {
        "one baseline run": ["--baseline", str(base[0]), "--candidate", *map(str, cand)],
        "one candidate run": ["--baseline", *map(str, base), "--candidate", str(cand[0])],
        "no shared scenarios": ["--baseline", *map(str, base),
                                "--candidate", str(REPO / "examples" / "traces.json"),
                                str(REPO / "examples" / "traces.json")],
        "unknown budget key": [*both, "--budget", str(budget)],
        "negative budget": [*both, "--budget", str(neg)],
        "malformed price entry": [*both, "--prices", str(prices)],
        "too few resamples": [*both, "--resamples", "50"],
    }


@pytest.mark.parametrize("case", ["one baseline run", "one candidate run", "no shared scenarios",
                                  "unknown budget key", "negative budget", "malformed price entry",
                                  "too few resamples"])
def test_bad_input_exits_2_with_one_line(case, tmp_path, capsys):
    code, out, err = _run(["compare", *_cases(tmp_path)[case]], capsys)
    assert code == 2, out
    assert out == ""
    assert len(err.strip().splitlines()) == 1, err
    assert err.startswith("faultgate: ")


def test_ci_level_printed_exact_when_g_is_four(capsys, tmp_path):
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"approach": {"gate": False}}))
    base = write_side(tmp_path, "base", runs=2, world_seed=1, run_seed=1)
    cand = write_side(tmp_path, "cand", runs=2, world_seed=1, run_seed=2)
    md = tmp_path / "ci.md"
    code, out, _ = _run(["compare", "--baseline", *map(str, base), "--candidate", *map(str, cand),
                         "--budget", str(budget), "--resamples", "400", "--markdown", str(md)], capsys)
    assert code == 0
    assert "CI 98.75% (4 gated dimensions)" in out
    assert "98.75%" in md.read_text()
