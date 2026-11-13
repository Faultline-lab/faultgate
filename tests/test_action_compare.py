"""The composite action in compare mode: the args it builds, the outputs it writes.

The gate script is the same step as in check mode (found by its `faultgate "${args[@]}"` line);
only the fake `faultgate` differs, because compare writes a compare heading. `_bash` from
test_report.py is not reused: it installs its own fakes file.
"""

import subprocess

import pytest

from tests.test_report import _run_scripts, needs_bash

FAKES = r'''
faultgate() {
  printf '%s\n' "$@" > "$FAKE_DIR/faultgate.args"
  h="## faultgate compare: BLOCK"; [ "${1:-}" = check ] && h="## faultgate: FAIL"
  while [ $# -gt 0 ]; do
    if [ "$1" = "--markdown" ]; then printf '<!-- faultgate -->\n%s\n' "$h" > "$2"; fi
    shift
  done
  return "${FAKE_CODE:-1}"
}
'''


def _gate() -> str:
    return next(s for s in _run_scripts() if 'faultgate "${args[@]}"' in s)


def _sh(script: str, env: dict, tmp) -> subprocess.CompletedProcess:
    (tmp / "fakes.sh").write_text(FAKES, encoding="utf-8")
    (tmp / "step.sh").write_text(script, encoding="utf-8")
    full = {**__import__("os").environ, **env, "BASH_ENV": str(tmp / "fakes.sh"), "FAKE_DIR": str(tmp)}
    return subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", str(tmp / "step.sh")],
                          env=full, capture_output=True, text=True, cwd=tmp)


def _env(tmp, **kw) -> dict:
    env = {k: "" for k in ("TRACES", "BASELINE", "CANDIDATE", "AUDIT", "BAND", "GOLDEN", "POLICY",
                           "OUTCOMES", "CHECKS", "JUDGE", "BUDGET", "PRICES", "STRICT")}
    env.update({"FORMAT": "auto", "REDACT": "builtin", "SEED": "0", "RESAMPLES": "2000",
                "REPORT": "r.json", "AUDIT": "audit", "GITHUB_OUTPUT": str(tmp / "out"),
                "GITHUB_STEP_SUMMARY": str(tmp / "summary"), "RUNNER_TEMP": str(tmp)})
    env.update(kw)
    return env


@needs_bash
def test_compare_args_from_a_glob_and_from_lines(tmp_path):
    for name in ("b1.json", "b2.json", "c1.json", "c2.json"):
        (tmp_path / name).touch()
    p = _sh(_gate(), _env(tmp_path, BASELINE="b*.json", CANDIDATE="c1.json\nc2.json",
                          BUDGET="budget.json", PRICES="prices.json", SEED="7", RESAMPLES="400",
                          STRICT="true", GOLDEN="g.json", POLICY="p.json", OUTCOMES="o.json",
                          CHECKS="search_loop", JUDGE="laya", FAKE_CODE="1"), tmp_path)
    assert p.returncode == 0, p.stderr
    args = (tmp_path / "faultgate.args").read_text().splitlines()
    assert args[:7] == ["compare", "--baseline", "b1.json", "b2.json", "--candidate", "c1.json", "c2.json"]
    for flag, value in [("--json", "r.json"), ("--audit", "audit"), ("--seed", "7"), ("--resamples", "400"),
                        ("--budget", "budget.json"), ("--prices", "prices.json"),
                        ("--golden", "g.json"), ("--policy", "p.json"), ("--outcomes", "o.json"),
                        ("--check", "search_loop"), ("--judge", "laya")]:
        assert args[args.index(flag) + 1] == value, flag
    assert args[args.index("--markdown") + 1].endswith("/faultgate-summary.md")
    assert "--strict" in args
    outputs = (tmp_path / "out").read_text()
    assert "verdict=BLOCK" in outputs and "code=1" in outputs
    assert "## faultgate compare: BLOCK" in (tmp_path / "summary").read_text()


@needs_bash
def test_compare_shell_words_in_inputs_stay_literal(tmp_path):
    evil = '$(touch pwned) `touch pwned2` ;touch pwned3'
    (tmp_path / "c1.json").touch()
    p = _sh(_gate(), _env(tmp_path, BASELINE=evil, CANDIDATE="c1.json"), tmp_path)
    assert p.returncode == 0, p.stderr
    assert not [f for f in tmp_path.iterdir() if f.name.startswith("pwned")]
    args = (tmp_path / "faultgate.args").read_text().splitlines()
    assert "pwned)" in args and "`touch" in args and "pwned3" in args


@needs_bash
@pytest.mark.parametrize("env,msg", [
    ({"TRACES": "t.json", "CANDIDATE": "c1.json"},
     "::error::faultgate: set traces (check) or baseline + candidate (compare), not both"),
    ({}, "::error::faultgate: set traces (check) or baseline + candidate (compare)"),
    ({"CANDIDATE": "c1.json"},
     "::error::faultgate: compare mode needs baseline as well as candidate"),
])
def test_compare_refuses_both_or_neither_mode(tmp_path, env, msg):
    p = _sh(_gate(), _env(tmp_path, **env), tmp_path)
    assert p.returncode == 0, p.stderr
    assert [ln for ln in p.stdout.splitlines() if ln.startswith("::error::")] == [msg]
    assert "code=2" in (tmp_path / "out").read_text()
    assert not (tmp_path / "faultgate.args").exists()  # faultgate never ran


@needs_bash
def test_compare_strict_false_and_empty_optional_paths(tmp_path):
    for name in ("b1.json", "b2.json", "c1.json", "c2.json"):
        (tmp_path / name).touch()
    p = _sh(_gate(), _env(tmp_path, BASELINE="b*.json", CANDIDATE="c*.json", STRICT="false",
                          FAKE_CODE="0"), tmp_path)
    assert p.returncode == 0, p.stderr
    args = (tmp_path / "faultgate.args").read_text().splitlines()
    assert args[:2] == ["compare", "--baseline"]
    for flag in ("--strict", "--budget", "--prices"):
        assert flag not in args


@needs_bash
def test_check_mode_untouched(tmp_path):
    p = _sh(_gate(), _env(tmp_path, TRACES="t.json", BAND="band.json", SEED="7", RESAMPLES="400",
                          STRICT="true", FAKE_CODE="0"), tmp_path)
    assert p.returncode == 0, p.stderr
    args = (tmp_path / "faultgate.args").read_text().splitlines()
    assert args[:2] == ["check", "t.json"]
    assert args[args.index("--band") + 1] == "band.json"
    for flag in ("--strict", "--seed", "--resamples"):
        assert flag not in args
    assert "verdict=FAIL" in (tmp_path / "out").read_text()
