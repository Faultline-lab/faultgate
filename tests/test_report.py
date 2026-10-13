import json
from pathlib import Path

from faultgate import band as bandmod
from faultgate import cli
from faultgate.checks import select
from faultgate.runner import run
from faultgate.trace import load_otlp
from tests.conftest import EXAMPLES

GATE = EXAMPLES.parent / "gate"


def test_trace_measures():
    t = load_otlp(GATE / "baseline_glm_run1.json")[0]
    assert t.latency_ms > 0
    real = load_otlp(EXAMPLES)[0]
    assert real.usage  # examples/traces.json records token usage
    rep = run([t], select("search_loop"), None)
    assert rep.measures[t.trace_id]["latency_ms"] == t.latency_ms


def test_band_records_usage_and_gate_reports_delta():
    reps = [run(load_otlp(GATE / f"baseline_glm_run{i}.json"), select("search_loop"), None) for i in (1, 2)]
    b = bandmod.build(reps)
    assert b["usage"]["latency_ms"] > 0 and b["usage"]["tokens"] is None
    cand = run(load_otlp(GATE / "candidate_gpt.json"), select("search_loop"), None)
    g = bandmod.evaluate(cand, b)
    lat = g["usage"]["latency_ms"]
    assert lat["baseline"] == b["usage"]["latency_ms"] and lat["candidate"] > 0
    assert lat["delta_pct"] == round(100 * (lat["candidate"] - lat["baseline"]) / lat["baseline"], 1)
    assert "tokens" not in g["usage"]  # nothing to compare


def test_markdown_and_audit_from_cli(tmp_path, capsys):
    md, audit = tmp_path / "pr.md", tmp_path / "audit"
    code = cli.main(["check", str(GATE / "candidate_gpt.json"), "--band", str(GATE / "band.json"),
                     "--golden", str(GATE / "golden.json"), "--markdown", str(md), "--audit", str(audit)])
    assert code == 1
    text = md.read_text(encoding="utf-8")
    assert text.startswith("<!-- faultgate -->")
    assert "## faultgate: FAIL" in text
    assert "clean rate 0.30" in text and "p=" in text
    assert "| search_loop | 18 |" in text
    assert "### Failing traces" in text and "| r-0214 |" in text
    assert "### Latency and cost vs baseline" in text and "latency / run" in text
    assert "### Reproduce locally" in text and "faultgate check" in text and "candidate_gpt.json" in text
    verdict = json.loads((audit / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "FAIL" and verdict["exit_code"] == 1
    assert verdict["faultgate_version"] and verdict["created_utc"].endswith("Z")
    assert verdict["redaction"] == "builtin" and verdict["gate"]["verdict"] == "FAIL"
    inputs = {i["role"]: i for i in verdict["inputs"]}
    assert set(inputs) == {"traces", "band", "golden"} and all(len(i["sha256"]) == 64 for i in inputs.values())
    failing = json.loads((audit / "failing_traces.json").read_text(encoding="utf-8"))
    assert len(failing) == 21 and all(f["checks"] and f["checks"][0]["reason"] for f in failing)
    repro = (audit / "repro.sh").read_text(encoding="utf-8")
    assert repro.startswith("#!/bin/sh") and "faultgate check" in repro and "--band" in repro
    # zero retention: no trace payloads (prompts, answers) in the audit files
    prompt = load_otlp(GATE / "candidate_gpt.json")[0].prompt
    for f in audit.iterdir():
        assert prompt[:40] not in f.read_text(encoding="utf-8"), f.name
    assert f"markdown written to {md}" in capsys.readouterr().out


def test_markdown_without_band_and_with_policy_block(tmp_path):
    md = tmp_path / "pr.md"
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"max_calls": 0}))
    assert cli.main(["check", str(EXAMPLES), "--markdown", str(md)]) == 1
    text = md.read_text(encoding="utf-8")
    assert "## faultgate: FAIL" in text and "2/5 traces failed" in text and "### Policy" not in text
    assert cli.main(["check", str(GATE / "baseline_glm_run2.json"), "--band", str(GATE / "band.json"),
                     "--golden", str(GATE / "golden.json"), "--policy", str(pol), "--markdown", str(md)]) == 1
    text = md.read_text(encoding="utf-8")
    assert "### Policy: hard block" in text and "30 tool calls" not in text.split("### Policy")[0]


def test_markdown_pass_is_short(tmp_path):
    md = tmp_path / "pr.md"
    assert cli.main(["check", str(GATE / "baseline_glm_run2.json"), "--band", str(GATE / "band.json"),
                     "--golden", str(GATE / "golden.json"), "--markdown", str(md)]) == 0
    text = md.read_text(encoding="utf-8")
    assert "## faultgate: PASS" in text and "### Reproduce locally" in text


def test_markdown_masks_reasons(tmp_path):
    from tests.test_redact import test_cli_redact_flag  # noqa: F401  (same fixture idea)

    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"identity": {"arg": "q"}}))
    doc = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    for s in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]:
        if not s.get("parentSpanId"):
            s["attributes"].append({"key": "enduser.id", "value": {"stringValue": "jane@corp.example.com"}})
        for a in s["attributes"]:
            if a["key"] == "gen_ai.tool.call.arguments":
                a["value"] = {"stringValue": '{"q": "eve@evil.example"}'}
    tr = tmp_path / "t.json"
    tr.write_text(json.dumps(doc))
    md = tmp_path / "pr.md"
    cli.main(["check", str(tr), "--policy", str(pol), "--check", "policy", "--markdown", str(md)])
    assert "eve@evil" not in md.read_text(encoding="utf-8") and "<EMAIL>" in md.read_text(encoding="utf-8")


def test_action_posts_comment_and_summary():
    """The composite action writes the Markdown to the job summary and, on pull requests, upserts one PR comment."""
    action = (Path(__file__).resolve().parents[1] / "action.yml").read_text(encoding="utf-8")
    assert "--markdown" in action and "GITHUB_STEP_SUMMARY" in action
    assert "<!-- faultgate -->" in action and "pull_request" in action
    assert "--audit" in action and "upload-artifact" in action
    assert "comment:" in action


# --- review round 1 (Codex) ---

def test_json_report_includes_measures(tmp_path):
    out = tmp_path / "r.json"
    cli.main(["check", str(GATE / "baseline_glm_run1.json"), "--json", str(out)])
    m = json.loads(out.read_text())["measures"]
    assert len(m) == 30 and all(v["latency_ms"] > 0 for v in m.values())


def test_no_absolute_paths_in_markdown_or_audit(tmp_path, monkeypatch):
    """Inputs under the working directory are written relative to it; anything else by file name only."""
    outside = tmp_path / "secret-dir" / "candidate.json"
    outside.parent.mkdir()
    outside.write_bytes((GATE / "candidate_gpt.json").read_bytes())
    monkeypatch.chdir(GATE.parents[1])  # the repo root
    md, audit = tmp_path / "pr.md", tmp_path / "audit"
    cli.main(["check", str(outside), "--band", str(GATE / "band.json"), "--golden", str(GATE / "golden.json"),
              "--markdown", str(md), "--audit", str(audit)])
    texts = [md.read_text(encoding="utf-8")] + [f.read_text(encoding="utf-8") for f in audit.iterdir()]
    for t in texts:
        assert str(tmp_path) not in t and str(GATE.parents[1]) not in t and "secret-dir" not in t
        assert str(tmp_path).replace("\\", "\\\\") not in t
    assert "examples/gate/band.json" in texts[0] and "candidate.json" in texts[0]
    roles = {i["role"]: i["path"] for i in json.loads((audit / "verdict.json").read_text())["inputs"]}
    assert roles == {"traces": "candidate.json", "band": "examples/gate/band.json", "golden": "examples/gate/golden.json"}


def test_markdown_table_cells_are_escaped(tmp_path):
    doc = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    for s in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]:
        if not s.get("parentSpanId"):
            s["attributes"].append({"key": "faultgate.scenario", "value": {"stringValue": "a|b\n| injected | row |"}})
    tr = tmp_path / "t.json"
    tr.write_text(json.dumps(doc))
    md = tmp_path / "pr.md"
    cli.main(["check", str(tr), "--markdown", str(md)])
    text = md.read_text(encoding="utf-8")
    assert "a\|b \| injected \| row \|" in text
    assert not any(line.startswith("| injected") for line in text.splitlines())


# --- the composite action, executed: its bash steps run with fake faultgate / gh ---

import re
import shutil
import subprocess

ACTION = Path(__file__).resolve().parents[1] / "action.yml"


def _run_scripts() -> list[str]:
    """Each `run: |` block of action.yml, dedented."""
    lines = ACTION.read_text(encoding="utf-8").splitlines()
    blocks, i = [], 0
    while i < len(lines):
        m = re.match(r"^(\s*)run: \|\s*$", lines[i])
        if m:
            ind, body, i = len(m.group(1)), [], i + 1
            while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > ind):
                body.append(lines[i])
                i += 1
            pad = min(len(b) - len(b.lstrip()) for b in body if b.strip())
            blocks.append("\n".join(b[pad:] for b in body) + "\n")
        else:
            i += 1
    return blocks


FAKES = r'''
faultgate() {
  printf '%s\n' "$@" > "$FAKE_DIR/faultgate.args"
  while [ $# -gt 0 ]; do
    if [ "$1" = "--markdown" ]; then printf '<!-- faultgate -->\n## faultgate: FAIL\n' > "$2"; fi
    shift
  done
  return "${FAKE_CODE:-0}"
}
gh() {
  printf '%s\n' "$*" >> "$FAKE_DIR/gh.log"
  case "$*" in
    *--paginate*) [ -n "$FAKE_LOOKUP_FAIL" ] && return 1; [ -n "$FAKE_EXISTING" ] && echo "$FAKE_EXISTING"; return 0 ;;
  esac
}
pip() { printf '%s
' "$@" > "$FAKE_DIR/pip.args"; }
'''


def _bash(script: str, env: dict, tmp: Path) -> subprocess.CompletedProcess:
    (tmp / "fakes.sh").write_text(FAKES, encoding="utf-8")
    (tmp / "step.sh").write_text(script, encoding="utf-8")
    full = {**__import__("os").environ, **env, "BASH_ENV": str(tmp / "fakes.sh"), "FAKE_DIR": str(tmp)}
    return subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", str(tmp / "step.sh")],
                          env=full, capture_output=True, text=True, cwd=tmp)


needs_bash = __import__("pytest").mark.skipif(shutil.which("bash") is None, reason="needs bash")


def test_action_scripts_never_interpolate_inputs():
    for s in _run_scripts():
        assert "${{" not in s, s  # inputs reach the shell only as environment variables


@needs_bash
def test_action_gate_step_quotes_hostile_inputs_and_writes_outputs(tmp_path):
    scripts = _run_scripts()
    gate = next(s for s in scripts if "faultgate \"${args[@]}\"" in s)
    out, summ = tmp_path / "out", tmp_path / "summary"
    evil = 'x"; touch pwned; echo "'
    env = {"TRACES": evil, "REPORT": "r.json", "AUDIT": "audit", "FORMAT": "auto", "REDACT": "builtin",
           "BAND": "", "GOLDEN": "", "POLICY": "", "CHECKS": "", "JUDGE": "",
           "GITHUB_OUTPUT": str(out), "GITHUB_STEP_SUMMARY": str(summ), "RUNNER_TEMP": str(tmp_path), "FAKE_CODE": "1"}
    p = _bash(gate, env, tmp_path)
    assert p.returncode == 0, p.stderr
    assert not (tmp_path / "pwned").exists()
    args = (tmp_path / "faultgate.args").read_text().splitlines()
    assert args[:2] == ["check", evil]
    outputs = out.read_text()
    assert "code=1" in outputs and "verdict=FAIL" in outputs and "summary=" in outputs
    assert "## faultgate: FAIL" in summ.read_text()


@needs_bash
def test_action_comment_step_updates_creates_and_degrades(tmp_path):
    comment = next(s for s in _run_scripts() if "gh api" in s)
    md = tmp_path / "s.md"
    md.write_text("<!-- faultgate -->\nhi\n")
    base = {"PR": "7", "REPO": "o/r", "MD": str(md), "GH_TOKEN": "x"}
    assert _bash(comment, {**base, "FAKE_EXISTING": "123"}, tmp_path).returncode == 0
    log = (tmp_path / "gh.log").read_text()
    assert "-X PATCH repos/o/r/issues/comments/123 -F body=@" in log and "POST" not in log
    (tmp_path / "gh.log").unlink()
    assert _bash(comment, base, tmp_path).returncode == 0
    assert "-X POST repos/o/r/issues/7/comments -F body=@" in (tmp_path / "gh.log").read_text()
    p = _bash(comment, {**base, "FAKE_LOOKUP_FAIL": "1"}, tmp_path)
    assert p.returncode == 0 and "::warning::" in p.stdout


@needs_bash
def test_action_final_step_fails_closed(tmp_path):
    final = next(s for s in _run_scripts() if "exit" in s and "CODE" in s)
    assert _bash(final, {"CODE": ""}, tmp_path).returncode == 2
    assert _bash(final, {"CODE": "0"}, tmp_path).returncode == 0
    assert _bash(final, {"CODE": "1"}, tmp_path).returncode == 1


@needs_bash
@__import__("pytest").mark.parametrize("version, spec", [
    ("0.3.0", "faultgate==0.3.0"),          # the client's input: pip read it as a package called 0.3.0
    ("v0.3.0", "faultgate==0.3.0"),
    (">=0.3", "faultgate>=0.3"),
    ("==0.3.0", "faultgate==0.3.0"),
    ("faultgate==0.3.0", "faultgate==0.3.0"),
    ("faultgate", "faultgate"),
    (".", "."),
    ("git+https://github.com/Faultline-lab/faultgate@main", "git+https://github.com/Faultline-lab/faultgate@main"),
])
def test_action_version_input_becomes_a_pip_requirement(tmp_path, version, spec):
    install = next(s for s in _run_scripts() if "pip install" in s)
    assert _bash(install, {"VERSION": version}, tmp_path).returncode == 0
    assert (tmp_path / "pip.args").read_text().splitlines() == ["install", "--quiet", spec]
