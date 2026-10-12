"""What a gate run leaves behind: a Markdown PR comment and audit files.

Both are built from the report and gate result only, never from trace payloads:
reasons are already masked (``--redact``), and prompts, answers and tool data are
not copied out. The audit directory is what an external reviewer needs to replay
the decision: ``verdict.json`` (verdict, exit code, version, inputs with SHA-256,
gate numbers), ``failing_traces.json`` (each failing trace and every reason), and
``repro.sh`` (the exact command).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from faultgate import __version__
from faultgate.runner import Report

MARKER = "<!-- faultgate -->"
MAX_ROWS = 20


def verdict_of(report: Report, gate: dict[str, Any] | None) -> str:
    if gate:
        return gate["verdict"]
    return "PASS" if report.passed else "FAIL"


def _failing(report: Report) -> list[dict[str, Any]]:
    out = []
    for tid in report.failed_traces:
        rows = [r for r in report.results if r.trace_id == tid]
        out.append({"trace_id": tid, "scenario": rows[0].scenario,
                    "checks": [{"check": r.check, "reason": r.verdict.reason} for r in rows if r.verdict.detected]})
    return out


def _cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")[:160]


def _fmt(measure: str, v: float) -> str:
    if measure == "latency_ms":
        return f"{v / 1000:.1f} s"
    if measure == "cost_usd":
        return f"${v:.4f}"
    return f"{v:,.0f}"


LABELS = {"tokens": "tokens / run", "latency_ms": "latency / run", "cost_usd": "cost / run"}


def markdown(report: Report, gate: dict[str, Any] | None, band: dict[str, Any] | None, command: str) -> str:
    v = verdict_of(report, gate)
    n = len(report.trace_ids)
    lines = [MARKER, f"## faultgate: {v}", ""]
    if gate:
        s = gate["stats"]
        lines.append(f"**Gate:** clean rate {gate['clean_rate']:.2f} vs band [{gate['band'][0]:.2f}, {gate['band'][1]:.2f}] "
                     f"from {band['runs'] if band else '?'} baseline runs · fail below {gate['fail_threshold']:.2f} · p={s['p_value']:.3f}")
        if v == "INCONCLUSIVE":
            lines.append(f"Not enough evidence: a drop of {s['drop']:.2f} needs {s['required_n']} scenarios "
                         f"(α {s['alpha']}, power {s['power']:.2f}); run {s['additional_n']} more.")
        if gate["regressions"]:
            shown = ", ".join(f"`{r}`" for r in gate["regressions"][:MAX_ROWS])
            more = f" and {len(gate['regressions']) - MAX_ROWS} more" if len(gate["regressions"]) > MAX_ROWS else ""
            lines.append(f"**Regressions** (clean in every baseline run, not now): {shown}{more}")
        lines.append("")
    lines.append(f"{len(report.failed_traces)}/{n} traces failed · judge {report.judge} · redaction {report.redaction}")
    lines += ["", "| check | fired |", "|---|---|"]
    lines += [f"| {c} | {k} |" for c, k in report.fired.items()]
    violations = gate["policy_violations"] if gate else report.fired_on("policy")
    if violations:
        lines += ["", f"### Policy: hard block", "",
                  f"Policy violated in {len(violations)} trace(s); the gate fails whatever the clean rate.", "",
                  "| trace | reason |", "|---|---|"]
        for r in report.results:
            if r.check == "policy" and r.verdict.detected and len(lines) < 200:
                lines.append(f"| `{r.trace_id[:12]}` | {_cell(r.verdict.reason)} |")
    failing = _failing(report)
    if failing:
        lines += ["", f"### Failing traces" + (f" (first {MAX_ROWS} of {len(failing)})" if len(failing) > MAX_ROWS else ""), "",
                  "| scenario | trace | check | reason |", "|---|---|---|---|"]
        for f in failing[:MAX_ROWS]:
            for c in f["checks"]:
                lines.append(f"| {f['scenario']} | `{f['trace_id'][:12]}` | {c['check']} | {_cell(c['reason'])} |")
    usage = (gate or {}).get("usage") or {}
    if usage:
        lines += ["", "### Latency and cost vs baseline", "", "| | baseline | this run | Δ |", "|---|---|---|---|"]
        for m, u in usage.items():
            lines.append(f"| {LABELS.get(m, m)} | {_fmt(m, u['baseline'])} | {_fmt(m, u['candidate'])} | {u['delta_pct']:+.1f}% |")
    lines += ["", "### Reproduce locally", "", "```sh", f"pip install 'faultgate=={__version__}'", command, "```", ""]
    return "\n".join(lines)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_audit(out_dir: str | Path, report: Report, gate: dict[str, Any] | None, inputs: dict[str, str | None],
                command: str, exit_code: int) -> None:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    verdict = {
        "faultgate_version": __version__,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verdict": verdict_of(report, gate),
        "exit_code": exit_code,
        "command": command,
        "inputs": [{"role": role, "path": p, "sha256": _sha256(p)} for role, p in inputs.items() if p],
        "judge": report.judge,
        "checks": report.checks,
        "redaction": report.redaction,
        "traces": len(report.trace_ids),
        "failed_traces": len(report.failed_traces),
        "fired": report.fired,
        "gate": gate,
    }
    (d / "verdict.json").write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    (d / "failing_traces.json").write_text(json.dumps(_failing(report), indent=1), encoding="utf-8")
    (d / "repro.sh").write_text(f"#!/bin/sh\n# faultgate {__version__} · verdict {verdict['verdict']} · exit {exit_code}\n"
                                f"pip install 'faultgate=={__version__}'\n{command}\n", encoding="utf-8")
