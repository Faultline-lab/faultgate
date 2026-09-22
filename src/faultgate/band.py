"""Tolerance bands: is this run a regression, or just noise?

Ported from FAULTLINE P11 (faultline_p2/gate/band.py, gate.py), where five
baseline runs of the same model on identical inputs scored 0.60–0.87 — the
case against a fixed threshold. A band is built from >= 2 baseline runs of the
same scenario set; a candidate is judged against the *observed* spread:

    PASS  clean_rate >= min_observed - 1/n
    FAIL  clean_rate <  min_observed - 3/n
    WARN  otherwise

plus the list of scenarios that were clean in every baseline and are not now.
"""
from __future__ import annotations

from typing import Any

from faultgate.runner import Report

MIN_RUNS = 2


def build(reports: list[Report]) -> dict[str, Any]:
    if len(reports) < MIN_RUNS:
        raise ValueError(f"a band needs at least {MIN_RUNS} baseline runs, got {len(reports)}")
    judges = {r.judge for r in reports}
    checks = {tuple(r.checks) for r in reports}
    if len(judges) > 1 or len(checks) > 1:
        raise ValueError("all baseline runs must use the same judge and checks")

    per_run = [r.clean_by_scenario for r in reports]
    scenarios = sorted(set.intersection(*(set(m) for m in per_run)))
    if not scenarios:
        raise ValueError("baseline runs share no scenarios (keys come from faultgate.scenario, gen_ai.agent.id or the prompt)")
    n = len(scenarios)
    rates = [sum(m[s] for s in scenarios) / n for m in per_run]
    agreement = {s: sum(m[s] for m in per_run) for s in scenarios}
    return {
        "judge": reports[0].judge,
        "checks": reports[0].checks,
        "runs": len(reports),
        "n": n,
        "scenarios": scenarios,
        "clean_rate": {"min": min(rates), "max": max(rates), "mean": sum(rates) / len(rates), "per_run": rates},
        "clean_in_all_runs": [s for s in scenarios if agreement[s] == len(reports)],
        "rule": "PASS if clean_rate >= min - 1/n; FAIL if clean_rate < min - 3/n; else WARN",
    }


def evaluate(report: Report, band: dict[str, Any]) -> dict[str, Any]:
    clean = report.clean_by_scenario
    scenarios = band["scenarios"]
    n = len(scenarios)
    missing = [s for s in scenarios if s not in clean]
    rate = sum(clean.get(s, False) for s in scenarios) / n
    lo = band["clean_rate"]["min"]
    eps = 1e-9
    if rate < lo - 3 / n - eps:
        verdict = "FAIL"
    elif rate >= lo - 1 / n - eps:
        verdict = "PASS"
    else:
        verdict = "WARN"
    warnings = []
    if report.judge != band["judge"] or report.checks != band["checks"]:
        warnings.append(f"band was built with judge={band['judge']} checks={band['checks']}; this run used judge={report.judge} checks={report.checks}")
    if missing:
        warnings.append(f"{len(missing)} band scenario(s) absent from this run, counted as not clean")
    return {
        "verdict": verdict,
        "clean_rate": rate,
        "band": [lo, band["clean_rate"]["max"]],
        "pass_threshold": lo - 1 / n,
        "fail_threshold": lo - 3 / n,
        "regressions": [s for s in band["clean_in_all_runs"] if not clean.get(s, False)],
        "missing": missing,
        "warnings": warnings,
    }
