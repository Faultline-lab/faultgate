"""Tolerance bands: is this run a regression, or just noise?

Ported from FAULTLINE P11 (faultline_p2/gate/band.py, gate.py), where five
baseline runs of the same model on identical inputs scored 0.60–0.87 — the
case against a fixed threshold. A band is built from >= 2 baseline runs of the
same scenario set (3 or more recommended); a candidate is judged against the
*observed* spread:

    PASS          clean_rate >= min_observed - 1/n
    FAIL          clean_rate <  min_observed - 3/n, or below PASS and the drop
                  is significant (one-sided test, alpha 0.05)
    INCONCLUSIVE  otherwise, with the number of scenarios needed to tell
                  (power 0.80)

The test compares the candidate's clean rate with the baseline mean; its
variance adds the binomial noise of both sides to the run-to-run variance the
baseline measured, so a noisy baseline needs a bigger drop to FAIL.

Plus the list of scenarios that were clean in every baseline and are not now.

A policy violation is a hard block: any trace where the ``policy`` check fired
makes the verdict FAIL whatever the clean rate. Noise explains a flaky answer;
it does not explain a refund over the limit.
"""
from __future__ import annotations

import math
from statistics import NormalDist, variance
from typing import Any

from faultgate.runner import Report

MIN_RUNS = 2
RECOMMENDED_RUNS = 3
ALPHA = 0.05
POWER = 0.80


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
        "rule": f"PASS if clean_rate >= min - 1/n; FAIL if clean_rate < min - 3/n or the drop from the baseline mean "
                f"is significant (one-sided, alpha {ALPHA}); else INCONCLUSIVE with the n needed for power {POWER}",
    }


def stats(rate: float, n: int, band: dict[str, Any]) -> dict[str, Any]:
    """One-sided test that the candidate's clean rate is below the baseline mean, and the n needed to tell.

    Variance = baseline binomial (runs x n trials) + candidate binomial (n trials) + run-to-run variance.
    ``required_n`` is the two-proportion sample size per arm for the observed drop, or None when there is no drop.
    """
    per_run = band["clean_rate"]["per_run"]
    runs = len(per_run)
    p_b = sum(per_run) / runs
    between = variance(per_run) if runs > 1 else 0.0
    se = math.sqrt(p_b * (1 - p_b) / (runs * n) + rate * (1 - rate) / n + between)
    drop = p_b - rate
    if se == 0:
        p_value = 0.0 if drop > 0 else 1.0
    else:
        p_value = 1 - NormalDist().cdf(drop / se)
    required = None
    if drop > 0:
        z_a = NormalDist().inv_cdf(1 - ALPHA)
        z_b = NormalDist().inv_cdf(POWER)
        p_bar = (p_b + rate) / 2
        num = z_a * math.sqrt(2 * p_bar * (1 - p_bar)) + z_b * math.sqrt(p_b * (1 - p_b) + rate * (1 - rate))
        required = math.ceil((num / drop) ** 2 - 1e-9)
    return {"baseline_mean": p_b, "drop": drop, "p_value": p_value, "alpha": ALPHA, "power": POWER,
            "required_n": required, "additional_n": max(0, required - n) if required is not None else None}


def evaluate(report: Report, band: dict[str, Any]) -> dict[str, Any]:
    clean = report.clean_by_scenario
    scenarios = band["scenarios"]
    n = len(scenarios)
    missing = [s for s in scenarios if s not in clean]
    rate = sum(clean.get(s, False) for s in scenarios) / n
    lo = band["clean_rate"]["min"]
    eps = 1e-9
    violations = report.fired_on("policy")
    st = stats(rate, n, band)
    if violations or rate < lo - 3 / n - eps:
        verdict = "FAIL"
    elif rate >= lo - 1 / n - eps:
        verdict = "PASS"
    elif st["p_value"] < ALPHA:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"
    if verdict != "INCONCLUSIVE":
        st["required_n"] = st["additional_n"] = None
    warnings = []
    if report.judge != band["judge"] or report.checks != band["checks"]:
        warnings.append(f"band was built with judge={band['judge']} checks={band['checks']}; this run used judge={report.judge} checks={report.checks}")
    if missing:
        warnings.append(f"{len(missing)} band scenario(s) absent from this run, counted as not clean")
    if len(band["clean_rate"]["per_run"]) < RECOMMENDED_RUNS:
        warnings.append(f"band built from {len(band['clean_rate']['per_run'])} baseline runs; use {RECOMMENDED_RUNS} or more "
                        "for a stable noise estimate")
    return {
        "verdict": verdict,
        "clean_rate": rate,
        "band": [lo, band["clean_rate"]["max"]],
        "pass_threshold": lo - 1 / n,
        "fail_threshold": lo - 3 / n,
        "stats": st,
        "regressions": [s for s in band["clean_in_all_runs"] if not clean.get(s, False)],
        "missing": missing,
        "policy_violations": violations,
        "warnings": warnings,
    }
