"""Compare two sets of runs: is the candidate better or worse? (M2 + M3 of the compare spec.)

One engine for every dimension: a two-level paired bootstrap (scenarios, then runs within a
scenario), Bonferroni-corrected CIs, five gateable statistical dimensions plus a hard policy
block. Standard library only; nothing leaves the machine.
"""
from __future__ import annotations

import difflib
import json
import math
import random
from pathlib import Path
from statistics import NormalDist
from typing import Any

from faultgate import __version__

DIMENSIONS = ("correctness", "consistency", "approach", "cost", "speed")  # statistical, gateable
BUDGET_KEYS = DIMENSIONS + ("speed_p95",)
DEFAULT_BUDGET = {"approach": 0.10, "cost": 0.05, "speed": 0.10, "speed_p95": 0.20}
RELATIVE = {"approach", "cost", "speed"}  # change is (cand - base) / base; the rest absolute
UNITS = {"correctness": "rate", "consistency": "rate", "approach": "calls", "speed": "ms", "speed_p95": "ms"}
ALPHA = 0.05
POWER = 0.80
EPS = 1e-12


def _load_json(path: str | Path, what: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"{what}: cannot read {path}: {e}") from None


def _number(where: str, field: str, v: Any, minimum: float) -> None:
    op = ">= 0" if minimum == 0 else ">= 1"
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < minimum:
        raise ValueError(f"{where}: {field} must be a finite number {op}")


def upper_bound_95(violations: int, n: int) -> float | None:
    """One-sided 95% upper bound on the violation rate after 0 violations in n runs; None if any was seen."""
    return round(1 - 0.05 ** (1 / n), 4) if violations == 0 else None


def load_budget(path: str | Path) -> dict[str, dict[str, Any]]:
    """budget.json -> {dimension: {"budget": float, "gate": bool}}; ValueError (one line) on any violation."""
    doc = _load_json(path, "budget.json")
    if not isinstance(doc, dict):
        raise ValueError(f"budget.json: expected a JSON object of dimension -> {{budget, gate}}")
    out: dict[str, dict[str, Any]] = {}
    for name, entry in doc.items():
        if name not in BUDGET_KEYS:
            raise ValueError(f"budget.json: unknown key {name!r} (allowed: {', '.join(BUDGET_KEYS)})")
        if not isinstance(entry, dict) or set(entry) - {"budget", "gate"}:
            raise ValueError(f"budget.json: {name!r} must be an object with only 'budget' and/or 'gate'")
        got: dict[str, Any] = {}
        if "budget" in entry:
            _number("budget.json", f"{name}.budget", entry["budget"], 0)
            got["budget"] = float(entry["budget"])
        if "gate" in entry:
            if not isinstance(entry["gate"], bool):
                raise ValueError(f"budget.json: {name}.gate must be true or false")
            got["gate"] = entry["gate"]
        out[name] = got
    return out


def load_prices(path: str | Path) -> dict[str, dict[str, float]]:
    """prices.json -> {model: {input_per_m, output_per_m} | {gpu_hour_usd, concurrency}}; ValueError otherwise."""
    doc = _load_json(path, "prices.json")
    if not isinstance(doc, dict):
        raise ValueError("prices.json: expected a JSON object of model -> price")
    token_keys, gpu_keys = {"input_per_m", "output_per_m"}, {"gpu_hour_usd", "concurrency"}
    for model, p in doc.items():
        if not isinstance(p, dict):
            raise ValueError(f"prices.json: {model!r} must be an object")
        if set(p) == token_keys:
            for k in token_keys:
                _number("prices.json", f"{model!r}.{k}", p[k], 0)
        elif set(p) == gpu_keys:
            _number("prices.json", f"{model!r}.gpu_hour_usd", p["gpu_hour_usd"], 0)
            _number("prices.json", f"{model!r}.concurrency", p["concurrency"], 1)
        else:
            raise ValueError(f"prices.json: {model!r} needs input_per_m+output_per_m or gpu_hour_usd+concurrency")
    return doc


def _mean_of(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _scenario_mean(table: dict[str, list[float | None]], scen: list[str]) -> float | None:
    """Mean over scenarios of each scenario's mean of its runs; None when a side measured nothing."""
    means = [m for s in scen if (m := _mean_of(table[s])) is not None]
    return sum(means) / len(means) if means else None


def _pass_k(values: list[float], k: int) -> float:
    """Unbiased pass^k = C(c, k) / C(r, k)."""
    return math.comb(int(sum(values)), k) / math.comb(len(values), k)


def _percentile(vals: list[float], p: float) -> float | None:
    """Linear-interpolated percentile of a pooled list; None when empty."""
    if not vals:
        return None
    v = sorted(vals)
    if len(v) == 1:
        return v[0]
    at = (len(v) - 1) * p / 100
    lo, hi = math.floor(at), math.ceil(at)
    return v[lo] + (v[hi] - v[lo]) * (at - lo)


def _clean(x: Any) -> Any:
    """Every float rounded to 6 places, so json.dumps(sort_keys=True) is byte-stable."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    return round(float(x), 6)


def _trace_cost(trace: Any, measure: dict[str, Any],
                prices: dict[str, dict[str, float]]) -> tuple[float | None, str | None, bool]:
    """Cost of one trace from the first source that exists: (value, unit, used the self-hosted estimate)."""
    if trace.cost_usd is not None:
        return float(trace.cost_usd), "usd", False
    p = prices.get(trace.model) if trace.model else None
    if p and "input_per_m" in p:
        usage = trace.usage
        if usage:
            value = (usage.get("input_tokens", 0) * p["input_per_m"]
                     + usage.get("output_tokens", 0) * p["output_per_m"]) / 1e6
            return value, "usd", False
    if p and "gpu_hour_usd" in p and trace.latency_ms is not None:
        return trace.latency_ms / 3.6e6 * p["gpu_hour_usd"] / p["concurrency"], "usd", True
    tokens = measure.get("tokens")
    return (float(tokens), "tokens", False) if tokens is not None else (None, None, False)


def _resolve_cost(side: list[tuple[Any, list[Any]]],
                  prices: dict[str, dict[str, float]] | None) -> dict[str, Any]:
    """One unit for a whole side: usd if every measured trace is usd, else tokens if every trace has them."""
    raw: dict[str, tuple[float | None, str | None]] = {}
    tokens: dict[str, float | None] = {}
    models: set[str] = set()
    self_hosted = False
    for report, traces in side:
        by_id = {t.trace_id: t for t in traces}
        for tid in report.trace_ids:
            measure = report.measures.get(tid, {})
            tokens[tid] = measure.get("tokens")
            trace = by_id.get(tid)
            if trace is None:
                raw[tid] = (None, None)
                continue
            if trace.model:
                models.add(trace.model)
            value, unit, estimate = _trace_cost(trace, measure, prices or {})
            self_hosted = self_hosted or estimate
            raw[tid] = (value, unit)
    valued = [(v, u) for v, u in raw.values() if v is not None]
    if valued and all(u == "usd" for _, u in valued):
        unit, values = "usd", {tid: v for tid, (v, _) in raw.items()}
    elif raw and all(tokens.get(tid) is not None for tid in raw):
        unit, values = "tokens", {tid: tokens[tid] for tid in raw}
    else:
        unit, values = "usd", {tid: v if u == "usd" else None for tid, (v, u) in raw.items()}
    return {"unit": unit, "values": values, "models": models, "self_hosted": self_hosted}


def _side_view(side: list[tuple[Any, list[Any]]], scen: list[str],
               prices: dict[str, dict[str, float]] | None) -> dict[str, Any]:
    cost = _resolve_cost(side, prices)
    tables: dict[str, dict[str, list[float | None]]] = {d: {s: [] for s in scen} for d in DIMENSIONS}
    clean_runs, seqs = [], []
    for report, traces in side:
        by_id = {t.trace_id: t for t in traces}
        run_ids = {s: t for t, s in report.scenario_of.items()}  # ponytail: repeated scenario -> last trace
        clean = report.clean_by_scenario
        clean_runs.append(clean)
        seqs.append({s: tuple(st.tool for st in by_id[t].steps) for s, t in run_ids.items() if t in by_id})
        for s in scen:
            tid = run_ids.get(s)
            measure = report.measures.get(tid, {}) if tid is not None else {}
            tables["correctness"][s].append(1.0 if tid is not None and clean.get(s, False) else 0.0)
            tables["approach"][s].append(measure.get("steps") if tid is not None else None)
            tables["speed"][s].append(measure.get("latency_ms") if tid is not None else None)
            tables["cost"][s].append(cost["values"].get(tid) if tid is not None else None)
    tables["consistency"] = {s: list(v) for s, v in tables["correctness"].items()}
    return {"tables": tables, "cost": cost, "clean": clean_runs, "seqs": seqs}


def compare(baseline: list[tuple[Any, list[Any]]], candidate: list[tuple[Any, list[Any]]],
            budget: dict | None = None, prices: dict | None = None,
            resamples: int = 2000, seed: int = 0) -> dict:
    """Two-level paired bootstrap over scenarios then runs; returns the JSON-ready result dict.

    ``baseline`` / ``candidate``: one (Report, traces) pair per run file. Raise ValueError (-> CLI exit 2)
    for fewer than 2 runs a side, mismatched checks/judge, no shared scenarios, or resamples < 200.
    """
    if len(baseline) < 2 or len(candidate) < 2:
        raise ValueError(f"compare needs at least 2 runs on each side, got baseline {len(baseline)}, "
                         f"candidate {len(candidate)}")
    if resamples < 200:
        raise ValueError(f"--resamples must be 200 or more, got {resamples}")
    reports = [r for r, _ in baseline] + [r for r, _ in candidate]
    if len({r.judge for r in reports}) > 1 or len({tuple(r.checks) for r in reports}) > 1:
        raise ValueError("all runs on both sides must use the same judge and checks")
    scen = sorted(set.intersection(*(set(r.clean_by_scenario) for r, _ in baseline)))
    if not scen:
        raise ValueError("baseline runs share no scenarios (keys come from faultgate.scenario, "
                         "gen_ai.agent.id or the prompt)")
    cand_union = set().union(*(set(r.clean_by_scenario) for r, _ in candidate))
    if not set(scen) & cand_union:
        raise ValueError("baseline and candidate share no scenarios (keys come from faultgate.scenario, "
                         "gen_ai.agent.id or the prompt)")
    n, rb, rc = len(scen), len(baseline), len(candidate)
    k = min(rb, rc)
    base = _side_view(baseline, scen, prices)
    cand = _side_view(candidate, scen, prices)

    warnings: list[str] = []
    missing = sorted(({"run": i, "scenario": s} for i, (r, _) in enumerate(candidate, 1)
                      for s in scen if s not in r.scenario_of.values()), key=lambda m: (m["run"], m["scenario"]))
    if missing:
        warnings.append(f"{len(missing)} scenario-run(s) in the baseline are absent from candidate runs, "
                        "counted as failed")

    conf = {name: dict(v) for name, v in (budget or {}).items()}

    def budget_of(d: str) -> float:
        t = conf.get(d, {}).get("budget")
        if t is None:
            t = 1.0 / n if d in ("correctness", "consistency") else DEFAULT_BUDGET[d]
        return float(t)

    def gate_of(d: str) -> bool:
        return bool(conf.get(d, {}).get("gate", True))

    unit_mismatch = base["cost"]["unit"] != cand["cost"]["unit"]
    active = {d: any(v is not None for s in scen for v in base["tables"][d][s])
                 and any(v is not None for s in scen for v in cand["tables"][d][s]) for d in DIMENSIONS}
    if unit_mismatch:
        active["cost"] = False

    g = max(1, sum(1 for d in DIMENSIONS if active[d] and gate_of(d)))
    alpha = ALPHA / g
    z = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(POWER)
    means = {d: (_scenario_mean(base["tables"][d], scen), _scenario_mean(cand["tables"][d], scen))
             for d in DIMENSIONS}
    means["consistency"] = (_scenario_mean(_pass_k_table(base, scen, k), scen),
                            _scenario_mean(_pass_k_table(cand, scen, k), scen))

    rng = random.Random(seed)
    draws = [[(rng.randrange(n), [rng.randrange(rb) for _ in range(rb)], [rng.randrange(rc) for _ in range(rc)])
              for _ in range(n)] for _ in range(resamples)]
    results: dict[str, dict[str, Any]] = {}
    for d in DIMENSIONS:
        t = budget_of(d)
        if not active[d]:
            unit = None if d == "cost" else UNITS[d]
            results[d] = {"baseline": None, "candidate": None, "delta": None, "ci": None, "budget": t,
                          "gates": False, "label": "n/a", "scenarios_needed": None, "unit": unit, "note": None}
            if d == "cost" and unit_mismatch:
                bad = "candidate" if base["cost"]["unit"] == "usd" else "baseline"
                other = "baseline" if bad == "candidate" else "candidate"
                side = cand if bad == "candidate" else base
                names = f": {', '.join(sorted(side['cost']['models']))}" if side["cost"]["models"] else ""
                warnings.append(f"cost n/a: {other} is in USD, {bad} only has tokens; give --prices for the "
                                f"{bad} model(s){names}")
            else:
                warnings.append(f"{d} n/a: missing on one side (every value is None), not gated")
            if d == "speed":
                results[d]["p50"] = {"baseline": None, "candidate": None}
                results[d]["p95"] = {"baseline": None, "candidate": None}
            continue

        bt, ct = base["tables"][d], cand["tables"][d]
        pairs: list[tuple[float, float]] = []
        for slots in draws:
            bs: list[float] = []
            cs: list[float] = []
            for si, bi, ci in slots:
                s = scen[si]
                if d == "consistency":
                    bs.append(math.comb(int(sum(bt[s][j] for j in bi)), k) / math.comb(rb, k))
                    cs.append(math.comb(int(sum(ct[s][j] for j in ci)), k) / math.comb(rc, k))
                    continue
                bv = [bt[s][j] for j in bi if bt[s][j] is not None]
                cv = [ct[s][j] for j in ci if ct[s][j] is not None]
                if bv and cv:
                    bs.append(sum(bv) / len(bv))
                    cs.append(sum(cv) / len(cv))
            # ponytail: a resample that drew no non-None value on a side falls back to the full-data
            # mean instead of dropping the resample; only reachable for sparse cost/approach/speed data
            pairs.append((sum(bs) / len(bs) if bs else means[d][0],
                          sum(cs) / len(cs) if cs else means[d][1]))

        base0, cand0 = means[d]
        relative = d in RELATIVE
        absolute = not relative or base0 == 0 or any(b == 0 for b, _ in pairs)
        deltas = [(c - b) if absolute else (c - b) / b for b, c in pairs]
        delta0 = (cand0 - base0) if absolute else (cand0 - base0) / base0
        higher_better = d in ("correctness", "consistency")
        if higher_better:
            deltas, w0 = [-x for x in deltas], -delta0
        else:
            w0 = delta0
        ordered = sorted(deltas)
        lo = ordered[math.floor(alpha / 2 * (resamples - 1))]
        hi = ordered[math.ceil((1 - alpha / 2) * (resamples - 1))]
        ci = [-hi, -lo] if higher_better else [lo, hi]
        needed: int | str | None = None
        if lo > t:
            label = "WORSE"
        elif hi < 0:
            label = "BETTER"
        elif hi <= t:
            label = "SAME"
        else:
            label = "INCONCLUSIVE"
            half = (hi - lo) / 2
            gap = abs(w0 - t)
            if gap < EPS:
                needed = "cannot tell at any n"
            else:
                needed = math.ceil(n * ((z + z_power) * (half / z) / gap) ** 2)
        notes = []
        if relative and absolute:
            notes.append("absolute change: baseline mean is 0")
            warnings.append(f"{d}: absolute change used because the baseline mean is 0")
        if d == "cost":
            if base["cost"]["self_hosted"] or cand["cost"]["self_hosted"]:
                notes.append("self-hosted cost is an estimate")
            if base["cost"]["unit"] == "tokens" and cand["cost"]["unit"] == "tokens":
                bmodels, cmodels = sorted(base["cost"]["models"]), sorted(cand["cost"]["models"])
                if bmodels != cmodels and not prices:
                    warnings.append(f"tokens are not comparable across tokenizers (baseline: {', '.join(bmodels)}, "
                                    f"candidate: {', '.join(cmodels)}); give --prices to compare in money")
        unit = ("usd" if base["cost"]["unit"] == "usd" else "tokens (no price)") if d == "cost" else UNITS[d]
        results[d] = {"baseline": base0, "candidate": cand0, "delta": delta0, "ci": ci, "budget": t,
                      "gates": gate_of(d), "label": label, "scenarios_needed": needed, "unit": unit,
                      "note": "; ".join(notes) or None}
        if d == "speed":
            lat_b = [m["latency_ms"] for r, _ in baseline for m in r.measures.values()
                     if m.get("latency_ms") is not None]
            lat_c = [m["latency_ms"] for r, _ in candidate for m in r.measures.values()
                     if m.get("latency_ms") is not None]
            results[d]["p50"] = {"baseline": _percentile(lat_b, 50), "candidate": _percentile(lat_c, 50)}
            results[d]["p95"] = {"baseline": _percentile(lat_b, 95), "candidate": _percentile(lat_c, 95)}

    sp95 = conf.get("speed_p95", {})
    if sp95.get("gate"):
        t95 = sp95["budget"] if sp95.get("budget") is not None else DEFAULT_BUDGET["speed_p95"]
        lat_b = _latencies(baseline)
        lat_c = _latencies(candidate)
        b95, c95 = _percentile(lat_b, 95), _percentile(lat_c, 95)
        note = "low confidence: pooled p95, point estimate, no CI"
        if b95 is None or c95 is None:
            results["speed_p95"] = {"baseline": b95, "candidate": c95, "delta": None, "ci": None,
                                    "budget": float(t95), "gates": False, "label": "n/a",
                                    "scenarios_needed": None, "unit": "ms", "note": note}
        else:
            delta = (c95 - b95) / b95 if b95 else c95 - b95
            results["speed_p95"] = {"baseline": b95, "candidate": c95, "delta": delta, "ci": None,
                                    "budget": float(t95), "gates": True,
                                    "label": "WORSE" if delta > t95 else "SAME", "scenarios_needed": None,
                                    "unit": "ms", "note": note}

    violations = sum(len(r.fired_on("policy")) for r, _ in candidate)
    n_traces = sum(len(r.trace_ids) for r, _ in candidate)
    safety = {"violations": violations, "runs": n_traces, "upper_bound_95": upper_bound_95(violations, n_traces),
              "label": "BLOCK" if violations else "PASS"}
    gated = [e for e in results.values() if e["gates"] and e["label"] != "n/a"]
    blocked = violations > 0 or any(e["label"] == "WORSE" for e in gated)
    verdict = "BLOCK" if blocked else ("REVIEW" if any(e["label"] == "INCONCLUSIVE" for e in gated) else "SHIP")

    cps_b, cps_c = (None, None) if unit_mismatch else (_per_solved(base, baseline), _per_solved(cand, candidate))
    cost_per_solved = {"baseline": cps_b, "candidate": cps_c,
                       "delta_pct": (100 * (cps_c - cps_b) / cps_b) if cps_b and cps_c is not None else None}

    regressions = sorted(s for s in scen if all(clean.get(s, False) for clean in base["clean"])
                         and not all(clean.get(s, False) for clean in cand["clean"]))
    check_deltas = {}
    for c in reports[0].checks:
        b = sum(r.fired.get(c, 0) for r, _ in baseline) / rb
        cc = sum(r.fired.get(c, 0) for r, _ in candidate) / rc
        check_deltas[c] = {"baseline": b, "candidate": cc, "delta": cc - b}
    ratios = []
    for s in scen:
        bs = [seq[s] for seq in base["seqs"] if s in seq]
        cs = [seq[s] for seq in cand["seqs"] if s in seq]
        if bs and cs:
            ratios.append(sum(difflib.SequenceMatcher(None, a, b).ratio() for a in bs for b in cs) / (len(bs) * len(cs)))
    trajectory_similarity = sum(ratios) / len(ratios) if ratios else None
    base_tools = {tool for seq in base["seqs"] for tools in seq.values() for tool in tools}
    cand_tools = {tool for seq in cand["seqs"] for tools in seq.values() for tool in tools}
    policy_violations = sorted(({"run": i, "scenario": r.scenario_of[tid]} for i, (r, _) in enumerate(candidate, 1)
                                for tid in r.fired_on("policy")), key=lambda x: (x["run"], x["scenario"]))

    out = {
        "verdict": verdict, "exit_code": 1 if blocked else 0, "seed": seed, "resamples": resamples,
        "ci_level": 1 - alpha, "n": n, "runs": {"baseline": rb, "candidate": rc}, "dimensions": results,
        "safety": safety, "cost_per_solved": cost_per_solved, "trajectory_similarity": trajectory_similarity,
        "new_tools": sorted(cand_tools - base_tools), "regressions": regressions, "missing": missing,
        "policy_violations": policy_violations, "check_deltas": check_deltas, "warnings": warnings,
        "faultgate_version": __version__,
    }
    return _clean(out)


def _pass_k_table(view: dict[str, Any], scen: list[str], k: int) -> dict[str, list[float]]:
    """Per scenario, the full-data pass^k as a one-element run list (reuses _scenario_mean)."""
    return {s: [_pass_k(view["tables"]["correctness"][s], k)] for s in scen}


def _latencies(side: list[tuple[Any, list[Any]]]) -> list[float]:
    return [m["latency_ms"] for r, _ in side for m in r.measures.values() if m.get("latency_ms") is not None]


def _per_solved(view: dict[str, Any], side: list[tuple[Any, list[Any]]]) -> float | None:
    values = [v for v in view["cost"]["values"].values() if v is not None]
    clean = sum(sum(r.clean_by_scenario.values()) for r, _ in side)
    return sum(values) / clean if values and clean else None
