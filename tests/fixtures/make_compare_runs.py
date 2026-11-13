"""Seeded synthetic OTLP runs for compare: known per-scenario success, calls, latency, cost and tokens.

No fixture file to regenerate: import and call.

    from tests.fixtures.make_compare_runs import POLICY, make_run, write_side

``world_seed`` fixes what is shared by both sides (each scenario's success probability and its
multipliers for calls / latency / cost / tokens), so changing ``cost_usd`` alone scales every
scenario value by the same factor ("+30% cost" = ``cost_usd=0.04 * 1.3``, same world seed).
``run_seed`` adds run-to-run noise. A failed scenario run carries one structural marker (its last
tool result is an error that is never recovered), so exactly one rule fires and the tool-call count
is unchanged; the clean rate is the asked ``p_clean``.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

TOOL = "lookup_record"
VIOLATION_TOOL = "export_all"  # deliberately absent from POLICY
POLICY: dict[str, Any] = {"allow": [TOOL]}

BASE_NS = 1_760_000_000_000_000_000  # 2025-10-09, so timestamps are plausible
FAILED = "Error: lookup failed"


def _attr(k: str, v: Any) -> dict[str, Any]:
    if isinstance(v, int):
        return {"key": k, "value": {"intValue": str(v)}}
    if isinstance(v, float):
        return {"key": k, "value": {"doubleValue": v}}
    return {"key": k, "value": {"stringValue": str(v)}}


def _world(world_seed: int, n: int, p_clean: float) -> list[dict[str, float]]:
    """Scenario-level parameters: fixed count of draws per scenario, so they do not move with the request."""
    rng = random.Random(world_seed)
    out = []
    for _ in range(n):
        p = min(1.0, max(0.0, p_clean + rng.uniform(-0.1, 0.1)))
        mul = [math.exp(rng.uniform(math.log(0.5), math.log(2))) for _ in range(4)]  # lognormal-ish, [0.5, 2]
        out.append({"p": p, "calls": mul[0], "latency": mul[1], "cost": mul[2], "tokens": mul[3]})
    return out


def _noise(run_seed: int, n: int) -> list[dict[str, float]]:
    rng = random.Random(run_seed)
    out = []
    for _ in range(n):
        u = rng.random()
        out.append({"u": u, "calls": rng.uniform(0.85, 1.15), "latency": rng.uniform(0.85, 1.15),
                    "cost": rng.uniform(0.85, 1.15), "tokens": rng.uniform(0.85, 1.15)})
    return out


def make_run(world_seed: int, run_seed: int, n: int = 30, p_clean: float = 0.9, calls: float = 6.0,
             latency_ms: float = 5000.0, cost_usd: float | None = 0.04, tokens: tuple[int, int] = (2000, 500),
             model: str = "model-a", policy_violation: bool = False, zero_calls: bool = False) -> dict:
    """One OTLP JSON export (dict) of n scenarios, one trace each.

    ponytail: the [0.5, 2] scenario spread puts a 30-scenario side's absolute mean within ~25% of the knob;
    calibration compares two sides from the same world seed, where the ratio is exact. Widening the spread or
    raising n would tighten the absolute mean.
    """
    world = _world(world_seed, n, p_clean)
    noise = _noise(run_seed, n)
    spans: list[dict[str, Any]] = []
    for i, (w, k) in enumerate(zip(world, noise)):
        key = f"s{i:02d}"
        ok = k["u"] < w["p"]
        n_calls = 0 if zero_calls else max(1, round(calls * w["calls"] * k["calls"]))
        lat_ns = int(round(latency_ms * w["latency"] * k["latency"] * 1e6))
        tok = w["tokens"] * k["tokens"]
        tid = f"{world_seed:08x}{run_seed:08x}{i + 1:016x}"
        root_id = f"{i + 1:08x}00000000"
        t0 = BASE_NS + i * 10**11
        root_attrs = [
            _attr("gen_ai.operation.name", "invoke_agent"),
            _attr("gen_ai.agent.name", "compare"),
            _attr("gen_ai.request.model", model),
            _attr("faultgate.scenario", key),
            _attr("gen_ai.prompt", f"scenario {key}"),
            _attr("gen_ai.completion", "Done."),
            _attr("faultgate.termination", "stop"),
            _attr("gen_ai.usage.input_tokens", int(round(tokens[0] * tok))),
            _attr("gen_ai.usage.output_tokens", int(round(tokens[1] * tok))),
        ]
        if cost_usd is not None:
            root_attrs.append(_attr("faultgate.cost_usd", cost_usd * w["cost"] * k["cost"]))
        root: dict[str, Any] = {"traceId": tid, "spanId": root_id, "name": "invoke_agent",
                                "startTimeUnixNano": str(t0), "endTimeUnixNano": str(t0 + lat_ns),
                                "attributes": root_attrs}
        if not ok and n_calls == 0:  # no tool span to carry the failure marker
            root["status"] = {"code": "STATUS_CODE_ERROR", "message": FAILED}
        spans.append(root)
        for j in range(n_calls):
            tool = VIOLATION_TOOL if (policy_violation and i == 0 and j == 0) else TOOL
            failed_here = not ok and j == n_calls - 1
            start = t0 + lat_ns * (j + 1) // (n_calls + 1)
            spans.append({
                "traceId": tid, "spanId": f"{i + 1:08x}{j + 1:08x}", "parentSpanId": root_id,
                "name": f"execute_tool {tool}", "startTimeUnixNano": str(start), "endTimeUnixNano": str(start + 10**6),
                "attributes": [
                    _attr("gen_ai.operation.name", "execute_tool"),
                    _attr("gen_ai.tool.name", tool),
                    _attr("gen_ai.tool.call.arguments", json.dumps({"scenario": key, "step": j + 1})),
                    _attr("gen_ai.tool.call.result", FAILED if failed_here else json.dumps({"ok": True, "step": j + 1})),
                ],
            })
    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}


def write_side(out_dir: str | Path, prefix: str, runs: int = 3, world_seed: int = 0, run_seed: int = 0,
               **kw) -> list[Path]:
    """Write `runs` files <prefix>_run<i>.json (run seeds run_seed*1000 + i) and return their paths."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(runs):
        p = d / f"{prefix}_run{i}.json"
        p.write_text(json.dumps(make_run(world_seed, run_seed * 1000 + i, **kw), sort_keys=True) + "\n",
                     encoding="utf-8")
        paths.append(p)
    return paths
