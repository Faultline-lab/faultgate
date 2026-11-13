"""Calibrate compare: false-BLOCK rate and injected-change detection (spec §14 criteria 2 and 3). $0.

Same-distribution pairs must BLOCK in at most 5% of seeds; each injected change (+30% cost,
+30% latency, +30% calls, correctness 0.9 -> 0.7 / 0.6) must be WORSE on its dimension and no
other (a correctness drop may also lower consistency) in at least 80% of seeds.

    .venv\\Scripts\\python scripts/calibrate_compare.py --seeds 200 [--resamples 2000] [--scenarios 30] [--runs 3]
    .venv\\Scripts\\python scripts/calibrate_compare.py --quick

Synthetic runs come from tests/fixtures/make_compare_runs.py (maintainer tool, not shipped).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from faultgate import formats, runner  # noqa: E402
from faultgate.checks import Policy, select  # noqa: E402
from faultgate.compare import DIMENSIONS, compare  # noqa: E402
from tests.fixtures.make_compare_runs import POLICY, write_side  # noqa: E402

CHECKS = select(None, Policy.from_dict(POLICY))

# name, make_run kwargs for the candidate, dimension that must move
CHANGES = [
    ("cost +30%", {"cost_usd": 0.04 * 1.3}, "cost"),
    ("latency +30%", {"latency_ms": 5000.0 * 1.3}, "speed"),
    ("calls +30%", {"calls": 6.0 * 1.3}, "approach"),
    ("correctness 0.9->0.7", {"p_clean": 0.7}, "correctness"),
    ("correctness 0.9->0.6", {"p_clean": 0.6}, "correctness"),
]


def tmp_base() -> Path:
    d = Path(os.environ.get("TEMP") or os.environ.get("TMP") or r"I:\AES\Data\tmp")
    if d.drive.upper() != "I:":
        raise SystemExit(f"refusing to write temp files outside I: (TEMP={d}); set TEMP=I:\\AES\\Data\\tmp")
    return d


def side(td: Path, prefix: str, world_seed: int, runs: int, run_seed: int, scenarios: int, **kw) -> list:
    out = []
    for p in write_side(td, prefix, runs=runs, world_seed=world_seed, run_seed=run_seed, n=scenarios, **kw):
        traces = formats.load(p)
        out.append((runner.run(traces, CHECKS), traces))
    return out


def detected(labels: dict, target: str) -> bool:
    worse = {d for d, label in labels.items() if label == "WORSE"}
    allowed = {target} | ({"consistency"} if target == "correctness" else set())
    return target in worse and worse <= allowed


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate compare's false-BLOCK and detection rates.")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--scenarios", type=int, default=30)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="seeds 20 (smoke run)")
    a = ap.parse_args()
    if a.quick:
        a.seeds = 20
    if a.runs < 2:
        ap.error("--runs must be at least 2 (compare needs two runs a side)")

    false_block = 0
    hits = dict.fromkeys((c[0] for c in CHANGES), 0)
    target_worse = dict.fromkeys(hits, 0)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=tmp_base()) as td_name:
        td = Path(td_name)
        for seed in range(a.seeds):
            # ci 0 = same distribution; ci 1..5 = injected changes, each with a fresh baseline
            for ci, (name, kw, dim) in enumerate([("same distribution", {}, None), *CHANGES]):
                run_seed = seed * 100 + ci * 10
                base = side(td, "base", seed, a.runs, run_seed, a.scenarios)
                cand = side(td, "cand", seed, a.runs, run_seed + 5, a.scenarios, **kw)
                res = compare(base, cand, resamples=a.resamples, seed=seed)
                if dim is None:
                    false_block += res["verdict"] == "BLOCK"
                else:
                    labels = {d: res["dimensions"][d]["label"] for d in DIMENSIONS}
                    hits[name] += detected(labels, dim)
                    target_worse[name] += labels[dim] == "WORSE"

    print(f"{a.seeds} seeds, {a.scenarios} scenarios x {a.runs} runs a side, {a.resamples} resamples")
    print(f"\nsame distribution: BLOCK {100 * false_block / a.seeds:5.1f}%   target <= 5%")
    print(f"\n{'change':22}{'dimension':12}{'detected':>10}{'target worse':>14}")
    for name, _, dim in CHANGES:
        print(f"{name:22}{dim:12}{100 * hits[name] / a.seeds:9.1f}%{100 * target_worse[name] / a.seeds:13.1f}%")
    print("detection target >= 80%")
    print(f"\nelapsed {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
