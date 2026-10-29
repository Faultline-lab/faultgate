#!/usr/bin/env python3
"""Join a tau2 results file with a faultgate JSON report and list disagreements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("report", type=Path)
    p.add_argument("--json", type=Path, help="also write the joined rows here")
    args = p.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    fired: dict[str, list[dict]] = {}
    for item in report.get("results", []):
        if item.get("detected"):
            fired.setdefault(item["scenario"], []).append(item)

    rows = []
    for task in results["tasks"]:
        scenario = f"retail-{task['task_id']}"
        checks = fired.get(scenario, [])
        tau2_pass = task.get("reward") == 1.0
        faultgate_pass = not checks
        rows.append({
            "task_id": task["task_id"],
            "tau2_reward": task.get("reward"),
            "tau2_pass": tau2_pass,
            "faultgate_pass": faultgate_pass,
            "fired": [{"check": c["check"], "reason": c["reason"]} for c in checks],
        })

    print(f"{'task':>4}  {'tau2':>5}  {'faultgate':>9}  checks")
    for row in rows:
        checks = ", ".join(c["check"] for c in row["fired"]) or "-"
        print(f"{row['task_id']:>4}  {str(row['tau2_reward']):>5}  "
              f"{'pass' if row['faultgate_pass'] else 'FAIL':>9}  {checks}")

    false_positives = [r for r in rows if r["tau2_pass"] and not r["faultgate_pass"]]
    misses = [r for r in rows if not r["tau2_pass"] and r["faultgate_pass"]]
    print(f"\nagreement: {len(rows) - len(false_positives) - len(misses)}/{len(rows)}")
    print(f"tau2 pass / faultgate fail (false positive or process-only): {len(false_positives)}")
    for row in false_positives:
        print(f"  task {row['task_id']}: " + "; ".join(
            f"{c['check']}: {c['reason']}" for c in row["fired"]))
    print(f"tau2 fail / faultgate pass (outside its checks): {len(misses)}")
    for row in misses:
        print(f"  task {row['task_id']}: reward={row['tau2_reward']}")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
