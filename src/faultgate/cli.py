"""faultgate CLI. Exit 0 = clean, 1 = a check fired, 2 = usage or input error."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from faultgate import __version__
from faultgate import band as bandmod
from faultgate.checks import CHECKS, Golden, Policy, select
from faultgate.judge import JUDGE_SPECS, get_judge
from faultgate.runner import Report
from faultgate.runner import run as run_checks
from faultgate.trace import load_otlp


def _table(report: Report) -> str:
    width = max(len("trace"), *(len(t[:12]) for t in report.trace_ids))
    head = f"{'trace':<{width}}  " + "  ".join(f"{c:<16}" for c in report.checks)
    rows = [head, "-" * len(head)]
    for tid in report.trace_ids:
        cells = []
        for v in report.for_trace(tid):
            cells.append(f"{('FAIL' if v.detected else 'ok'):<5}{v.confidence:.2f}".ljust(16))
        rows.append(f"{tid[:12]:<{width}}  " + "  ".join(cells))
    return "\n".join(rows)


def _load(path: str):
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"no such file: {p}")
    traces = load_otlp(p)
    if not traces:
        raise ValueError(f"no traces found in {p}")
    no_content = [t for t in traces if not t.has_content]
    if no_content:
        print(f"warning: {len(no_content)} trace(s) in {p} carry no content attributes; the judge sees structure only", file=sys.stderr)
    return traces


def _run(path: str, checks, judge) -> tuple[Report, float]:
    t0 = time.perf_counter()
    report = run_checks(_load(path), checks, judge)
    return report, time.perf_counter() - t0


def _judge_or_none(args, checks):
    judge = get_judge(args.judge) if args.judge else None
    no_judge = [c.name for c in checks if c.kind == "judge"] if judge is None else []
    no_golden = [c.name for c in checks if c.kind == "golden" and c.rule is None]
    if len(no_judge) + len(no_golden) == len(checks):
        raise ValueError("nothing to run: " + "; ".join(
            filter(None, [no_judge and f"{', '.join(no_judge)} need --judge laya | litellm:<model>",
                          no_golden and f"{', '.join(no_golden)} need --golden golden.json"])))
    for name in no_judge:
        print(f"note: {name} skipped — it needs a judge (--judge laya | litellm:<model>); see README for each judge's measured κ", file=sys.stderr)
    for name in no_golden:
        print(f"note: {name} skipped — it needs --golden golden.json (expected answers)", file=sys.stderr)
    return judge


def _golden(args) -> Golden | None:
    return Golden.load(args.golden) if args.golden else None


def _policy(args) -> Policy | None:
    return Policy.load(args.policy) if args.policy else None


def cmd_check(args: argparse.Namespace) -> int:
    try:
        checks = select(args.check, _policy(args), _golden(args))
        judge = _judge_or_none(args, checks)
        band = json.loads(Path(args.band).read_text(encoding="utf-8")) if args.band else None
        report, elapsed = _run(args.traces, checks, judge)
    except (ValueError, KeyError, OSError, ImportError) as e:
        print(f"faultgate: {e}", file=sys.stderr)
        return 2

    print(_table(report))
    fired = ", ".join(f"{k}={v}" for k, v in report.fired.items())
    print(f"\n{len(report.failed_traces)}/{len(report.trace_ids)} traces failed · {fired} · judge {report.judge} · {elapsed:.1f}s")
    if judge and getattr(judge, "truncated", None):
        print(f"warning: {judge.truncated}/{judge.calls} judge calls exceeded Laya's 512-token context; those verdicts saw a truncated trace", file=sys.stderr)

    out = report.to_dict()
    code = 0 if report.passed else 1
    if band:
        g = bandmod.evaluate(report, band)
        out["gate"] = g
        for w in g["warnings"]:
            print(f"warning: {w}", file=sys.stderr)
        lo, hi = g["band"]
        print(f"gate: {g['verdict']} · clean rate {g['clean_rate']:.2f} vs band [{lo:.2f}, {hi:.2f}] from {band['runs']} baseline runs"
              f" · fail below {g['fail_threshold']:.2f}")
        if g["regressions"]:
            print(f"regressions (clean in every baseline, not now): {', '.join(g['regressions'])}")
        code = 1 if g["verdict"] == "FAIL" else 0
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"report written to {args.json}")
    return code


def cmd_baseline(args: argparse.Namespace) -> int:
    try:
        checks = select(args.check, _policy(args), _golden(args))
        judge = _judge_or_none(args, checks)
        reports = []
        for path in args.traces:
            report, elapsed = _run(path, checks, judge)
            rate = sum(report.clean_by_scenario.values()) / len(report.trace_ids)
            print(f"{path}: {len(report.trace_ids)} traces · clean rate {rate:.2f} · {elapsed:.1f}s")
            reports.append(report)
        band = bandmod.build(reports)
    except (ValueError, KeyError, OSError) as e:
        print(f"faultgate: {e}", file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(band, indent=1), encoding="utf-8")
    c = band["clean_rate"]
    print(f"band [{c['min']:.2f}, {c['max']:.2f}] over {band['n']} scenarios × {band['runs']} runs · "
          f"{len(band['clean_in_all_runs'])} clean in every run · written to {args.out}")
    return 0


def cmd_checks(_: argparse.Namespace) -> int:
    for c in CHECKS.values():
        print(f"{c.name:<16} {c.kind:<6} {c.description}")
    return 0


def cmd_judges(_: argparse.Namespace) -> int:
    for k, v in JUDGE_SPECS.items():
        print(f"{k:<16} {v}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="faultgate", description="Local-first release gate for AI agents.")
    p.add_argument("--version", action="version", version=f"faultgate {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="run checks over an OTLP/JSON trace export")
    c.add_argument("traces", help="OTLP JSON file (GenAI semantic conventions)")
    c.add_argument("--judge", default=None, help="enable judge checks: laya or litellm:<model> (default: rules only)")
    c.add_argument("--check", default=None, help="comma-separated subset of checks (default: all)")
    c.add_argument("--json", default=None, help="also write the full report to this path")
    c.add_argument("--band", default=None, help="band.json from `faultgate baseline`; exit 1 only on FAIL, not on noise")
    c.add_argument("--policy", default=None, help='policy.json: {"allow": [...], "max_calls": N, "max_arg_chars": N, "deny_patterns": [...]}')
    c.add_argument("--golden", default=None, help="golden.json: {scenario key or prompt: expected answer}; enables wrong_answer and truncation")
    c.set_defaults(fn=cmd_check)
    b = sub.add_parser("baseline", help="build a tolerance band from >= 2 baseline runs of the same scenarios")
    b.add_argument("traces", nargs="+", help="OTLP JSON files, one per baseline run")
    b.add_argument("--judge", default=None)
    b.add_argument("--check", default=None)
    b.add_argument("--policy", default=None)
    b.add_argument("--golden", default=None)
    b.add_argument("--out", default="band.json")
    b.set_defaults(fn=cmd_baseline)
    sub.add_parser("checks", help="list built-in checks").set_defaults(fn=cmd_checks)
    sub.add_parser("judges", help="list judge specs").set_defaults(fn=cmd_judges)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
