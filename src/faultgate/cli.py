"""faultgate CLI. Exit 0 = clean, 1 = a check fired, 2 = usage or input error."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from faultgate import __version__
from faultgate.checks import CHECKS, select
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


def cmd_check(args: argparse.Namespace) -> int:
    path = Path(args.traces)
    if not path.is_file():
        print(f"faultgate: no such file: {path}", file=sys.stderr)
        return 2
    try:
        traces = load_otlp(path)
        checks = select(args.check)
        judge = get_judge(args.judge)
    except (ValueError, KeyError) as e:
        print(f"faultgate: {e}", file=sys.stderr)
        return 2
    if not traces:
        print("faultgate: no traces found in file", file=sys.stderr)
        return 2
    no_content = [t.trace_id for t in traces if not t.has_content]
    if no_content:
        print(f"warning: {len(no_content)} trace(s) carry no content attributes; the judge sees structure only", file=sys.stderr)

    t0 = time.perf_counter()
    report = run_checks(traces, checks, judge)
    elapsed = time.perf_counter() - t0

    print(_table(report))
    fired = ", ".join(f"{k}={v}" for k, v in report.fired.items())
    print(f"\n{len(report.failed_traces)}/{len(report.trace_ids)} traces failed · {fired} · judge {report.judge} · {elapsed:.1f}s")
    if getattr(judge, "truncated", None):
        print(f"warning: {judge.truncated}/{judge.calls} judge calls exceeded Laya's 512-token context; those verdicts saw a truncated trace", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=1), encoding="utf-8")
        print(f"report written to {args.json}")
    return 0 if report.passed else 1


def cmd_checks(_: argparse.Namespace) -> int:
    for c in CHECKS.values():
        print(f"{c.name:<16} {c.description}")
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
    c.add_argument("--judge", default="laya", help="laya (default) or litellm:<model>")
    c.add_argument("--check", default=None, help="comma-separated subset of checks (default: all)")
    c.add_argument("--json", default=None, help="also write the full report to this path")
    c.set_defaults(fn=cmd_check)
    sub.add_parser("checks", help="list built-in checks").set_defaults(fn=cmd_checks)
    sub.add_parser("judges", help="list judge specs").set_defaults(fn=cmd_judges)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
