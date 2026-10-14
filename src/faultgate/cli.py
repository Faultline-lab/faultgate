"""faultgate CLI. Exit 0 = clean, 1 = a check fired, 2 = usage or input error."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

from faultgate import __version__
from faultgate import band as bandmod
from faultgate.checks import CHECKS, OUTCOME, Golden, Outcomes, Policy, select
from faultgate.judge import JUDGE_SPECS, get_judge
from faultgate.runner import Report
from faultgate.runner import run as run_checks
from faultgate.formats import FORMATS
from faultgate.formats import load as load_traces
from faultgate.precheck import DEFAULT_SAMPLE, describe_policy, precheck
from faultgate.redact import REDACTIONS, get_redactor
from faultgate.report import _outcome_line, display_path, markdown, write_audit


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


def _load(path: str, fmt: str = "auto"):
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"no such file: {p}")
    traces = load_traces(p, fmt)
    if not traces:
        raise ValueError(f"no traces found in {p}")
    no_content = [t for t in traces if not t.has_content]
    if no_content:
        print(f"warning: {len(no_content)} trace(s) in {p} carry no content attributes; the judge sees structure only", file=sys.stderr)
    return traces


def _run(path: str, checks, judge, fmt: str = "auto", redact=None) -> tuple[Report, float]:
    t0 = time.perf_counter()
    report = run_checks(_load(path, fmt), checks, judge, redact)
    return report, time.perf_counter() - t0


def _judge_or_none(args, checks):
    judge = get_judge(args.judge) if args.judge else None
    no_judge = [c.name for c in checks if c.kind == "judge"] if judge is None else []
    no_golden = [c.name for c in checks if c.kind == "golden" and c.rule is None]
    no_outcome = [c.name for c in checks if c.kind == "outcome" and c.rule is None]
    if len(no_judge) + len(no_golden) + len(no_outcome) == len(checks):
        raise ValueError("nothing to run: " + "; ".join(
            filter(None, [no_judge and f"{', '.join(no_judge)} need --judge laya | litellm:<model>",
                          no_golden and f"{', '.join(no_golden)} need --golden golden.json",
                          no_outcome and "outcome needs --outcomes outcomes.json"])))
    for name in no_outcome:
        print(f"note: {name} skipped — it needs --outcomes outcomes.json (true / false per scenario)", file=sys.stderr)
    for name in no_judge:
        print(f"note: {name} skipped — it needs a judge (--judge laya | litellm:<model>); see README for each judge's measured κ", file=sys.stderr)
    for name in no_golden:
        print(f"note: {name} skipped — it needs --golden golden.json (expected answers)", file=sys.stderr)
    return judge


def _golden(args) -> Golden | None:
    return Golden.load(args.golden) if args.golden else None


def _policy(args) -> Policy | None:
    return Policy.load(args.policy) if args.policy else None


def _outcomes(args) -> Outcomes | None:
    return Outcomes.load(args.outcomes) if getattr(args, "outcomes", None) else None


def cmd_check(args: argparse.Namespace) -> int:
    try:
        checks = select(args.check, _policy(args), _golden(args), _outcomes(args))
        judge = _judge_or_none(args, checks)
        band = json.loads(Path(args.band).read_text(encoding="utf-8")) if args.band else None
        report, elapsed = _run(args.traces, checks, judge, args.format, get_redactor(args.redact))
    except (ValueError, KeyError, OSError, ImportError) as e:
        print(f"faultgate: {e}", file=sys.stderr)
        return 2

    print(_table(report))
    fired = ", ".join(f"{k}={v}" for k, v in report.fired.items())
    print(f"\n{len(report.failed_traces)}/{len(report.trace_ids)} traces failed · {fired} · judge {report.judge} · {elapsed:.1f}s")
    if report.outcome is not None:
        print(_outcome_line(report.outcome))
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
        s = g["stats"]
        print(f"gate: {g['verdict']} · clean rate {g['clean_rate']:.2f} vs band [{lo:.2f}, {hi:.2f}] from {band['runs']} baseline runs"
              f" · fail below {g['fail_threshold']:.2f} · p={s['p_value']:.3f}")
        if g["verdict"] == "INCONCLUSIVE":
            print(f"inconclusive: a drop of {s['drop']:.2f} needs {s['required_n']} scenarios to tell from noise "
                  f"(alpha {s['alpha']}, power {s['power']:.2f}) · run {s['additional_n']} more")
        if g["policy_violations"]:
            print(f"gate: FAIL · policy violation in {len(g['policy_violations'])} trace(s) (hard block)")
        if g["regressions"]:
            print(f"regressions (clean in every baseline, not now): {', '.join(g['regressions'])}")
        code = 1 if g["verdict"] == "FAIL" else 0
    command = _repro(getattr(args, "argv", None) or ["check", args.traces])
    try:
        if args.json:
            _write(args.json, json.dumps(out, indent=1))
            print(f"report written to {args.json}")
        if args.markdown:
            _write(args.markdown, markdown(report, out.get("gate"), band, command))
            print(f"markdown written to {args.markdown}")
        if args.audit:
            inputs = {"traces": args.traces, "band": args.band, "policy": args.policy, "golden": args.golden,
                  "outcomes": args.outcomes}
            write_audit(args.audit, report, out.get("gate"), inputs, command, code)
            print(f"audit written to {args.audit}")
    except OSError as e:
        # an output that could not be written is a usage error, not a gate verdict: exit 1 would read as FAIL
        print(f"faultgate: cannot write output: {e}", file=sys.stderr)
        return 2
    return code


def _write(path: str, text: str) -> None:
    """Write an output file, creating its parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


OUTPUT_FLAGS = {"--json", "--markdown", "--audit"}


def _repro(argv: list[str]) -> str:
    """The command that reproduces this verdict: the same inputs and options, without the output files."""
    keep, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in OUTPUT_FLAGS:
            skip = True
            continue
        if a.split("=", 1)[0] in OUTPUT_FLAGS:
            continue
        keep.append(display_path(a) if not a.startswith("-") and Path(a).is_file() else a)
    return shlex.join(["faultgate", *keep])


def cmd_baseline(args: argparse.Namespace) -> int:
    try:
        checks = select(args.check, _policy(args), _golden(args), _outcomes(args))
        judge = _judge_or_none(args, checks)
        reports = []
        redactor = get_redactor(args.redact)
        for path in args.traces:
            report, elapsed = _run(path, checks, judge, args.format, redactor)
            rate = sum(report.clean_by_scenario.values()) / len(report.trace_ids)
            print(f"{path}: {len(report.trace_ids)} traces · clean rate {rate:.2f} · {elapsed:.1f}s")
            reports.append(report)
        band = bandmod.build(reports)
        if len(reports) < bandmod.RECOMMENDED_RUNS:
            print(f"note: {len(reports)} baseline runs measure noise poorly; use {bandmod.RECOMMENDED_RUNS} or more", file=sys.stderr)
    except (ValueError, KeyError, OSError) as e:
        print(f"faultgate: {e}", file=sys.stderr)
        return 2
    try:
        _write(args.out, json.dumps(band, indent=1))
    except OSError as e:
        print(f"faultgate: cannot write output: {e}", file=sys.stderr)
        return 2
    c = band["clean_rate"]
    print(f"band [{c['min']:.2f}, {c['max']:.2f}] over {band['n']} scenarios × {band['runs']} runs · "
          f"{len(band['clean_in_all_runs'])} clean in every run · written to {args.out}")
    return 0


def cmd_precheck(args: argparse.Namespace) -> int:
    needs_json = False
    if args.policy:
        try:
            keys, needs_json = describe_policy(args.policy)
        except (ValueError, OSError) as e:
            print(f"faultgate: {e}", file=sys.stderr)
            return 2
        print(f"policy keys: {keys}")
    fmt, total, checked, steps, issues = precheck(args.traces, args.format, args.sample, needs_json)
    for i in issues:
        print(f"{i.level:<8} {i.trace_id:<12}  {i.message}")
    errors = sum(i.level == "error" for i in issues)
    warnings = len(issues) - errors
    verdict = "FAILED" if errors else "OK"
    print(f"precheck: {verdict} · format {fmt} · checked {checked} of {total} traces · {steps} tool steps read · "
          f"{errors} error(s), {warnings} warning(s)")
    return 2 if errors else 0


def cmd_checks(_: argparse.Namespace) -> int:
    for c in [*CHECKS.values(), OUTCOME]:
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
    c.add_argument("--markdown", default=None, help="write a Markdown summary (PR comment / job summary) to this path")
    c.add_argument("--audit", default=None, help="write verdict.json, failing_traces.json and repro.sh to this directory")
    c.add_argument("--band", default=None, help="band.json from `faultgate baseline`; exit 1 only on FAIL, not on noise")
    c.add_argument("--policy", default=None, help='policy.json: {"allow": [...], "max_calls": N, "max_arg_chars": N, "deny_patterns": [...], '
                                                  '"limits": [...], "identity": {...}, "requires": [...]}; a violation fails the gate even with --band')
    c.add_argument("--golden", default=None, help="golden.json: {scenario key or prompt: expected answer}; enables wrong_answer and truncation")
    c.add_argument("--outcomes", default=None, help="outcomes.json: {scenario key or prompt: true | false} from your own evaluator; "
                                                    "enables the outcome check")
    c.add_argument("--format", default="auto", choices=FORMATS, help="trace format (default: detect)")
    c.add_argument("--redact", default="builtin", choices=REDACTIONS,
                   help="mask PII and secrets in judge input (API judges) and in every reason written out (default builtin)")
    c.set_defaults(fn=cmd_check)
    b = sub.add_parser("baseline", help="build a tolerance band from >= 2 baseline runs of the same scenarios")
    b.add_argument("traces", nargs="+", help="OTLP JSON files, one per baseline run")
    b.add_argument("--judge", default=None)
    b.add_argument("--check", default=None)
    b.add_argument("--policy", default=None)
    b.add_argument("--golden", default=None)
    b.add_argument("--outcomes", default=None, help="outcomes.json: {scenario key or prompt: true | false}")
    b.add_argument("--out", default="band.json")
    b.add_argument("--format", default="auto", choices=FORMATS, help="trace format (default: detect)")
    b.add_argument("--redact", default="builtin", choices=REDACTIONS,
                   help="mask PII and secrets in judge input (API judges) and in every reason written out (default builtin)")
    b.set_defaults(fn=cmd_baseline)
    pc = sub.add_parser("precheck", help="validate a trace export (format, roots, span ids, timestamps, content) before checking it")
    pc.add_argument("traces", help="trace export: OTLP/JSON, OpenInference, Langfuse, LangSmith or Braintrust")
    pc.add_argument("--format", default="auto", choices=FORMATS)
    pc.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help=f"traces to inspect (default {DEFAULT_SAMPLE}; 0 = all)")
    pc.add_argument("--policy", default=None, help="policy.json: list the keys it uses and check the traces carry what they read")
    pc.set_defaults(fn=cmd_precheck)
    sub.add_parser("checks", help="list built-in checks").set_defaults(fn=cmd_checks)
    sub.add_parser("judges", help="list judge specs").set_defaults(fn=cmd_judges)
    return p


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)
    args.argv = raw
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
