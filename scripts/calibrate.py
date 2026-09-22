"""Calibrate faultgate's checks against FAULTLINE's human failure-mode labels. $0.

Reads FAULTLINE P03's real agent runs (200 × openai/gpt-5.6-luna) and P04's human
coding sheet, converts each run to the same OTLP shape faultgate consumes, runs a
judge over the built-in checks, and reports confusion / TPR / TNR / Cohen's κ per
check — on all 200 and on P05's frozen test split (n=60) for comparability.

    .venv/bin/python scripts/calibrate.py --judge laya [--faultline /path] [--out calibration.json]

Requires a FAULTLINE checkout (this is a maintainer tool, not shipped).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faultgate.checks import CHECKS  # noqa: E402
from faultgate.judge import get_judge  # noqa: E402
from faultgate.runner import run  # noqa: E402
from faultgate.trace import Span, Trace  # noqa: E402

LABEL = {  # faultgate check -> P4 axial failure mode
    "search_loop": "OVERCONSTRAINED_SEARCH_LOOP",
    "abstention": "RETRIEVAL_FAILURE_ABSTENTION",
    "wrong_direction": "MULTI_HOP_DIRECTION_ERROR",
}


def to_trace(row: dict, sheet: dict) -> Trace:
    sid = row["task_id"]
    o = row["output"]
    H = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
    root = Span(span_id=H("r" + sid)[:16], parent_id=None, name="invoke_agent", start_ns=0, attrs={
        "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "faultline-agent", "gen_ai.agent.id": sid,
        "gen_ai.input.messages": json.dumps([{"role": "user", "parts": [{"type": "text", "content": sheet[sid]["prompt"]}]}]),
        "gen_ai.output.messages": json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": o.get("answer") or ""}]}]),
        "faultgate.termination": o.get("reason") or o.get("status") or "",
    })
    spans = [root]
    for st in o.get("trace") or []:
        tc = st.get("tool_call") or {}
        if not tc:
            continue
        ob = st.get("observation") or {}
        cands = ob.get("candidates") or []
        result = {"ok": ob.get("ok"), "error": ob.get("error"), "n_candidates": len(cands),
                  "candidates": [c if isinstance(c, str) else (c.get("doc_id") or str(c)[:60]) for c in cands][:5]}
        spans.append(Span(span_id=H(f"{sid}:{st['index']}")[:16], parent_id=root.span_id, name="execute_tool", start_ns=st["index"], attrs={
            "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tc.get("tool", "tool"),
            "gen_ai.tool.call.arguments": json.dumps({k: v for k, v in tc.items() if k != "tool"}),
            "gen_ai.tool.call.result": json.dumps(result),
        }))
    return Trace(trace_id=H("t" + sid)[:32], agent="faultline-agent", spans=spans)


def kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def score(report, sheet, ids) -> dict:
    out = {}
    for check, mode in LABEL.items():
        if check not in report.checks:
            continue
        tp = fp = fn = tn = 0
        for r in report.results:
            if r.check != check or r.scenario not in ids:
                continue
            truth = sheet[r.scenario]["axial_failure_mode"] == mode
            pred = r.verdict.detected
            tp += truth and pred; fp += (not truth) and pred; fn += truth and (not pred); tn += (not truth) and (not pred)
        out[check] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "positives": tp + fn,
                      "tpr": round(tp / (tp + fn), 3) if tp + fn else None,
                      "tnr": round(tn / (tn + fp), 3) if tn + fp else None,
                      "kappa": kappa(tp, fp, fn, tn)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=None, help="rules only when omitted")
    ap.add_argument("--faultline", default="/Volumes/SamirDrive/Development/FAULTLINE")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N scenarios only (smoke)")
    a = ap.parse_args()
    fl = Path(a.faultline)
    sheet = {r["scenario_id"]: r for r in csv.DictReader(open(fl / "projects/p04_taxonomy/coding_sheet.csv", encoding="utf-8"))}
    test_ids = set(json.load(open(fl / "projects/p05_judge/manifest.json"))["test_ids"])
    rows = [r for r in json.load(open(fl / "projects/p03_grounding/sweep_output.json"))["results"] if r["task_id"] in sheet]
    if a.limit:
        rows = rows[: a.limit]
    traces = [to_trace(r, sheet) for r in rows]
    judge = get_judge(a.judge) if a.judge else None
    t0 = time.perf_counter()
    report = run(traces, list(CHECKS.values()), judge)
    elapsed = time.perf_counter() - t0
    res = {"judge": report.judge, "n": len(traces), "seconds": round(elapsed, 1),
           "truncated": f"{getattr(judge, 'truncated', 0)}/{getattr(judge, 'calls', 0)}",
           "all": score(report, sheet, {t.key for t in traces}),
           "p5_test_split": score(report, sheet, test_ids & {t.key for t in traces})}
    for split in ("all", "p5_test_split"):
        print(f"\n== {split}  (judge {res['judge']}, {res['n']} traces, {res['seconds']}s, truncated {res['truncated']})")
        print(f"{'check':16}{'pos':>4}{'tp':>4}{'fp':>4}{'fn':>4}{'tn':>4}{'TPR':>7}{'TNR':>7}{'kappa':>8}")
        for c, m in res[split].items():
            print(f"{c:16}{m['positives']:>4}{m['tp']:>4}{m['fp']:>4}{m['fn']:>4}{m['tn']:>4}{str(m['tpr']):>7}{str(m['tnr']):>7}{str(m['kappa']):>8}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
