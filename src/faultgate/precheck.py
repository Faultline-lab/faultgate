"""Validate a trace export before evaluating it.

Catches the input problems that otherwise surface as a confusing verdict or a parser
crash: unparseable files, unknown formats, traces with no root (or several), duplicate
span ids, spans pointing at parents that are not in the export, missing or
backwards timestamps, unnamed tool calls, and runs with no prompt, answer or content.
Errors make the file unusable (exit 2); warnings are worth reading but do not block.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from faultgate.formats import detect, load
from faultgate.trace import Trace

DEFAULT_SAMPLE = 5


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    trace_id: str
    message: str


def _trace_issues(t: Trace) -> list[Issue]:
    out: list[Issue] = []
    tid = t.trace_id[:12]

    def err(m: str) -> None:
        out.append(Issue("error", tid, m))

    def warn(m: str) -> None:
        out.append(Issue("warning", tid, m))

    ids = [s.span_id for s in t.spans]
    for sid in sorted({i for i in ids if ids.count(i) > 1}):
        err(f"duplicate span id {sid}")
    roots = [s for s in t.spans if s.parent_id is None]
    known = set(ids)
    dangling = [s for s in t.spans if s.parent_id is not None and s.parent_id not in known]
    if not roots:
        warn("no root span in the export: prompt and answer are read from child spans")
    elif len(roots) > 1:
        warn(f"{len(roots)} root spans; the first is used as the run")
    if dangling:
        parents = sorted({s.parent_id for s in dangling})
        warn(f"{len(dangling)} span(s) point at {len(parents)} parent(s) not in the export ({', '.join(parents[:3])})")
    for s in t.spans:
        if s.start_ns <= 0:
            err(f"span {s.span_id} ({s.name}): missing start time")
        elif s.end_ns and s.end_ns < s.start_ns:
            err(f"span {s.span_id} ({s.name}): ends before it starts")
    if roots and roots[0].start_ns > 0:
        early = [s for s in t.spans if 0 < s.start_ns < roots[0].start_ns]
        if early:
            warn(f"{len(early)} span(s) start before the root span")
    for st in t.steps:
        if not st.tool or st.tool == "unknown":
            err(f"step {st.index}: tool call without a name")
    if not t.has_content:
        warn("no content attributes (prompt, answer, tool arguments or results): rules see structure only")
    if not t.prompt.strip():
        warn("no prompt: scenario keys fall back to a hash of the empty string unless faultgate.scenario is set")
    if not t.answer.strip() and not t.termination:
        warn("no final answer and no faultgate.termination")
    return out


def precheck(path: str | Path, fmt: str = "auto", sample: int = DEFAULT_SAMPLE) -> tuple[str, int, int, list[Issue]]:
    """Return (format, traces in file, traces checked, issues). ``sample`` <= 0 checks every trace."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return "unknown", 0, 0, [Issue("error", "-", f"cannot parse {p}: {e}")]
    try:
        found = detect(doc) if fmt == "auto" else fmt
        traces = load(p, fmt)
    except (ValueError, KeyError, TypeError) as e:
        return fmt, 0, 0, [Issue("error", "-", f"cannot load as {fmt}: {e}")]
    if not traces:
        return found, 0, 0, [Issue("error", "-", "no traces in file")]
    checked = traces if sample <= 0 else traces[:sample]
    return found, len(traces), len(checked), [i for t in checked for i in _trace_issues(t)]
