"""Validate a trace export before evaluating it.

Catches the input problems that otherwise surface as a confusing verdict or a parser
crash: unparseable files, unknown formats, traces with no root (or several), duplicate
span ids, spans pointing at parents that are not in the export, missing or
backwards timestamps, unnamed tool calls, and runs with no prompt, answer or content.
It also reports what the checks will actually read: the tool steps found, tool spans
recognised only by their name, spans that carry tool attributes but are not read as
steps, and exports that mix native tool spans with OpenInference LLM spans.
Errors make the file unusable (exit 2); warnings are worth reading but do not block.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from faultgate.checks import Policy, _json_result, _NOT_JSON
from faultgate.formats import detect, load
from faultgate.trace import Trace, is_tool_span

DEFAULT_SAMPLE = 5


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    trace_id: str
    message: str


TOOL_ATTRS = ("gen_ai.tool.name", "gen_ai.tool.call.arguments", "gen_ai.tool.call.result", "tool.name")


def _trace_issues(t: Trace, fmt: str = "otlp") -> list[Issue]:
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
    steps = t.steps
    for st in steps:
        if not st.tool or st.tool == "unknown":
            err(f"step {st.index}: tool call without a name")
    by_name = [s for s in t.spans if is_tool_span(s) and "gen_ai.operation.name" not in s.attrs]
    if by_name:
        warn(f"{len(by_name)} tool span(s) recognised by span name only; set gen_ai.operation.name = execute_tool")
    read = {st.span_id for st in steps}
    unread = [s for s in t.spans if s.span_id not in read and s.parent_id is not None and any(k in s.attrs for k in TOOL_ATTRS)]
    if unread:
        warn(f"{len(unread)} span(s) carry tool attributes but are not read as tool steps ({', '.join(sorted({s.name for s in unread})[:3])}); "
             "set gen_ai.operation.name = execute_tool")
    if fmt == "openinference" and any(is_tool_span(s) and "openinference.span.kind" not in s.attrs for s in t.spans):
        warn("mixed export: native execute_tool spans next to OpenInference spans; steps are read from the tool spans, "
             "not from LLM messages (--format otlp reads the same steps)")
    if not t.has_content:
        warn("no content attributes (prompt, answer, tool arguments or results): rules see structure only")
    if not t.prompt.strip():
        warn("no prompt: scenario keys fall back to a hash of the empty string unless faultgate.scenario is set")
    if not t.answer.strip() and not t.termination:
        warn("no final answer and no faultgate.termination")
    return out


def describe_policy(path: str | Path) -> tuple[str, bool]:
    """The keys a policy file uses (requires with its sub-keys), and whether a requirement reads JSON tool results.
    Raises ValueError on a malformed policy, as check does."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    pol = Policy.from_dict(d)
    keys = []
    for k in sorted(d):
        if k == "requires" and d[k]:
            sub = sorted({s for r in d[k] for s in r})
            keys.append(f"requires ({', '.join(sub)})")
        else:
            keys.append(k)
    return ", ".join(keys) or "(none)", any(r.result or r.latest for r in pol.requires)


def precheck(path: str | Path, fmt: str = "auto", sample: int = DEFAULT_SAMPLE,
             needs_json: bool = False) -> tuple[str, int, int, int, list[Issue]]:
    """Return (format, traces in file, traces checked, tool steps read in those, issues). ``sample`` <= 0 checks every trace.
    ``needs_json``: the policy reads tool results as JSON (``result`` / ``latest``); warn when none of them are."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return "unknown", 0, 0, 0, [Issue("error", "-", f"cannot parse {p}: {e}")]
    try:
        found = detect(doc) if fmt == "auto" else fmt
        traces = load(p, fmt)
    except (ValueError, KeyError, TypeError) as e:
        return fmt, 0, 0, 0, [Issue("error", "-", f"cannot load as {fmt}: {e}")]
    if not traces:
        return found, 0, 0, 0, [Issue("error", "-", "no traces in file")]
    checked = traces if sample <= 0 else traces[:sample]
    issues = [i for t in checked for i in _trace_issues(t, found)]
    if needs_json and not any(_json_result(st.result) is not _NOT_JSON for t in checked for st in t.steps):
        issues.append(Issue("warning", "-", "the policy's requires use result / latest, but no tool result in the checked "
                                            "traces is JSON: those prerequisites can never be met"))
    return found, len(traces), len(checked), sum(len(t.steps) for t in checked), issues
