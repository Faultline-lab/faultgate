"""Run checks over traces with a judge. Pure: no I/O."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

from faultgate.checks import Check
from faultgate.judge import Judge, Verdict
from faultgate.trace import Trace


@dataclass(frozen=True)
class Result:
    trace_id: str
    scenario: str
    check: str
    verdict: Verdict


@dataclass
class Report:
    judge: str
    checks: list[str]
    results: list[Result] = field(default_factory=list)
    redaction: str = "off"
    measures: dict[str, dict[str, float | None]] = field(default_factory=dict)  # trace id -> tokens, latency_ms, cost_usd

    @property
    def trace_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.results:
            seen.setdefault(r.trace_id)
        return list(seen)

    @property
    def scenario_of(self) -> dict[str, str]:
        return {r.trace_id: r.scenario for r in self.results}

    @property
    def clean_by_scenario(self) -> dict[str, bool]:
        """scenario key -> True if no check fired on that trace."""
        return {self.scenario_of[t]: t not in self.failed_traces for t in self.trace_ids}

    @property
    def failed_traces(self) -> list[str]:
        return [t for t in self.trace_ids if any(r.detected for r in self.for_trace(t))]

    def for_trace(self, trace_id: str) -> list[Verdict]:
        return [r.verdict for r in self.results if r.trace_id == trace_id]

    def fired_on(self, check: str) -> list[str]:
        """Trace ids where ``check`` fired, in report order."""
        hit = {r.trace_id for r in self.results if r.check == check and r.verdict.detected}
        return [t for t in self.trace_ids if t in hit]

    @property
    def fired(self) -> dict[str, int]:
        counts = {c: 0 for c in self.checks}
        for r in self.results:
            if r.verdict.detected:
                counts[r.check] += 1
        return counts

    @property
    def passed(self) -> bool:
        return not self.failed_traces

    def to_dict(self) -> dict:
        return {
            "judge": self.judge,
            "checks": self.checks,
            "redaction": self.redaction,
            "traces": len(self.trace_ids),
            "failed_traces": self.failed_traces,
            "fired": self.fired,
            "results": [
                {"trace_id": r.trace_id, "scenario": r.scenario, "check": r.check, "detected": r.verdict.detected,
                 "confidence": round(r.verdict.confidence, 4), "reason": r.verdict.reason}
                for r in self.results
            ],
        }


def run(traces: list[Trace], checks: list[Check], judge: Judge | None = None,
        redact: Callable[[str], str] | None = None) -> Report:
    """Rules always run. Judge checks need a judge, golden checks a golden file; unbound ones are dropped from the report.

    With ``redact``, every verdict reason is masked, and so is the trace sent to a judge marked ``remote``
    (an API model). Rules always read the raw trace: masking happens only on what can leave.
    """
    active = [c for c in checks if c.rule or (c.question and judge is not None)]
    report = Report(judge=judge.name if judge else "none", checks=[c.name for c in active],
                    redaction=getattr(redact, "name", "custom") if redact else "off")
    for t in traces:
        u = t.usage
        report.measures[t.trace_id] = {"tokens": float(sum(u.values())) if u else None,
                                       "latency_ms": t.latency_ms, "cost_usd": t.cost_usd}
        state = None
        for c in active:
            if c.rule:
                v = c.rule(t)
            else:
                if state is None:
                    state = t.render(getattr(judge, "budget", None))
                    if redact and getattr(judge, "remote", False):
                        state = redact(state)
                v = judge.ask(state, c.question)
            if redact:
                v = replace(v, reason=redact(v.reason))
            report.results.append(Result(t.trace_id, t.key, c.name, v))
    return report
