"""Run checks over traces with a judge. Pure: no I/O."""
from __future__ import annotations

from dataclasses import dataclass, field

from faultgate.checks import Check
from faultgate.judge import Judge, Verdict
from faultgate.trace import Trace


@dataclass(frozen=True)
class Result:
    trace_id: str
    check: str
    verdict: Verdict


@dataclass
class Report:
    judge: str
    checks: list[str]
    results: list[Result] = field(default_factory=list)

    @property
    def trace_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.results:
            seen.setdefault(r.trace_id)
        return list(seen)

    @property
    def failed_traces(self) -> list[str]:
        return [t for t in self.trace_ids if any(r.detected for r in self.for_trace(t))]

    def for_trace(self, trace_id: str) -> list[Verdict]:
        return [r.verdict for r in self.results if r.trace_id == trace_id]

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
            "traces": len(self.trace_ids),
            "failed_traces": self.failed_traces,
            "fired": self.fired,
            "results": [
                {"trace_id": r.trace_id, "check": r.check, "detected": r.verdict.detected,
                 "confidence": round(r.verdict.confidence, 4), "reason": r.verdict.reason}
                for r in self.results
            ],
        }


def run(traces: list[Trace], checks: list[Check], judge: Judge) -> Report:
    report = Report(judge=judge.name, checks=[c.name for c in checks])
    for t in traces:
        state = t.render()
        for c in checks:
            report.results.append(Result(t.trace_id, c.name, judge.ask(state, c.question)))
    return report
