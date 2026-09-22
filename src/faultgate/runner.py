"""Run checks over traces with a judge. Pure: no I/O."""
from __future__ import annotations

from dataclasses import dataclass, field

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
                {"trace_id": r.trace_id, "scenario": r.scenario, "check": r.check, "detected": r.verdict.detected,
                 "confidence": round(r.verdict.confidence, 4), "reason": r.verdict.reason}
                for r in self.results
            ],
        }


def run(traces: list[Trace], checks: list[Check], judge: Judge | None = None) -> Report:
    """Rules always run. Judge checks run only when a judge is given; otherwise they are dropped from the report."""
    active = [c for c in checks if c.rule or judge is not None]
    report = Report(judge=judge.name if judge else "none", checks=[c.name for c in active])
    for t in traces:
        state = None
        for c in active:
            if c.rule:
                v = c.rule(t)
            else:
                state = state if state is not None else t.render(getattr(judge, "budget", None))
                v = judge.ask(state, c.question)
            report.results.append(Result(t.trace_id, t.key, c.name, v))
    return report
