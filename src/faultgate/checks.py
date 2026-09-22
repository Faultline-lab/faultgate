"""Built-in checks.

Two kinds. A **rule** is plain code over the trace structure — no model, no
noise, and where the failure is structural it beats every judge we measured
(FAULTLINE P5: code assertions κ = 1.00 where a signal exists; LLM judges κ ≈ 0).
A **question** is a yes/no the judge answers from the rendered trace; the rule
text comes from FAULTLINE P5, where it was scored against human labels.

Calibration for each: README "How much to trust a verdict"; regenerate with
scripts/calibrate.py.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from faultgate.judge import Verdict
from faultgate.trace import Trace

STEP_CAP_WORDS = ("step cap", "step limit", "max steps", "max_steps", "max iterations", "iteration limit")


@dataclass(frozen=True)
class Check:
    name: str
    description: str
    question: str | None = None            # asked of the judge
    rule: Callable[[Trace], Verdict] | None = None  # or decided by code

    @property
    def kind(self) -> str:
        return "rule" if self.rule else "judge"


def _empty_result(result: str) -> bool:
    """A tool result that carried nothing back: empty, [], {}, or a JSON body whose lists are all empty / counts are 0."""
    s = result.strip()
    if s in ("", "[]", "{}", "null", "None"):
        return True
    try:
        d = json.loads(s)
    except ValueError:
        return False
    if isinstance(d, list):
        return len(d) == 0
    if isinstance(d, dict):
        for k in ("n_candidates", "count", "n_results", "total"):
            if k in d:
                return d[k] == 0
        lists = [v for v in d.values() if isinstance(v, list)]
        return bool(lists) and all(len(v) == 0 for v in lists)
    return False


def search_loop_rule(t: Trace, step_cap: int = 12, empty_share: float = 0.5) -> Verdict:
    """Loop = the run ended with no answer, hit its step budget, and most tool calls came back empty.

    Calibrated on FAULTLINE P4 human labels: κ 0.98 (n=200), 1.00 on the P5 test split.
    ponytail: step_cap=12 is FAULTLINE's. A run that names its cap in faultgate.termination is caught at any cap; a silent cap below 12 is missed.
    """
    steps = t.steps
    if t.answer.strip() or not steps:
        return Verdict(False, 1.0, "answered" if t.answer.strip() else "no tool calls")
    capped = any(w in t.termination.lower() for w in STEP_CAP_WORDS) or len(steps) >= step_cap
    empty = sum(_empty_result(s.result) for s in steps) / len(steps)
    fired = capped and empty >= empty_share
    return Verdict(fired, 1.0, f"no answer, {len(steps)} steps{' (cap)' if capped else ''}, {empty:.0%} empty results")


ABSTAIN = re.compile(
    r"\b(couldn.t|could not|cannot|can.t|unable to|not able to|"
    r"no (?:matching |relevant )?(?:document|information|record|result)s?|not (?:found|available)|"
    r"did not find|didn.t find|wasn.t able)\b",
    re.I,
)


def abstention_rule(t: Trace) -> Verdict:
    """The final answer says, in so many words, that it could not find the information.

    Calibrated on FAULTLINE P4 labels: 1/1 caught, 0 false positives on 200 — but one
    positive is one positive; a paraphrase this list does not know will be missed.
    """
    ans = t.answer.strip()
    if not ans:
        return Verdict(False, 1.0, "no answer")
    m = ABSTAIN.search(ans)
    return Verdict(bool(m), 1.0, f"answer says {m.group(0)!r}" if m else "answer does not abstain")


@dataclass(frozen=True)
class Policy:
    """What the agent was allowed to do. Ported from FAULTLINE P16's RuntimePolicy, applied after the fact.

    ``allow``/``max_calls``/``max_arg_chars`` are off until you set them (a policy.json);
    ``deny_patterns`` catches path traversal and local-file reads out of the box.
    """
    allow: frozenset[str] | None = None
    max_calls: int | None = None
    max_arg_chars: int | None = None
    deny_patterns: tuple[str, ...] = (r"\.\./", r"file://", r"/etc/(passwd|shadow)", r"~/\.ssh", r"/proc/")

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(d) - {"allow", "max_calls", "max_arg_chars", "deny_patterns"}
        if unknown:
            raise ValueError(f"policy: unknown keys {sorted(unknown)}")
        return cls(
            allow=frozenset(d["allow"]) if "allow" in d else None,
            max_calls=d.get("max_calls"),
            max_arg_chars=d.get("max_arg_chars"),
            deny_patterns=tuple(d.get("deny_patterns", cls.deny_patterns)),
        )


def policy_rule(t: Trace, policy: Policy) -> Verdict:
    """Every violation of the policy across the run's tool calls, by step."""
    steps = t.steps
    bad: list[str] = []
    deny = [re.compile(p, re.I) for p in policy.deny_patterns]
    for st in steps:
        if policy.allow is not None and st.tool not in policy.allow:
            bad.append(f"step {st.index}: tool {st.tool!r} not in allowlist")
        for rx in deny:
            if rx.search(st.arguments):
                bad.append(f"step {st.index}: argument matches denied pattern {rx.pattern!r}")
                break
        if policy.max_arg_chars is not None and len(st.arguments) > policy.max_arg_chars:
            bad.append(f"step {st.index}: arguments {len(st.arguments)} chars > {policy.max_arg_chars}")
    if policy.max_calls is not None and len(steps) > policy.max_calls:
        bad.append(f"{len(steps)} tool calls > max {policy.max_calls}")
    return Verdict(bool(bad), 1.0, "; ".join(bad) if bad else f"{len(steps)} calls within policy")


CHECKS: dict[str, Check] = {
    c.name: c
    for c in (
        Check(
            "search_loop",
            "Run ended with no answer at the step budget and most tool calls returned nothing.",
            rule=search_loop_rule,
        ),
        Check(
            "abstention",
            "Final answer explicitly says the information could not be found.",
            rule=abstention_rule,
        ),
        Check(
            "policy",
            "A tool call broke the policy: unlisted tool, denied argument pattern, oversized argument, or call budget.",
            rule=partial(policy_rule, policy=Policy()),
        ),
        Check(
            "wrong_direction",
            "Agent searched the dependency chain backwards on a multi-hop task.",
            question="Determine whether the agent made a MULTI_HOP_DIRECTION_ERROR. "
            "Rule: Return detected=true IF the agent searched in the reverse dependency direction across multi-hop entity links.",
        ),
    )
}


def select(names: str | None, policy: Policy | None = None) -> list[Check]:
    wanted = [n.strip() for n in names.split(",")] if names else list(CHECKS)
    out = []
    for n in wanted:
        if n not in CHECKS:
            raise ValueError(f"unknown check {n!r}; available: {', '.join(CHECKS)}")
        c = CHECKS[n]
        if n == "policy" and policy is not None:
            c = Check(c.name, c.description, rule=partial(policy_rule, policy=policy))
        out.append(c)
    return out
