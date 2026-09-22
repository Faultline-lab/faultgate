"""Built-in checks. Each is one yes/no question a judge answers about a trace.

The rule text is taken from FAULTLINE P5 (projects/p05_judge), where each was
calibrated against human failure-mode labels; see README "How much to trust a
verdict". Two P5 modes (answer truncation, premature stop) need an expected
answer, which recorded traces do not carry — they arrive with ``--golden`` in v1.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    name: str
    description: str
    question: str


CHECKS: dict[str, Check] = {
    c.name: c
    for c in (
        Check(
            "search_loop",
            "Agent looped on over-constrained queries that returned nothing until it ran out of steps.",
            "Determine whether the agent experienced an OVERCONSTRAINED_SEARCH_LOOP. "
            "Rule: Return detected=true IF the agent issued overly constrained, multi-word queries that returned 0 candidates, "
            "looping through reformulations until step cap exhaustion.",
        ),
        Check(
            "abstention",
            "Agent explicitly said it could not find the information instead of answering.",
            "Determine whether the agent output represents a RETRIEVAL_FAILURE_ABSTENTION. "
            "Rule: Return detected=true IF the agent explicitly stated that it could not find the information, "
            "refusing/abstaining from guessing rather than providing a fabricated answer.",
        ),
        Check(
            "wrong_direction",
            "Agent searched the dependency chain backwards on a multi-hop task.",
            "Determine whether the agent made a MULTI_HOP_DIRECTION_ERROR. "
            "Rule: Return detected=true IF the agent searched in the reverse dependency direction across multi-hop entity links.",
        ),
    )
}


def select(names: str | None) -> list[Check]:
    if not names:
        return list(CHECKS.values())
    out = []
    for n in names.split(","):
        n = n.strip()
        if n not in CHECKS:
            raise ValueError(f"unknown check {n!r}; available: {', '.join(CHECKS)}")
        out.append(CHECKS[n])
    return out
