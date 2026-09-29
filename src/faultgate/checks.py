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
from faultgate.trace import Span, Trace

STEP_CAP_WORDS = ("step cap", "step limit", "max steps", "max_steps", "max iterations", "iteration limit")


@dataclass(frozen=True)
class Check:
    name: str
    description: str
    question: str | None = None            # asked of the judge
    rule: Callable[[Trace], Verdict] | None = None  # or decided by code
    needs: str | None = None               # "golden": only runs when --golden is given

    @property
    def kind(self) -> str:
        return "judge" if self.question else ("golden" if self.needs == "golden" else "rule")


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


class Golden(dict):
    """Expected final answers, keyed by scenario key (faultgate.scenario / gen_ai.agent.id / prompt hash) or by the prompt text itself."""

    @classmethod
    def load(cls, path: str | Path) -> "Golden":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(d, dict) or not all(isinstance(v, str) for v in d.values()):
            raise ValueError("golden: expected a JSON object of {scenario key or prompt: expected answer}")
        return cls(d)

    def expected(self, t: Trace) -> str | None:
        return self.get(t.key) or self.get(t.prompt)


def wrong_answer_rule(t: Trace, golden: Golden) -> Verdict:
    """Answered, and the expected answer does not appear in the final answer (case-insensitive)."""
    exp = golden.expected(t)
    if exp is None:
        return Verdict(False, 0.0, "no golden answer for this scenario")
    ans = t.answer.strip()
    if not ans:
        return Verdict(False, 1.0, "no answer to grade")
    ok = exp.lower() in ans.lower()
    return Verdict(not ok, 1.0, f"expected {exp!r} {'found' if ok else 'not found'} in answer")


def truncation_rule(t: Trace, golden: Golden) -> Verdict:
    """Answer carries the head of the expected token but not the whole token — e.g. 'Onyx' for 'Onyx-4413'."""
    exp = golden.expected(t)
    ans = t.answer.strip()
    if exp is None or not ans:
        return Verdict(False, 0.0 if exp is None else 1.0, "no golden answer for this scenario" if exp is None else "no answer to grade")
    if exp.lower() in ans.lower():
        return Verdict(False, 1.0, "expected answer present in full")
    head = re.split(r"[-_/ ]", exp, maxsplit=1)[0]
    if len(head) >= 3 and head != exp and re.search(r"\b" + re.escape(head) + r"\b", ans, re.I):
        return Verdict(True, 1.0, f"answer has {head!r} but not {exp!r}")
    return Verdict(False, 1.0, "no truncated form of the expected answer")


def normalize_tool_arguments(args: Any) -> str:
    """Normalize tool arguments into a deterministic canonical string."""
    if args is None:
        return "{}"
    if isinstance(args, str):
        s = args.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                parsed = json.loads(s)
                return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            except Exception:
                pass
        return re.sub(r"\s+", " ", s)
    if isinstance(args, dict):
        return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return str(args)


def repeated_call_rule(t: Trace) -> Verdict:
    """FAULTLINE P18: kappa 0.95 vs a blind second rater (Claude Opus 5.5) on 97 real runs (Exgentic, 6 benchmarks)."""
    steps = t.steps
    if len(steps) < 3:
        return Verdict(False, 1.0, "fewer than 3 tool calls")

    calls_by_key: dict[tuple[str, str], list[int]] = {}
    for idx, st in enumerate(steps):
        norm_args = normalize_tool_arguments(st.arguments)
        key = (st.tool, norm_args)
        calls_by_key.setdefault(key, []).append(idx)

    for (tool_name, norm_args), indices in calls_by_key.items():
        if len(indices) < 3:
            continue
        for i in range(len(indices) - 2):
            idx1, idx2, idx3 = indices[i], indices[i + 1], indices[i + 2]
            resp1 = re.sub(r"\s+", " ", steps[idx1].result).strip()
            resp2 = re.sub(r"\s+", " ", steps[idx2].result).strip()
            resp3 = re.sub(r"\s+", " ", steps[idx3].result).strip()
            if resp1 == resp2 == resp3:
                evidence_step = steps[idx3].index
                reason = (
                    f"Tool '{tool_name}' with arguments {norm_args[:60]} issued "
                    f">= 3 times (steps {steps[idx1].index}, {steps[idx2].index}, {evidence_step}) "
                    f"where all 3 responses are identical after whitespace normalization."
                )
                return Verdict(True, 1.0, reason)

    return Verdict(False, 1.0, "no repeated calls without progress")


ERROR_PREFIXES = (
    "Error:",
    "ERROR:",
    "Traceback (most recent call last)",
    "An error occurred while parsing tool arguments",
)


def _is_structural_error(result: str) -> tuple[bool, str]:
    """Determine whether a tool result represents a structural error."""
    s = result.strip()
    if not s:
        return False, ""

    candidate_objs: list[dict[str, Any]] = []
    if s.startswith("{") and s.endswith("}"):
        try:
            p = json.loads(s)
            if isinstance(p, dict):
                candidate_objs.append(p)
        except Exception:
            pass
    elif s.startswith("[") and s.endswith("]"):
        try:
            p = json.loads(s)
            if isinstance(p, list):
                for item in p:
                    if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
                        t_s = item["text"].strip()
                        if t_s.startswith("{") and t_s.endswith("}"):
                            try:
                                sub = json.loads(t_s)
                                if isinstance(sub, dict):
                                    candidate_objs.append(sub)
                            except Exception:
                                pass
        except Exception:
            pass

    for obj in candidate_objs:
        if "error" in obj and obj["error"]:
            return True, f"JSON top-level error: {str(obj['error'])[:80]}"
        for code_key in ("exit_code", "return_code", "exitcode", "returncode", "exit_status"):
            if code_key in obj:
                try:
                    if int(obj[code_key]) != 0:
                        return True, f"Non-zero {code_key}: {obj[code_key]}"
                except (ValueError, TypeError):
                    pass

    if "<tool_use_error>" in s and "</tool_use_error>" in s:
        return True, "<tool_use_error> tag present"

    for pfx in ERROR_PREFIXES:
        if s.startswith(pfx):
            return True, f"Response starts with '{pfx}'"

    return False, ""


def _span_error(s: Span) -> str | None:
    if s.status_code == 2:
        return "Span status error: Code 2"
    if s.attrs.get("error.type"):
        return f"Span error: {s.attrs['error.type']}"
    return None


def unrecovered_tool_error_rule(t: Trace) -> Verdict:
    """FAULTLINE P18: caught 13 of 23 confirmed unrecovered errors (recall 0.57, kappa 0.54); misses plain-text errors and run-ending API failures (faultgate has no check for those yet; read flagged traces manually)."""
    steps = t.steps

    for i, st in enumerate(steps):
        is_err, desc = _is_structural_error(st.result)
        if not is_err:
            continue
        tool_name = st.tool
        recovered = False
        for future_st in steps[i + 1:]:
            if future_st.tool == tool_name:
                future_is_err, _ = _is_structural_error(future_st.result)
                if not future_is_err:
                    recovered = True
                    break
        if not recovered:
            return Verdict(
                True,
                1.0,
                f"Tool '{tool_name}' failed at step {st.index} ({desc}) and no subsequent call to '{tool_name}' succeeded.",
            )

    steps_by_span: dict[str, list] = {}
    for st in steps:
        steps_by_span.setdefault(st.span_id, []).append(st)

    for i, span in enumerate(t.spans):
        err = _span_error(span)
        if err is not None:
            recovered = False
            for later_span in t.spans[i + 1:]:
                if _span_error(later_span) is not None:
                    continue
                later_steps = steps_by_span.get(later_span.span_id, [])
                if later_steps and not any(_is_structural_error(s.result)[0] for s in later_steps):
                    recovered = True
                    break
            if not recovered:
                return Verdict(True, 1.0, f"Unrecovered span error at span {i+1} of {len(t.spans)}: {err}")

    return Verdict(False, 1.0, "no unrecovered tool errors")


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
            "repeated_call",
            "Same tool and arguments called >=3 times with identical results.",
            rule=repeated_call_rule,
        ),
        Check(
            "unrecovered_tool_error",
            "A tool call produced a structural error and the tool was never used successfully afterwards.",
            rule=unrecovered_tool_error_rule,
        ),
        Check(
            "wrong_answer",
            "Answered, but the expected answer is not in the final answer (needs --golden).",
            rule=None, needs="golden",
        ),
        Check(
            "truncation",
            "Answer has the head of the expected token but not the whole token (needs --golden).",
            rule=None, needs="golden",
        ),
        Check(
            "wrong_direction",
            "Agent searched the dependency chain backwards on a multi-hop task.",
            question="Determine whether the agent made a MULTI_HOP_DIRECTION_ERROR. "
            "Rule: Return detected=true IF the agent searched in the reverse dependency direction across multi-hop entity links.",
        ),
    )
}



GOLDEN_RULES = {"wrong_answer": wrong_answer_rule, "truncation": truncation_rule}


def select(names: str | None, policy: Policy | None = None, golden: Golden | None = None) -> list[Check]:
    """Resolve check names; bind a policy / golden file to the checks that take one.

    A golden check with no golden file keeps ``rule=None`` and is skipped by the runner.
    """
    wanted = [n.strip() for n in names.split(",")] if names else list(CHECKS)
    out = []
    for n in wanted:
        if n not in CHECKS:
            raise ValueError(f"unknown check {n!r}; available: {', '.join(CHECKS)}")
        c = CHECKS[n]
        if n == "policy" and policy is not None:
            c = Check(c.name, c.description, rule=partial(policy_rule, policy=policy))
        if n in GOLDEN_RULES and golden is not None:
            c = Check(c.name, c.description, rule=partial(GOLDEN_RULES[n], golden=golden), needs="golden")
        out.append(c)
    return out
