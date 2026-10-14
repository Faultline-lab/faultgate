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
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from faultgate.judge import Verdict
from faultgate.trace import Span, Step, Trace, is_tool_span

STEP_CAP_WORDS = ("step cap", "step limit", "max steps", "max_steps", "max iterations", "iteration limit")


@dataclass(frozen=True)
class Check:
    name: str
    description: str
    question: str | None = None            # asked of the judge
    rule: Callable[[Trace], Verdict] | None = None  # or decided by code
    needs: str | None = None               # "golden" / "outcome": only runs when --golden / --outcomes is given

    @property
    def kind(self) -> str:
        return "judge" if self.question else (self.needs or "rule")


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


# "could not" alone is not abstention: a support agent that "cannot modify the order again" or is "unable to
# verify" an unauthenticated caller did its job. The inability has to be about finding the information.
ABSTAIN = re.compile(
    r"\b(?:couldn.t|could not|cannot|can.t|unable to|not able to|wasn.t able to|was not able to|did not|didn.t|failed to)"
    r"\s+(?:\w+\s+){0,2}?(?:find|locate|retrieve|identify|determine|look\s?up|discover)\b"
    r"|\bno (?:matching |relevant )?(?:document|information|record|result|source|data)s?\b"
    r"|\b(?:information|answer|record|result|document|data)s? (?:(?:was|were|is|are) )?not (?:found|available)\b"
    r"|\bnot found\b"
    r"|\b(?:don.t|do not) (?:know|have (?:that|this|any|enough|the) information)\b",
    re.I,
)


def abstention_rule(t: Trace) -> Verdict:
    """The final answer says, in so many words, that it could not find the information.

    Calibrated on FAULTLINE P4 labels: 1/1 caught, 0 false positives on 200 — but one
    positive is one positive; a paraphrase this list does not know will be missed.
    On 180 tau2 retail support runs it fires once ("couldn't locate the ... payment method");
    the earlier "cannot / unable to" list fired 21 times, on policy reminders and refusals.
    """
    ans = t.answer.strip()
    if not ans:
        return Verdict(False, 1.0, "no answer")
    m = ABSTAIN.search(ans)
    return Verdict(bool(m), 1.0, f"answer says {m.group(0)!r}" if m else "answer does not abstain")


@dataclass(frozen=True)
class Limit:
    """A numeric floor/ceiling on one tool argument, e.g. issue_refund.amount <= 500."""
    tool: str
    arg: str                    # dotted path into the JSON arguments
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class Identity:
    """A tool argument that must equal the run's authenticated principal (a root span attribute)."""
    arg: str
    session_attr: str = "enduser.id"
    tools: frozenset[str] | None = None  # None: every tool


@dataclass(frozen=True)
class Predicate:
    """A typed condition on a dotted path into a parsed JSON tool result: ``op`` is equals, in, not_in or exists."""
    path: str
    op: str
    value: Any


@dataclass(frozen=True)
class Requirement:
    """``tool`` may only run after an earlier call to one of ``after`` (whose result contains ``result_contains``,
    if set, and whose ``same_arg`` argument equals this call's, if set: the order that was looked up is the order changed).

    0.4, each off by default: ``success`` (the earlier call must have succeeded), ``result`` (typed predicates on its
    parsed JSON result), ``latest`` (only the most recent matching call counts) and ``invalidated_by`` (a successful
    call to one of these tools, on the same ``same_arg`` value, voids every earlier call)."""
    tool: str
    after: tuple[str, ...]
    result_contains: str | None = None
    same_arg: str | None = None
    success: bool = False
    result: tuple[Predicate, ...] = ()
    latest: bool = False
    invalidated_by: frozenset[str] = frozenset()

    @property
    def stateful(self) -> bool:
        return self.success or bool(self.result) or self.latest or bool(self.invalidated_by)


@dataclass(frozen=True)
class PerValueBudget:
    """At most ``max`` successful calls to ``tools`` per value of ``arg``, e.g. one exchange per order_id.
    A call that returned an error changed nothing: an agent that fixes a bad item id and retries modified the order once."""
    tools: frozenset[str]
    arg: str
    max: int


@dataclass(frozen=True)
class AllowedValues:
    """``tool``'s ``arg`` must be one of ``allowed``, e.g. a cancellation reason from a fixed list."""
    tool: str
    arg: str
    allowed: tuple[str, ...]


@dataclass(frozen=True)
class RequiredArgs:
    """Every call to ``tool`` must carry each of ``args`` (dotted paths), and not empty."""
    tool: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class Confirmation:
    """Each call to one of ``tools`` needs an earlier successful ``by`` call (for the same ``same_arg`` value, if set),
    or the attribute ``faultgate.confirmed = true`` on its own span. Free-text confirmation is not read."""
    tools: frozenset[str]
    by: str
    same_arg: str | None = None


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _shape(where: str, d: Any, required: set[str], optional: set[str]) -> dict[str, Any]:
    if not isinstance(d, dict):
        raise ValueError(f"policy: {where} must be an object")
    missing = required - set(d)
    if missing:
        raise ValueError(f"policy: {where} needs {' and '.join(sorted(f'{k!r}' for k in missing))}")
    unknown = set(d) - required - optional
    if unknown:
        raise ValueError(f"policy: {where} has unknown keys {sorted(unknown)}")
    for k in (required | optional) & set(d):
        if k == "max" and "per_arg" in where:
            if not isinstance(d[k], int) or isinstance(d[k], bool) or d[k] < 1:
                raise ValueError(f"policy: {where}.max must be a whole number >= 1")
        elif k in ("min", "max"):
            if not _is_num(d[k]) or not math.isfinite(d[k]):
                raise ValueError(f"policy: {where}.{k} must be a finite number")
        elif k == "tools":
            if not isinstance(d[k], list) or not all(isinstance(x, str) for x in d[k]):
                raise ValueError(f"policy: {where}.tools must be a list of tool names")
        elif k == "after":
            if not (isinstance(d[k], str) or (isinstance(d[k], list) and d[k] and all(isinstance(x, str) for x in d[k]))):
                raise ValueError(f"policy: {where}.after must be a tool name or a list of tool names")
        elif k == "allowed":
            if not isinstance(d[k], list) or not d[k] or not all(isinstance(x, str) for x in d[k]):
                raise ValueError(f"policy: {where}.allowed must be a list of strings")
        elif k in ("success", "latest"):
            if not isinstance(d[k], bool):
                raise ValueError(f"policy: {where}.{k} must be true or false")
        elif k in ("invalidated_by", "args"):
            if not isinstance(d[k], list) or not d[k] or not all(isinstance(x, str) and x for x in d[k]):
                raise ValueError(f"policy: {where}.{k} must be a non-empty list of names")
        elif k == "result":
            if not isinstance(d[k], list) or not d[k]:
                raise ValueError(f"policy: {where}.result must be a non-empty list of predicates")
            for i, pred in enumerate(d[k]):
                _predicate(f"{where}.result[{i}]", pred)
        elif not isinstance(d[k], str):
            raise ValueError(f"policy: {where}.{k} must be a string")
    return d


PREDICATE_OPS = ("equals", "in", "not_in", "exists")


def _predicate(where: str, d: Any) -> Predicate:
    if not isinstance(d, dict):
        raise ValueError(f"policy: {where} must be an object")
    if not isinstance(d.get("path"), str) or not d["path"]:
        raise ValueError(f"policy: {where} needs a 'path' (a dotted path into the JSON result)")
    unknown = set(d) - {"path", *PREDICATE_OPS}
    if unknown:
        raise ValueError(f"policy: {where} has unknown keys {sorted(unknown)}")
    ops = [k for k in PREDICATE_OPS if k in d]
    if len(ops) != 1:
        raise ValueError(f"policy: {where} needs exactly one of {', '.join(PREDICATE_OPS)}")
    op = ops[0]
    if op in ("in", "not_in") and not isinstance(d[op], list):
        raise ValueError(f"policy: {where}.{op} must be a list")
    if op == "exists" and not isinstance(d[op], bool):
        raise ValueError(f"policy: {where}.exists must be true or false")
    return Predicate(d["path"], op, tuple(d[op]) if op in ("in", "not_in") else d[op])


def _items(d: dict[str, Any], key: str) -> list[Any]:
    v = d.get(key, [])
    if not isinstance(v, list):
        raise ValueError(f"policy: {key} must be a list")
    return v


@dataclass(frozen=True)
class Policy:
    """What the agent was allowed to do. Ported from FAULTLINE P16's RuntimePolicy, applied after the fact.

    ``allow``/``max_calls``/``max_arg_chars`` are off until you set them (a policy.json);
    ``deny_patterns`` catches path traversal and local-file reads out of the box.
    ``limits``/``identity``/``requires``/``max_calls_per_arg``/``values``/``required_args``/``confirmation`` are
    business rules on argument values and tool order; each fails closed when it cannot be checked.
    ``require_complete`` makes an incomplete run (no answer, error termination, no end time) a violation.
    """
    allow: frozenset[str] | None = None
    max_calls: int | None = None
    max_arg_chars: int | None = None
    deny_patterns: tuple[str, ...] = (r"\.\./", r"file://", r"/etc/(passwd|shadow)", r"~/\.ssh", r"/proc/")
    limits: tuple[Limit, ...] = ()
    identity: Identity | None = None
    requires: tuple[Requirement, ...] = ()
    max_calls_per_arg: tuple[PerValueBudget, ...] = ()
    values: tuple[AllowedValues, ...] = ()
    required_args: tuple[RequiredArgs, ...] = ()
    confirmation: Confirmation | None = None
    require_complete: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Policy":
        unknown = set(d) - {"allow", "max_calls", "max_arg_chars", "deny_patterns", "limits", "identity", "requires",
                            "max_calls_per_arg", "values", "required_args", "confirmation", "require_complete"}
        if unknown:
            raise ValueError(f"policy: unknown keys {sorted(unknown)}")
        limits = []
        for i, x in enumerate(_items(d, "limits")):
            x = _shape(f"limits[{i}]", x, {"tool", "arg"}, {"min", "max"})
            if "min" not in x and "max" not in x:
                raise ValueError(f"policy: limits[{i}] needs 'min' or 'max'")
            if "min" in x and "max" in x and x["min"] > x["max"]:
                raise ValueError(f"policy: limits[{i}] min {x['min']} > max {x['max']}")
            limits.append(Limit(x["tool"], x["arg"], x.get("min"), x.get("max")))
        identity = None
        if "identity" in d:
            x = _shape("identity", d["identity"], {"arg"}, {"session_attr", "tools"})
            identity = Identity(x["arg"], x.get("session_attr", "enduser.id"),
                                frozenset(x["tools"]) if "tools" in x else None)
        requires = [
            Requirement(x["tool"], tuple([x["after"]] if isinstance(x["after"], str) else x["after"]),
                        x.get("result_contains"), x.get("same_arg"), x.get("success", False),
                        tuple(_predicate(f"requires[{i}].result[{j}]", p) for j, p in enumerate(x.get("result", []))),
                        x.get("latest", False), frozenset(x.get("invalidated_by", [])))
            for i, x in enumerate(_shape(f"requires[{i}]", r, {"tool", "after"},
                                         {"result_contains", "same_arg", "success", "result", "latest", "invalidated_by"})
                                  for i, r in enumerate(_items(d, "requires")))
        ]
        budgets = [
            PerValueBudget(frozenset(x["tools"]), x["arg"], x["max"])
            for x in (_shape(f"max_calls_per_arg[{i}]", b, {"tools", "arg", "max"}, set())
                      for i, b in enumerate(_items(d, "max_calls_per_arg")))
        ]
        values = [
            AllowedValues(x["tool"], x["arg"], tuple(x["allowed"]))
            for x in (_shape(f"values[{i}]", v, {"tool", "arg", "allowed"}, set())
                      for i, v in enumerate(_items(d, "values")))
        ]
        required_args = [
            RequiredArgs(x["tool"], tuple(x["args"]))
            for x in (_shape(f"required_args[{i}]", v, {"tool", "args"}, set())
                      for i, v in enumerate(_items(d, "required_args")))
        ]
        confirmation = None
        if "confirmation" in d:
            x = _shape("confirmation", d["confirmation"], {"tools", "by"}, {"same_arg"})
            confirmation = Confirmation(frozenset(x["tools"]), x["by"], x.get("same_arg"))
        if not isinstance(d.get("require_complete", False), bool):
            raise ValueError("policy: require_complete must be true or false")
        return cls(
            allow=frozenset(d["allow"]) if "allow" in d else None,
            max_calls=d.get("max_calls"),
            max_arg_chars=d.get("max_arg_chars"),
            deny_patterns=tuple(d.get("deny_patterns", cls.deny_patterns)),
            limits=tuple(limits),
            identity=identity,
            requires=tuple(requires),
            max_calls_per_arg=tuple(budgets),
            values=tuple(values),
            required_args=tuple(required_args),
            confirmation=confirmation,
            require_complete=d.get("require_complete", False),
        )


_ABSENT = object()


def _dig(args: dict[str, Any], path: str) -> Any:
    cur: Any = args
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _ABSENT
        cur = cur[part]
    return cur


def _number(v: Any) -> float | None:
    """A JSON number, or a string like ' $1,200.50 '. Booleans and non-finite values are not numbers."""
    if _is_num(v):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, str):
        s = v.strip().removeprefix("$").replace(",", "")
        try:
            f = float(s)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def _args_object(arguments: str) -> dict[str, Any] | None:
    """The arguments as a JSON object: ``{}`` when there are none, one extra decode for JSON encoded twice, else None."""
    if not arguments.strip():
        return {}
    try:
        d = json.loads(arguments)
        if isinstance(d, str):
            d = json.loads(d)
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def _business_violations(t: Trace, st: Step, earlier: list[Step], policy: Policy) -> list[str]:
    bad: list[str] = []
    args = _args_object(st.arguments)
    for lim in policy.limits:
        if st.tool != lim.tool:
            continue
        if args is None:
            bad.append(f"step {st.index}: {st.tool} arguments are not a JSON object, cannot check {lim.arg}")
            continue
        raw = _dig(args, lim.arg)
        if raw is _ABSENT:
            continue
        n = _number(raw)
        if n is None:
            bad.append(f"step {st.index}: {st.tool}.{lim.arg} {raw!r} is not a number")
        elif lim.max is not None and n > lim.max:
            bad.append(f"step {st.index}: {st.tool}.{lim.arg} {n:g} > max {lim.max:g}")
        elif lim.min is not None and n < lim.min:
            bad.append(f"step {st.index}: {st.tool}.{lim.arg} {n:g} < min {lim.min:g}")
    ident = policy.identity
    if ident and (ident.tools is None or st.tool in ident.tools):
        # unreadable arguments: fail closed for a listed tool, or when the argument's name is visible in the text
        if args is None:
            if ident.tools is not None or ident.arg.split(".")[-1] in st.arguments:
                bad.append(f"step {st.index}: {st.tool} arguments are not a JSON object, cannot check {ident.arg}")
            raw = _ABSENT
        else:
            raw = _dig(args, ident.arg)
        if raw is not _ABSENT:
            root = t.root
            session = root.attrs.get(ident.session_attr) if root else None
            if session is None:
                bad.append(f"step {st.index}: cannot verify {ident.arg}, root span has no {ident.session_attr}")
            elif str(raw).strip() != str(session).strip():
                bad.append(f"step {st.index}: {st.tool}.{ident.arg} {str(raw).strip()!r} != session "
                           f"{ident.session_attr} {str(session).strip()!r}")
    for allowed in policy.values:
        if st.tool != allowed.tool:
            continue
        if args is None:
            bad.append(f"step {st.index}: {st.tool} arguments are not a JSON object, cannot check {allowed.arg}")
            continue
        raw = _dig(args, allowed.arg)
        if raw is not _ABSENT and str(raw).strip() not in allowed.allowed:
            bad.append(f"step {st.index}: {st.tool}.{allowed.arg} {str(raw).strip()!r} not in allowed values")
    for req in policy.requires:
        if st.tool != req.tool:
            continue
        if req.stateful:
            bad += _stateful_requirement(t, st, args, earlier, req)
            continue
        cond = f" returned {req.result_contains!r}" if req.result_contains is not None else ""
        if req.same_arg is None:
            ok = any(e.tool in req.after and (req.result_contains is None or req.result_contains in e.result) for e in earlier)
        else:
            mine = _dig(args, req.same_arg) if args is not None else _ABSENT
            if mine is _ABSENT:
                bad.append(f"step {st.index}: {st.tool} has no readable {req.same_arg}, cannot match it to {' or '.join(req.after)}")
                continue
            cond += f" for the same {req.same_arg} {str(mine).strip()!r}"
            ok = any(e.tool in req.after and (req.result_contains is None or req.result_contains in e.result)
                     and _same(e, req.same_arg, mine) for e in earlier)
        if not ok:
            bad.append(f"step {st.index}: {st.tool} before {' or '.join(req.after)}{cond}")
    for ra in policy.required_args:
        if st.tool != ra.tool:
            continue
        if args is None:
            bad.append(f"step {st.index}: {st.tool} arguments are not a JSON object, cannot check required {ra.args[0]}")
            continue
        for a in ra.args:
            if _empty(_dig(args, a)):
                bad.append(f"step {st.index}: {st.tool} missing required argument {a}")
    conf = policy.confirmation
    if conf and st.tool in conf.tools and not _confirmed_on_span(t, st):
        mine = _ABSENT
        if conf.same_arg is not None:
            mine = _dig(args, conf.same_arg) if args is not None else _ABSENT
            if mine is _ABSENT:
                bad.append(f"step {st.index}: {st.tool} has no readable {conf.same_arg}, cannot match it to a {conf.by}")
                return bad
        if not any(e.tool == conf.by and _succeeded(t, e) and (conf.same_arg is None or _same(e, conf.same_arg, mine))
                   for e in earlier):
            scope = f" for the same {conf.same_arg} {str(mine).strip()!r}" if conf.same_arg is not None else ""
            bad.append(f"step {st.index}: {st.tool} without an earlier successful {conf.by}{scope} "
                       "(or faultgate.confirmed = true on its span)")
    return bad


def _empty(v: Any) -> bool:
    return v is _ABSENT or v is None or (isinstance(v, (str, list, dict)) and not (v.strip() if isinstance(v, str) else v))


def _span_of(t: Trace, st: Step) -> Span | None:
    return next((s for s in t.spans if s.span_id == st.span_id), None)


def _succeeded(t: Trace, st: Step) -> bool:
    """No structural error in the result, and no error on the call's own tool span."""
    if _is_structural_error(st.result)[0]:
        return False
    span = _span_of(t, st)
    return not (span is not None and is_tool_span(span) and _span_error(span))


def _confirmed_on_span(t: Trace, st: Step) -> bool:
    span = _span_of(t, st)
    return span is not None and span.attrs.get("faultgate.confirmed") in (True, "true", "True")


_NOT_JSON = object()


def _json_result(result: str) -> Any:
    """The result parsed as JSON (one extra decode for JSON encoded twice), else _NOT_JSON."""
    try:
        v = json.loads(result)
        if isinstance(v, str):
            v = json.loads(v)
    except ValueError:
        return _NOT_JSON
    return v if isinstance(v, (dict, list)) else _NOT_JSON


def _path(v: Any, path: str) -> Any:
    """Dotted path into parsed JSON; a numeric part indexes a list."""
    for part in path.split("."):
        if isinstance(v, dict) and part in v:
            v = v[part]
        elif isinstance(v, list) and part.isdigit() and int(part) < len(v):
            v = v[int(part)]
        else:
            return _ABSENT
    return v


def _json_eq(a: Any, b: Any) -> bool:
    """JSON equality that keeps types apart: true is not 1, "1" is not 1."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_num(a) and _is_num(b):
        return a == b
    return type(a) is type(b) and a == b


def _describe(p: Predicate) -> str:
    if p.op == "equals":
        return f"{p.path} == {p.value!r}"
    if p.op == "exists":
        return f"{p.path} present" if p.value else f"no {p.path}"
    return f"{p.path} {'in' if p.op == 'in' else 'not in'} {list(p.value)!r}"


def _unmet(t: Trace, e: Step, req: Requirement) -> str | None:
    """Why the earlier call ``e`` does not satisfy ``req``, or None when it does."""
    if req.success and not _succeeded(t, e):
        return "failed"
    if req.result_contains is not None and req.result_contains not in e.result:
        return f"result does not contain {req.result_contains!r}"
    if req.result:
        doc = _json_result(e.result)
        if doc is _NOT_JSON:
            return "result is not JSON"
        for p in req.result:
            got = _path(doc, p.path)
            if p.op == "exists":
                if (got is not _ABSENT) != p.value:
                    return f"result has {'no ' if p.value else ''}{p.path}"
                continue
            if got is _ABSENT:
                return f"result has no {p.path}"
            if p.op == "equals" and not _json_eq(got, p.value):
                return f"{p.path} is {got!r}, not {p.value!r}"
            if p.op == "in" and not any(_json_eq(got, x) for x in p.value):
                return f"{p.path} is {got!r}, not in {list(p.value)!r}"
            if p.op == "not_in" and any(_json_eq(got, x) for x in p.value):
                return f"{p.path} is {got!r}"
    return None


def _stateful_requirement(t: Trace, st: Step, args: dict[str, Any] | None, earlier: list[Step],
                          req: Requirement) -> list[str]:
    after = " or ".join(req.after)
    mine: Any = _ABSENT
    if req.same_arg is not None:
        mine = _dig(args, req.same_arg) if args is not None else _ABSENT
        if mine is _ABSENT:
            return [f"step {st.index}: {st.tool} has no readable {req.same_arg}, cannot match it to {after}"]

    def bound(e: Step) -> bool:
        return req.same_arg is None or _same(e, req.same_arg, mine)

    cands = [e for e in earlier if e.tool in req.after and bound(e)]
    voided = None
    if req.invalidated_by:
        inv = [e for e in earlier if e.tool in req.invalidated_by and bound(e) and _succeeded(t, e)]
        if inv:
            stale = [e for e in cands if e.index < inv[-1].index]
            cands = [e for e in cands if e.index > inv[-1].index]
            if stale and not cands:
                voided = (stale[-1], inv[-1])
    if req.latest:
        cands = cands[-1:]
    if any(_unmet(t, e, req) is None for e in cands):
        return []
    cond = (f" before {'a successful ' if req.success else ''}{after}"
            + (f" returned {req.result_contains!r}" if req.result_contains is not None else "")
            + (" with " + " and ".join(_describe(p) for p in req.result) if req.result else "")
            + (f" for the same {req.same_arg} {str(mine).strip()!r}" if req.same_arg is not None else ""))
    if voided:
        why = f" ({voided[0].tool} at step {voided[0].index} voided by {voided[1].tool} at step {voided[1].index})"
    elif cands:
        e = cands[-1]
        why = (f" (latest {e.tool} at step {e.index}: {_unmet(t, e, req)})" if req.latest
               else f" (step {e.index} {e.tool}: {_unmet(t, e, req)})")
    else:
        why = ""
    return [f"step {st.index}: {st.tool}{cond}{why}"]


def _same(step: Step, arg: str, value: Any) -> bool:
    args = _args_object(step.arguments)
    other = _dig(args, arg) if args is not None else _ABSENT
    return other is not _ABSENT and str(other).strip() == str(value).strip()


def _budget_violations(steps: list[Step], policy: Policy) -> list[str]:
    """Successful calls over a per-value budget."""
    bad: list[str] = []
    for b in policy.max_calls_per_arg:
        seen: dict[str, list[int]] = {}
        for st in steps:
            if st.tool not in b.tools or _is_structural_error(st.result)[0]:
                continue
            args = _args_object(st.arguments)
            v = _dig(args, b.arg) if args is not None else _ABSENT
            if v is _ABSENT:
                bad.append(f"step {st.index}: {st.tool} has no readable {b.arg}, cannot count it against max {b.max} per {b.arg}")
                continue
            idx = seen.setdefault(str(v).strip(), [])
            idx.append(st.index)
            if len(idx) == b.max + 1:
                bad.append(f"step {st.index}: {len(idx)} successful {'/'.join(sorted(b.tools))} calls for {b.arg} "
                           f"{str(v).strip()!r} > max {b.max} (steps {', '.join(map(str, idx))})")
    return bad


# terminations that mean the run did not finish: errors, aborts, budgets
ERROR_TERMINATION = re.compile(r"error|exception|abort|fail|time ?out|timed out|crash|too many|max[_ ]?steps|"
                               r"step (?:cap|limit)|max iterations|iteration limit", re.I)


def _incomplete(t: Trace) -> list[str]:
    root = t.root
    if root is None:
        return ["run incomplete: no root span"]
    bad = []
    if not t.answer.strip():
        bad.append("run incomplete: no final answer")
    if ERROR_TERMINATION.search(t.termination):
        bad.append(f"run incomplete: terminated {t.termination!r}")
    err = _span_error(root)
    if err:
        bad.append(f"run incomplete: root span error ({err})")
    if not root.end_ns:
        bad.append("run incomplete: root span has no end time")
    return bad


def policy_rule(t: Trace, policy: Policy) -> Verdict:
    """Every violation of the policy across the run's tool calls, by step."""
    steps = t.steps
    bad: list[str] = []
    deny = [re.compile(p, re.I) for p in policy.deny_patterns]
    for i, st in enumerate(steps):
        if policy.allow is not None and st.tool not in policy.allow:
            bad.append(f"step {st.index}: tool {st.tool!r} not in allowlist")
        for rx in deny:
            if rx.search(st.arguments):
                bad.append(f"step {st.index}: argument matches denied pattern {rx.pattern!r}")
                break
        if policy.max_arg_chars is not None and len(st.arguments) > policy.max_arg_chars:
            bad.append(f"step {st.index}: arguments {len(st.arguments)} chars > {policy.max_arg_chars}")
        bad += _business_violations(t, st, steps[:i], policy)
    bad += _budget_violations(steps, policy)
    if policy.require_complete:
        bad += _incomplete(t)
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


CUTOFF_REASONS = {"length", "max_tokens", "max_output_tokens", "model_length", "model_context_window_exceeded"}
OVERFLOW = re.compile(
    r"context[ _-]?length[ _-]?exceeded|maximum context length|context[ _-]?window[ _-]?(?:exceeded|overflow)"
    r"|exceeds?\s+the\s+(?:model'?s\s+)?(?:context|token)\s+(?:window|limit)|prompt is too long|too many (?:input )?tokens"
    r"|token limit (?:reached|exceeded)|max(?:imum)?[ _-]?tokens? (?:limit )?(?:reached|exceeded)"
    r"|exceeds?\s+the\s+maximum\s+number\s+of\s+tokens|input is too long for (?:the )?requested model",
    re.I,
)


def context_overflow_rule(t: Trace) -> Verdict:
    """The run hit the model's context window or output token limit.

    Two signals, both structural: a span whose ``gen_ai.response.finish_reasons`` says the output was cut
    (``length``, ``max_tokens``, ...), or a provider error naming the context limit in a span status message,
    ``error.type``, a tool result that is an error, or the root's ``faultgate.termination``. Text that merely talks about tokens
    does not count: the phrases are the error forms providers return.
    """
    for i, s in enumerate(t.spans, start=1):
        raw = s.attrs.get("gen_ai.response.finish_reasons")
        reasons = raw if isinstance(raw, (list, tuple)) else ([] if raw is None else [raw])
        cut = next((str(r) for r in reasons if str(r).lower() in CUTOFF_REASONS), None)
        if cut:
            return Verdict(True, 1.0, f"span {i} ({s.name}) stopped on finish reason {cut!r}: output cut at the token limit")
        for where, text in (("status", s.status_message), ("error.type", str(s.attrs.get("error.type", "")))):
            m = OVERFLOW.search(text)
            if m:
                return Verdict(True, 1.0, f"span {i} ({s.name}) {where}: {m.group(0)!r}")
    for st in t.steps:
        # only results that are errors: a search hit quoting the error text is not an overflow
        if not (st.result.lstrip().startswith("Error") or _is_structural_error(st.result)[0]):
            continue
        m = OVERFLOW.search(st.result)
        if m:
            return Verdict(True, 1.0, f"step {st.index} ({st.tool}) result: {m.group(0)!r}")
    m = OVERFLOW.search(t.termination)
    if m:
        return Verdict(True, 1.0, f"run terminated: {m.group(0)!r}")
    return Verdict(False, 1.0, "no context overflow or token cutoff")


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
            "A tool call broke the policy: unlisted tool, denied argument pattern, oversized argument, call budget, "
            "argument limit, identity mismatch, missing prerequisite, per-value budget, value not allowed, "
            "missing required argument, missing confirmation, or (with require_complete) an incomplete run.",
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
            "context_overflow",
            "The run hit the context window or output token limit (finish reason or provider error).",
            rule=context_overflow_rule,
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


class Outcomes(dict):
    """Task outcomes from the user's own evaluator (e.g. tau2 reward == 1), keyed like golden answers:
    scenario key (faultgate.scenario / gen_ai.agent.id / prompt hash) or the prompt text itself."""

    @classmethod
    def load(cls, path: str | Path) -> "Outcomes":
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except ValueError as e:
            raise ValueError(f"outcomes: {path} is not JSON ({e})") from e
        if not isinstance(d, dict) or not all(isinstance(v, bool) for v in d.values()):
            raise ValueError("outcomes: expected a JSON object of {scenario key or prompt: true | false}")
        return cls(d)

    def outcome(self, t: Trace) -> bool | None:
        for k in (t.key, t.prompt):
            if k in self:
                return self[k]
        return None


def outcome_rule(t: Trace, outcomes: Outcomes) -> Verdict:
    """Fires when the user's evaluator marked the scenario failed. Scenarios without an outcome are not judged."""
    ok = outcomes.outcome(t)
    if ok is None:
        return Verdict(False, 0.0, "no outcome for this scenario")
    return Verdict(not ok, 1.0, "outcome true in the outcomes file" if ok else "outcome false in the outcomes file")


# not in CHECKS: it exists only when an outcomes file is given, so default reports keep their 0.3.1 check set
OUTCOME = Check("outcome", "The user's own evaluator marked the scenario failed (needs --outcomes).", rule=None, needs="outcome")


def select(names: str | None, policy: Policy | None = None, golden: Golden | None = None,
           outcomes: Outcomes | None = None) -> list[Check]:
    """Resolve check names; bind a policy / golden / outcomes file to the checks that take one.

    A golden check with no golden file keeps ``rule=None`` and is skipped by the runner. ``outcome`` joins the
    default set only when an outcomes file is given.
    """
    available = {**CHECKS, OUTCOME.name: OUTCOME}
    wanted = [n.strip() for n in names.split(",")] if names else list(CHECKS) + ([OUTCOME.name] if outcomes is not None else [])
    out = []
    for n in wanted:
        if n not in available:
            raise ValueError(f"unknown check {n!r}; available: {', '.join(available)}")
        c = available[n]
        if n == "policy" and policy is not None:
            c = Check(c.name, c.description, rule=partial(policy_rule, policy=policy))
        if n in GOLDEN_RULES and golden is not None:
            c = Check(c.name, c.description, rule=partial(GOLDEN_RULES[n], golden=golden), needs="golden")
        if n == OUTCOME.name and outcomes is not None:
            c = Check(c.name, c.description, rule=partial(outcome_rule, outcomes=outcomes), needs="outcome")
        out.append(c)
    return out
