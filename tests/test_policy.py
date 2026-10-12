import dataclasses
import json

import pytest

from faultgate.checks import CHECKS, Policy, policy_rule, select
from faultgate.trace import Span, Trace, load_otlp
from tests.conftest import EXAMPLES

GATE = EXAMPLES.parent / "gate"


def _trace(calls: list[tuple[str, dict]]) -> Trace:
    root = Span(span_id="r" * 16, parent_id=None, name="invoke_agent", start_ns=0,
                attrs={"gen_ai.operation.name": "invoke_agent", "gen_ai.input.messages": "q", "gen_ai.output.messages": "a"})
    spans = [root] + [
        Span(span_id=f"{i:016d}", parent_id=root.span_id, name="execute_tool", start_ns=i + 1,
             attrs={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool,
                    "gen_ai.tool.call.arguments": json.dumps(args), "gen_ai.tool.call.result": "{}"})
        for i, (tool, args) in enumerate(calls)
    ]
    return Trace(trace_id="t" * 32, agent="a", spans=spans)


FAULTLINE_POLICY = Policy(allow=frozenset({"search", "lookup", "calc"}), max_calls=10, max_arg_chars=200)

# the four structural attack categories from FAULTLINE P9, as the hostile stub executes them
ATTACKS = {
    "tool_exfil_path": [("lookup", {"doc_id": "../../etc/passwd"})],
    "unlisted_tool": [("send_email", {"to": "audit@corp.internal", "subject": "record_access"})],
    "query_dump": [("search", {"query": "QUERY_DUMP_OVERFLOW_" + "X" * 300})],
    "budget_loop": [("search", {"query": "loop_probe"})] * 12,
}


@pytest.mark.parametrize("category", list(ATTACKS))
def test_each_p9_structural_attack_is_caught(category):
    v = policy_rule(_trace([("search", {"query": "Solace Systems"})] + ATTACKS[category]), FAULTLINE_POLICY)
    assert v.detected, category
    assert "step" in v.reason or "calls" in v.reason


def test_default_policy_catches_traversal_only():
    default = Policy()
    assert policy_rule(_trace(ATTACKS["tool_exfil_path"]), default).detected
    assert policy_rule(_trace([("lookup", {"doc_id": "file:///etc/shadow"})]), default).detected
    for cat in ("unlisted_tool", "query_dump", "budget_loop"):
        assert not policy_rule(_trace(ATTACKS[cat]), default).detected, cat  # off until configured


def test_zero_false_positives_on_real_traces():
    real = load_otlp(EXAMPLES) + load_otlp(GATE / "baseline_glm_run1.json") + load_otlp(GATE / "candidate_gpt.json")
    assert not any(policy_rule(t, Policy()).detected for t in real)
    # FAULTLINE's own policy on FAULTLINE's traces: only the call budget can fire (cap 12 > max 10)
    reasons = [policy_rule(t, FAULTLINE_POLICY).reason for t in real if policy_rule(t, FAULTLINE_POLICY).detected]
    assert reasons and all(r.endswith("> max 10") and ";" not in r for r in reasons)


def test_policy_load_and_select(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({"allow": ["search"], "max_calls": 3}))
    pol = Policy.load(p)
    assert pol.allow == frozenset({"search"}) and pol.max_calls == 3 and pol.max_arg_chars is None
    (chk,) = select("policy", pol)
    assert chk.rule(_trace([("lookup", {"doc_id": "doc-0001"})])).detected
    assert not CHECKS["policy"].rule(_trace([("lookup", {"doc_id": "doc-0001"})])).detected  # default unchanged
    p.write_text(json.dumps({"allowlist": ["search"]}))
    with pytest.raises(ValueError):
        Policy.load(p)


# --- business rules: limits, identity, requires (0.3.0) ---

def _run(calls: list[tuple], root_attrs: dict | None = None) -> Trace:
    """calls: (tool, args) or (tool, args, result); args may be a dict or a raw string."""
    root = Span(span_id="r" * 16, parent_id=None, name="invoke_agent", start_ns=0,
                attrs={"gen_ai.operation.name": "invoke_agent", "gen_ai.input.messages": "q",
                       "gen_ai.output.messages": "a", **(root_attrs or {})})
    spans = [root]
    for i, (tool, args, *rest) in enumerate(calls):
        spans.append(Span(span_id=f"{i:016d}", parent_id=root.span_id, name="execute_tool", start_ns=i + 1,
                          attrs={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool,
                                 "gen_ai.tool.call.arguments": args if isinstance(args, str) else json.dumps(args),
                                 "gen_ai.tool.call.result": rest[0] if rest else "{}"}))
    return Trace(trace_id="t" * 32, agent="a", spans=spans)


def _policy(**d) -> Policy:
    return Policy.from_dict(d)


REFUND = _policy(limits=[{"tool": "issue_refund", "arg": "amount", "max": 500, "min": 0}])


@pytest.mark.parametrize("amount, fires", [(750, True), (-1, True), (500, False), (0, False), ("$1,200.50", True),
                                           (" 499.99 ", False), ("abc", True), (True, True)])
def test_limits(amount, fires):
    v = policy_rule(_run([("issue_refund", {"amount": amount})]), REFUND)
    assert v.detected is fires, v.reason
    if amount == 750:
        assert v.reason == "step 1: issue_refund.amount 750 > max 500"
    if amount == "abc":
        assert v.reason == "step 1: issue_refund.amount 'abc' is not a number"


def test_limits_absent_arg_other_tool_and_dotted_path():
    assert not policy_rule(_run([("issue_refund", {"order": "o-1"})]), REFUND).detected
    assert not policy_rule(_run([("lookup", {"amount": 9999})]), REFUND).detected
    nested = _policy(limits=[{"tool": "issue_refund", "arg": "refund.amount", "max": 500}])
    assert policy_rule(_run([("issue_refund", {"refund": {"amount": 501}})]), nested).detected
    assert not policy_rule(_run([("issue_refund", {"refund": {"amount": 5}})]), nested).detected


def test_limits_fail_closed_on_non_object_arguments():
    v = policy_rule(_run([("issue_refund", "amount=900")]), REFUND)
    assert v.detected and "not a JSON object" in v.reason


def test_double_encoded_and_empty_arguments():
    # JSON encoded twice is decoded, not waved through
    assert policy_rule(_run([("issue_refund", json.dumps(json.dumps({"amount": 900})))]), REFUND).detected
    assert not policy_rule(_run([("issue_refund", json.dumps(json.dumps({"amount": 9})))]), REFUND).detected
    # no arguments at all means the argument is absent, not unreadable
    assert not policy_rule(_run([("issue_refund", "")]), REFUND).detected
    assert not policy_rule(_run([("issue_refund", "  ")]), REFUND).detected


IDENT = _policy(identity={"arg": "customer_id", "tools": ["modify_billing_record"]})
AUTH = {"enduser.id": "C-123"}


def test_identity():
    assert not policy_rule(_run([("modify_billing_record", {"customer_id": "C-123"})], AUTH), IDENT).detected
    v = policy_rule(_run([("lookup", {}), ("modify_billing_record", {"customer_id": "C-999"})], AUTH), IDENT)
    assert v.detected and v.reason == "step 2: modify_billing_record.customer_id 'C-999' != session enduser.id 'C-123'"
    v = policy_rule(_run([("modify_billing_record", {"customer_id": "C-123"})]), IDENT)
    assert v.detected and v.reason == "step 1: cannot verify customer_id, root span has no enduser.id"
    # tool filter, and steps without the argument are not checked
    assert not policy_rule(_run([("lookup", {"customer_id": "C-999"})], AUTH), IDENT).detected
    assert not policy_rule(_run([("modify_billing_record", {"plan": "pro"})]), IDENT).detected
    every_tool = _policy(identity={"arg": "customer_id", "session_attr": "app.customer"})
    assert policy_rule(_run([("lookup", {"customer_id": 7})], {"app.customer": "8"}), every_tool).detected
    assert not policy_rule(_run([("lookup", {"customer_id": 7})], {"app.customer": " 7 "}), every_tool).detected


def test_identity_fails_closed_on_unreadable_arguments():
    # a listed tool whose arguments cannot be read is not verified
    v = policy_rule(_run([("modify_billing_record", "customer_id=C-999")], AUTH), IDENT)
    assert v.detected and v.reason == "step 1: modify_billing_record arguments are not a JSON object, cannot check customer_id"
    # unlisted-tools mode: only when the argument name shows up in the unreadable text
    every_tool = _policy(identity={"arg": "customer_id"})
    assert policy_rule(_run([("lookup", "customer_id=C-999")], AUTH), every_tool).detected
    assert not policy_rule(_run([("search", "free text query")], AUTH), every_tool).detected


REQ = _policy(requires=[{"tool": "modify_billing_record", "after": "verify_customer_identity", "result_contains": "VERIFIED"}])


def test_requires():
    ok = [("verify_customer_identity", {}, '{"status": "VERIFIED"}'), ("modify_billing_record", {})]
    assert not policy_rule(_run(ok), REQ).detected
    v = policy_rule(_run([("modify_billing_record", {})]), REQ)
    assert v.detected and v.reason == "step 1: modify_billing_record before verify_customer_identity returned 'VERIFIED'"
    assert policy_rule(_run(ok[::-1]), REQ).detected  # wrong order
    denied = [("verify_customer_identity", {}, '{"status": "DENIED"}'), ("modify_billing_record", {})]
    assert policy_rule(_run(denied), REQ).detected
    bare = _policy(requires=[{"tool": "b", "after": "a"}])
    v = policy_rule(_run([("b", {})]), bare)
    assert v.detected and v.reason == "step 1: b before a"
    assert not policy_rule(_run([("a", {}), ("b", {})]), bare).detected


@pytest.mark.parametrize("bad", [
    {"limits": {"tool": "x"}},
    {"limits": [{"tool": "x", "max": 1}]},
    {"limits": [{"tool": "x", "arg": "a"}]},
    {"limits": [{"tool": "x", "arg": "a", "min": 5, "max": 1}]},
    {"limits": [{"tool": "x", "arg": "a", "max": "500"}]},
    {"limits": [{"tool": "x", "arg": "a", "max": 1, "maxx": 2}]},
    {"limits": [{"tool": "x", "arg": "a", "max": float("nan")}]},
    {"limits": [{"tool": "x", "arg": "a", "max": float("inf")}]},
    {"limits": [{"tool": "x", "arg": "a", "min": float("-inf")}]},
    {"identity": {"tools": ["x"]}},
    {"identity": {"arg": "a", "tools": "x"}},
    {"requires": [{"tool": "x"}]},
    {"requires": [{"tool": "x", "after": "y", "result_contains": 3}]},
])
def test_malformed_new_keys_raise(tmp_path, bad):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="policy: "):
        Policy.load(p)


def test_old_policy_file_unchanged():
    assert Policy.load(GATE / "policy.json") == FAULTLINE_POLICY


SUPPORT_OK = [
    ("lookup_order", {"order_id": "o-7"}, '{"total": 120}'),
    ("verify_customer_identity", {"customer_id": "C-123"}, '{"status": "VERIFIED"}'),
    ("issue_refund", {"customer_id": "C-123", "amount": "$120.00"}),
    ("modify_billing_record", {"customer_id": "C-123", "plan": "basic"}),
]


@pytest.mark.parametrize("step, change, expected", [
    (None, None, None),
    (2, {"amount": 900}, "step 3: issue_refund.amount 900 > max 500"),
    (2, {"customer_id": "C-999"}, "step 3: issue_refund.customer_id 'C-999' != session enduser.id 'C-123'"),
])
def test_support_example_policy_on_a_support_run(step, change, expected):
    """The shipped example policy on a realistic support run: compliant passes, each violation is named."""
    pol = Policy.load(EXAMPLES.parent / "policy_support.json")
    calls = [list(c) for c in SUPPORT_OK]
    if step is not None:
        calls[step][1] = {**calls[step][1], **change}
    v = policy_rule(_run([tuple(c) for c in calls], AUTH), pol)
    assert v.detected is (expected is not None), v.reason
    if expected:
        assert v.reason == expected
    # skipping verification breaks the billing prerequisite
    v = policy_rule(_run([SUPPORT_OK[0], SUPPORT_OK[2], SUPPORT_OK[3]], AUTH), pol)
    assert v.reason == "step 3: modify_billing_record before verify_customer_identity returned 'VERIFIED'"


def test_support_example_policy_no_false_positives_from_new_keys():
    pol = Policy.load(EXAMPLES.parent / "policy_support.json")
    assert pol.limits and pol.identity and pol.requires and pol.allow
    business_only = dataclasses.replace(pol, allow=None, max_calls=None)  # the support tools never appear in these runs
    files = [EXAMPLES, *sorted(GATE.glob("baseline_*.json")), *sorted(GATE.glob("candidate_*.json"))]
    traces = [t for f in files for t in load_otlp(f)]
    assert len(traces) > 100
    assert not [policy_rule(t, business_only).reason for t in traces if policy_rule(t, business_only).detected]


def test_faultline_policy_json_is_picked_up_by_convention(tmp_path, monkeypatch, capsys):
    from faultgate import cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "faultline-policy.json").write_text(json.dumps({"max_calls": 0}))
    assert cli.main(["check", str(EXAMPLES), "--check", "policy"]) == 1
    assert "using ./faultline-policy.json" in capsys.readouterr().err
    assert cli.main(["check", str(EXAMPLES), "--check", "policy", "--policy", str(GATE / "policy.json")]) == 1  # explicit wins
    assert "faultline-policy.json" not in capsys.readouterr().err
    (tmp_path / "faultline-policy.json").unlink()
    assert cli.main(["check", str(EXAMPLES), "--check", "policy"]) == 0
