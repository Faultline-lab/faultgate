import json

import pytest

from faultgate import cli
from faultgate.checks import Policy, select
from faultgate.judge import Verdict
from faultgate.redact import builtin, get_redactor
from faultgate.runner import run
from faultgate.trace import load_otlp
from tests.conftest import EXAMPLES

R = builtin()


@pytest.mark.parametrize("text, kind", [
    ("mail jane.doe+x@corp.example.com now", "EMAIL"),
    ("card 4111 1111 1111 1111 on file", "CREDIT_CARD"),
    ("card 4111-1111-1111-1111", "CREDIT_CARD"),
    ("ssn 123-45-6789", "US_SSN"),
    ("call +1 (415) 555-0132 today", "PHONE"),
    ("call 020 7946 0958", "PHONE"),
    ("from 192.168.10.24", "IP_ADDRESS"),
    ("key sk-proj-abcdefghijklmnopqrstuvwx", "SECRET"),
    ("aws AKIAIOSFODNN7EXAMPLE", "SECRET"),
    ("gh ghp_" + "a" * 36, "SECRET"),
    ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345", "SECRET"),
    ("token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U", "SECRET"),
])
def test_builtin_masks_each_kind(text, kind):
    out = R(text)
    assert f"<{kind}>" in out, out


@pytest.mark.parametrize("text", [
    "order o-7 refunded $1,200.50 on 2026-10-01 at 10:00:01",
    "card 4111 1111 1111 1112",           # fails Luhn
    "customer C-123, step 12 of 30, version 1.2.3",
    "span 1000000000000001 trace 0f0f0f0f0f0f0f0f",
    "r-0336 scored 0.87 over 30 scenarios",
])
def test_builtin_leaves_ordinary_text_alone(text):
    assert R(text) == text


def test_shipped_reasons_and_prompts_have_nothing_to_mask():
    """Redaction must not damage the shipped real traces' report text."""
    traces = load_otlp(EXAMPLES) + load_otlp(EXAMPLES.parent / "gate" / "candidate_gpt.json")
    report = run(traces, select(None), None)
    assert all(R(r.verdict.reason) == r.verdict.reason for r in report.results)
    assert all(R(t.prompt) == t.prompt for t in traces)


class RecordingJudge:
    def __init__(self, remote):
        self.name, self.budget, self.remote, self.states = "rec", None, remote, []

    def ask(self, state, question):
        self.states.append(state)
        return Verdict(True, 0.9, "user wrote from jane@corp.example.com")


def _email_trace(tmp_path):
    doc = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    span = doc["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    span["attributes"].append({"key": "gen_ai.input.messages", "value": {"stringValue": json.dumps(
        [{"role": "user", "content": "I am jane@corp.example.com, card 4111 1111 1111 1111"}])}})
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc))
    return load_otlp(p)[:1]


def test_remote_judge_sees_masked_trace_local_judge_does_not(tmp_path):
    traces = _email_trace(tmp_path)
    remote, local = RecordingJudge(True), RecordingJudge(False)
    rep = run(traces, select("wrong_direction"), remote, redact=R)
    run(traces, select("wrong_direction"), local, redact=R)
    assert "jane@corp" not in remote.states[0] and "<EMAIL>" in remote.states[0] and "<CREDIT_CARD>" in remote.states[0]
    assert "jane@corp" in local.states[0]
    assert rep.results[0].verdict.reason == "user wrote from <EMAIL>"  # reasons are masked for every judge
    assert rep.redaction == "builtin"


def test_rules_see_raw_values_reasons_are_masked():
    from tests.test_policy import _run

    pol = Policy.from_dict({"identity": {"arg": "email"}})
    t = _run([("update_profile", {"email": "eve@evil.example"})], {"enduser.id": "jane@corp.example.com"})
    rep = run([t], [c for c in select("policy", pol)], None, redact=R)
    v = rep.results[0].verdict
    assert v.detected  # the comparison used the real values
    assert "eve@evil" not in v.reason and v.reason.count("<EMAIL>") == 2


def test_cli_redact_flag(tmp_path, capsys):
    """An identity mismatch puts both values in the reason; the written report must not carry them."""
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"identity": {"arg": "q"}}))
    doc = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    spans = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
    for s in spans:
        if not s.get("parentSpanId"):
            s["attributes"].append({"key": "enduser.id", "value": {"stringValue": "jane@corp.example.com"}})
        for a in s["attributes"]:
            if a["key"] == "gen_ai.tool.call.arguments":
                a["value"] = {"stringValue": '{"q": "eve@evil.example"}'}
    tr = tmp_path / "t.json"
    tr.write_text(json.dumps(doc))
    out = tmp_path / "r.json"
    assert cli.main(["check", str(tr), "--policy", str(pol), "--check", "policy", "--json", str(out)]) == 1
    text = out.read_text()
    assert json.loads(text)["redaction"] == "builtin" and "eve@evil" not in text and "<EMAIL>" in text
    assert "eve@evil" not in capsys.readouterr().out
    cli.main(["check", str(tr), "--policy", str(pol), "--check", "policy", "--json", str(out), "--redact", "off"])
    assert json.loads(out.read_text())["redaction"] == "off" and "eve@evil" in out.read_text()


def test_get_redactor_specs():
    assert get_redactor("off") is None and get_redactor("builtin").name == "builtin"
    with pytest.raises(ValueError, match="unknown redaction"):
        get_redactor("regex")
    try:
        import presidio_analyzer  # noqa: F401
    except ImportError:
        with pytest.raises(ValueError, match=r"faultgate\[pii\]"):
            get_redactor("presidio")


# --- review round 1 (Codex) ---

# Built at runtime from prefix + filler: real-looking token literals in source trip secret scanners.
def _tok(prefix: str, n: int, alphabet: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") -> str:
    return prefix + (alphabet * (n // len(alphabet) + 1))[:n]


@pytest.mark.parametrize("token", [
    _tok("xapp-1-A0123456789-", 40),                     # Slack app token
    _tok("ya29" + ".", 36),                              # Google OAuth access token
    _tok("SG" + ".", 22) + "." + _tok("", 43),           # SendGrid
    _tok("sk" + "_live_", 24),                           # Stripe
    _tok("hf" + "_", 34),                                # Hugging Face
    _tok("sk-ant-api03-", 40),                           # Anthropic
    _tok("xoxb-123456789012-", 30),                      # Slack bot token
], ids=["slack-app", "google-oauth", "sendgrid", "stripe", "huggingface", "anthropic", "slack-bot"])
def test_more_token_families(token):
    assert R(f"key={token} end") == "key=<SECRET> end"


@pytest.mark.parametrize("hostile", ["1-" * 50000 + "x", "a." * 50000, "1 " * 50000, "a@" * 30000, "9" * 100000,
                                     "aa:" * 40000, "(" + "1 " * 40000, "Bearer " + "a" * 100000],
                         ids=["dash-digits", "dots", "spaced-digits", "at-signs", "digits", "colons", "paren", "bearer"])
def test_no_catastrophic_backtracking(hostile):
    import time

    t0 = time.perf_counter()
    R(hostile)
    assert time.perf_counter() - t0 < 1.0


def test_scenario_keys_are_masked_but_stay_unique_and_stable():
    """Explicit scenario keys can carry PII; with redaction they are masked plus a short hash, so matching still works."""
    from faultgate import band as bandmod
    from faultgate.trace import Span, Trace

    def trace(tid, scenario):
        root = Span(span_id="r", parent_id=None, name="invoke_agent", start_ns=1,
                    attrs={"gen_ai.operation.name": "invoke_agent", "faultgate.scenario": scenario})
        return Trace(trace_id=tid, agent="a", spans=[root])

    keys = ["refund for jane@corp.example.com", "refund for bob@corp.example.com", "r-0336"]
    runs = [run([trace(f"{i}{k}", k) for k in keys], select("abstention"), None, redact=R) for i in range(3)]
    shown = [r.scenario for r in runs[0].results]
    assert all("@corp" not in s for s in shown) and len(set(shown)) == 3 and "r-0336" in shown
    assert shown == [r.scenario for r in runs[1].results]  # stable across runs
    b = bandmod.build(runs)
    assert b["n"] == 3 and not any("@corp" in s for s in b["scenarios"])
    assert bandmod.evaluate(runs[2], b)["missing"] == []
    raw = run([trace("x", keys[0])], select("abstention"), None)
    assert raw.results[0].scenario == keys[0]  # redaction off: untouched
