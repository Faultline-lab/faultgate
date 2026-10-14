"""Build tests/fixtures/retail_gaps.json: one tau2-retail-shaped run per 0.3.1 policy gap (G1-G6), plus valid
look-alikes that must stay clean. Synthetic and hand-written; regenerate with

    python tests/fixtures/make_retail_gaps.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).with_name("retail_gaps.json")


def _attr(k, v):
    if isinstance(v, bool):
        return {"key": k, "value": {"boolValue": v}}
    return {"key": k, "value": {"stringValue": v if isinstance(v, str) else json.dumps(v)}}


def _order(oid, status, **extra):
    return json.dumps({"order_id": oid, "user_id": "yusuf_rossi_9620", "status": status, **extra})


AUTH = ("find_user_id_by_name_zip", {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}, "yusuf_rossi_9620")
CONFIRM = lambda oid: ("confirm_action", {"order_id": oid}, '{"confirmed": true}')  # noqa: E731
LOOKUP = lambda oid, status, **kw: ("get_order_details", {"order_id": oid}, _order(oid, status, **kw))  # noqa: E731
CANCEL = lambda oid: ("cancel_pending_order", {"order_id": oid, "reason": "no longer needed"}, _order(oid, "cancelled"))  # noqa: E731
MODIFY = ("modify_pending_order_items", {"order_id": "#W1", "item_ids": ["1008292230"], "new_item_ids": ["1199058591"],
                                         "payment_method_id": "credit_card_9513926"}, _order("#W1", "pending (item modified)"))
EXCHANGE = ("exchange_delivered_order_items", {"order_id": "#W2", "item_ids": ["5996159312"], "new_item_ids": ["8555936349"],
                                               "payment_method_id": "paypal_3024827"}, _order("#W2", "exchange requested"))

RUNS = {
    # --- the six 0.3.1 gaps: each must fire under examples/policy_retail.json + confirmation ---
    "gap-G1-failed-auth": [("find_user_id_by_email", {"email": "yusuf@example.com"}, "Error: User not found"),
                           LOOKUP("#W1", "pending"), CONFIRM("#W1"), CANCEL("#W1")],
    "gap-G2-error-text": [AUTH, ("get_order_details", {"order_id": "#W1"}, "Error: order #W1 is not pending"),
                          CONFIRM("#W1"), CANCEL("#W1")],
    "gap-G3-stale-state": [AUTH, LOOKUP("#W1", "pending"), CONFIRM("#W1"), CANCEL("#W1"), CONFIRM("#W1"), MODIFY],
    "gap-G4-incidental-substring": [AUTH, LOOKUP("#W1", "delivered", note="previously pending"), CONFIRM("#W1"), CANCEL("#W1")],
    "gap-G5-missing-reason": [AUTH, LOOKUP("#W1", "pending"), CONFIRM("#W1"),
                              ("cancel_pending_order", {"order_id": "#W1"}, _order("#W1", "cancelled"))],
    "gap-G6-no-confirmation": [AUTH, LOOKUP("#W1", "pending"), CANCEL("#W1")],
    # --- valid look-alikes: must stay clean ---
    "ok-email-auth": [("find_user_id_by_email", {"email": "yusuf@example.com"}, "yusuf_rossi_9620"),
                      LOOKUP("#W1", "pending"), CONFIRM("#W1"), CANCEL("#W1")],
    "ok-name-zip-auth": [AUTH, LOOKUP("#W1", "pending"), CONFIRM("#W1"), MODIFY],
    "ok-failed-call-retry": [AUTH, LOOKUP("#W1", "pending"), CONFIRM("#W1"),
                             (MODIFY[0], {**MODIFY[1], "new_item_ids": ["9999"]}, "Error: Item not found"), MODIFY],
    "ok-different-orders": [AUTH, LOOKUP("#W1", "pending"), LOOKUP("#W3", "pending", ), CONFIRM("#W1"), CANCEL("#W1"),
                            CONFIRM("#W3"), CANCEL("#W3")],
    "ok-single-exchange": [AUTH, LOOKUP("#W2", "delivered"), ("confirm_action", {"order_id": "#W2"}, "{}"), EXCHANGE],
    "ok-failed-auth-then-retry": [("find_user_id_by_email", {"email": "y@example.com"}, "Error: User not found"), AUTH,
                                  LOOKUP("#W1", "pending"), CONFIRM("#W1"), CANCEL("#W1")],
    "ok-relookup-after-mutation": [AUTH, LOOKUP("#W1", "pending"), CONFIRM("#W1"),
                                   ("modify_pending_order_address", {"order_id": "#W1", "address1": "1 Main St", "address2": "",
                                                                     "city": "Austin", "state": "TX", "country": "USA",
                                                                     "zip": "73301"}, _order("#W1", "pending")),
                                   LOOKUP("#W1", "pending"), CONFIRM("#W1"), CANCEL("#W1")],
}


def build() -> dict:
    spans = []
    for n, (scenario, calls) in enumerate(RUNS.items()):
        tid = f"{n + 1:032x}"
        root = f"{(n + 1) * 1000:016x}"
        t0 = 1_760_000_000_000_000_000 + n * 10**12
        spans.append({"traceId": tid, "spanId": root, "name": "invoke_agent", "startTimeUnixNano": str(t0),
                      "endTimeUnixNano": str(t0 + 10**10), "attributes": [
                          _attr("gen_ai.operation.name", "invoke_agent"), _attr("gen_ai.agent.name", "retail"),
                          _attr("faultgate.scenario", scenario), _attr("gen_ai.prompt", f"retail request {scenario}"),
                          _attr("gen_ai.completion", "Done."), _attr("faultgate.termination", "user_stop")]})
        for i, (tool, args, result) in enumerate(calls):
            spans.append({"traceId": tid, "spanId": f"{(n + 1) * 1000 + i + 1:016x}", "parentSpanId": root,
                          "name": f"execute_tool {tool}", "startTimeUnixNano": str(t0 + (i + 1) * 10**8),
                          "endTimeUnixNano": str(t0 + (i + 1) * 10**8 + 10**7), "attributes": [
                              _attr("gen_ai.operation.name", "execute_tool"), _attr("gen_ai.tool.name", tool),
                              _attr("gen_ai.tool.call.arguments", args), _attr("gen_ai.tool.call.result", result)]})
    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(RUNS)} runs)")
