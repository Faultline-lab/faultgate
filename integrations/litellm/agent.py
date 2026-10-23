"""Support agent with 3 fake tools (LiteLLM recipe). Data lives in a dict."""

ORDERS = {
    "A1": {"order_id": "A1", "status": "pending", "customer_id": "C1", "price": 300},
    "B2": {"order_id": "B2", "status": "delivered", "customer_id": "C2", "price": 900},
}


def lookup_order(order_id):
    """Return the order details, or an error if the order does not exist."""
    return ORDERS.get(order_id, {"error": "order not found"})


def cancel_order(order_id, reason):
    """Pretend to cancel an order."""
    return {"order_id": order_id, "status": "cancelled"}


def issue_refund(order_id, amount):
    """Pretend to refund an amount."""
    return {"order_id": order_id, "refunded": amount}


TOOLS = {
    "lookup_order": lookup_order,
    "cancel_order": cancel_order,
    "issue_refund": issue_refund,
}

import json

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


def _attr(value):
    """Turn a Python value into an OTLP/JSON attribute value."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


class JsonFileExporter(SpanExporter):
    """Collect finished spans, then save them as one OTLP/JSON file."""

    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def save(self, path):
        out = []
        for s in self.spans:
            ctx = s.get_span_context()
            item = {
                "traceId": format(ctx.trace_id, "032x"),
                "spanId": format(ctx.span_id, "016x"),
                "name": s.name,
                "startTimeUnixNano": str(s.start_time),
                "endTimeUnixNano": str(s.end_time),
                "attributes": [
                    {"key": k, "value": _attr(v)} for k, v in s.attributes.items()
                ],
            }
            if s.parent is not None:
                item["parentSpanId"] = format(s.parent.span_id, "016x")
            out.append(item)
        with open(path, "w") as f:
            json.dump({"resourceSpans": [{"scopeSpans": [{"spans": out}]}]}, f, indent=1)