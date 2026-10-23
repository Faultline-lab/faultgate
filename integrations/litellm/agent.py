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