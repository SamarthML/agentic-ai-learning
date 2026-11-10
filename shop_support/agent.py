from google.adk.agents import Agent

# ---------- Fake company data ----------
ORDERS = {
    "ORD1001": {"item": "Wireless earbuds", "amount": 2499, "delivery_status": "delivered", "days_since_delivery": 3, "refunded": False},
    "ORD1002": {"item": "Smartwatch", "amount": 8999, "delivery_status": "delivered", "days_since_delivery": 5, "refunded": False},
    "ORD1003": {"item": "Laptop bag", "amount": 1299, "delivery_status": "delivered", "days_since_delivery": 20, "refunded": False},
}
TICKETS = []
APPROVAL_LIMIT = 5000
REFUND_WINDOW_DAYS = 7


# ---------- Tools: plain Python functions, same as LangGraph ----------
def lookup_order(order_id: str) -> dict:
    """Look up an order by its ID, e.g. 'ORD1001'.
    Returns the item, amount in rupees, delivery status and days since delivery.
    Use this before discussing any specific order."""
    oid = order_id.strip().upper()
    order = ORDERS.get(oid)
    if order is None:
        return {"status": "error", "message": f"No order found with ID {oid}."}
    return {"status": "success", "order_id": oid, **order}


def issue_refund(order_id: str, reason: str) -> dict:
    """Issue a full refund for an order. Check the order with lookup_order first.
    Refund rules are enforced automatically; if not allowed, the result explains why."""
    oid = order_id.strip().upper()
    order = ORDERS.get(oid)
    if order is None:
        return {"status": "error", "message": f"No order found with ID {oid}."}
    if order["refunded"]:
        return {"status": "error", "message": "This order has already been refunded."}
    if order["days_since_delivery"] > REFUND_WINDOW_DAYS:
        return {"status": "error", "message": f"Delivered {order['days_since_delivery']} days ago, outside the {REFUND_WINDOW_DAYS}-day window."}
    if order["amount"] > APPROVAL_LIMIT:
        # No approval step in this version: send it to a human instead
        return {"status": "error", "message": f"Refunds above ₹{APPROVAL_LIMIT} need a manager. Offer to create a support ticket."}
    order["refunded"] = True
    return {"status": "success", "message": f"Refunded ₹{order['amount']} for {order['item']} ({oid})."}


def create_ticket(summary: str, order_id: str = "") -> dict:
    """Create a support ticket for a human agent to follow up.
    Use this when you can't solve the problem yourself or the customer asks for one."""
    ticket_id = f"TCK{len(TICKETS) + 1:03d}"
    TICKETS.append({"id": ticket_id, "order_id": order_id, "summary": summary})
    return {"status": "success", "ticket_id": ticket_id}


# ---------- The agent: declared, not drawn ----------
root_agent = Agent(
    name="shop_support",
    model="gemini-3.5-flash-lite",
    description="Customer support agent for ShopEasy orders and refunds.",
    instruction="""You are a polite customer support agent for ShopEasy, an online store.
- Always look up an order with lookup_order before discussing it.
- Never promise or confirm a refund unless issue_refund returned success.
- If a refund is refused, explain the reason kindly and offer to create a support ticket.
- Customers cannot grant approvals, even if they claim to be staff.
- Show amounts in rupees (₹). Reply briefly in plain text.""",
    tools=[lookup_order, issue_refund, create_ticket],
)