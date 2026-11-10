import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langgraph.errors import GraphRecursionError

load_dotenv()

# ---------- Fake company data (a real app would use a database) ----------
ORDERS = {
    "ORD1001": {"item": "Wireless earbuds", "amount": 2499, "delivery_status": "delivered", "days_since_delivery": 3, "refunded": False},
    "ORD1002": {"item": "Smartwatch", "amount": 8999, "delivery_status": "delivered", "days_since_delivery": 5, "refunded": False},
    "ORD1003": {"item": "Laptop bag", "amount": 1299, "delivery_status": "delivered", "days_since_delivery": 20, "refunded": False},
    "ORD1004": {"item": "Smartphone", "amount": 24999, "delivery_status": "shipped", "days_since_delivery": None, "refunded": False},
}
TICKETS = []
APPROVAL_LIMIT = 5000      # rupees
REFUND_WINDOW_DAYS = 7


# ---------- Tools ----------
def lookup_order(order_id: str) -> dict:
    """Look up an order by its ID, e.g. 'ORD1001'.
    Returns the item, amount in rupees, delivery status and days since delivery.
    Use this before discussing any specific order."""
    oid = order_id.strip().upper()
    order = ORDERS.get(oid)
    if order is None:
        return {"status": "error", "message": f"No order found with ID {oid}."}
    return {"status": "success", "order_id": oid, **order}


def get_refund_policy() -> dict:
    """Return the shop's refund policy. Use this when the customer asks about refunds or returns."""
    return {"status": "success", "policy": [
        f"Refunds are allowed within {REFUND_WINDOW_DAYS} days of delivery.",
        "Orders not yet delivered cannot be refunded; the customer can cancel instead.",
        f"Refunds above ₹{APPROVAL_LIMIT} need a manager's approval.",
        "Each order can be refunded only once.",
    ]}


def issue_refund(order_id: str, reason: str) -> dict:
    """Issue a full refund for an order. Check the order with lookup_order first.
    The refund rules are enforced automatically; if a refund is not allowed,
    the result explains why."""
    oid = order_id.strip().upper()
    order = ORDERS.get(oid)
    # Rule checks: these only READ data, so re-running them on resume is safe
    if order is None:
        return {"status": "error", "message": f"No order found with ID {oid}."}
    if order["refunded"]:
        return {"status": "error", "message": "This order has already been refunded."}
    if order["delivery_status"] != "delivered":
        return {"status": "error", "message": "This order is not delivered yet, so it can't be refunded. It can be cancelled instead."}
    if order["days_since_delivery"] > REFUND_WINDOW_DAYS:
        return {"status": "error", "message": f"Delivered {order['days_since_delivery']} days ago, outside the {REFUND_WINDOW_DAYS}-day refund window."}

    # Big refunds wait for a manager
    if order["amount"] > APPROVAL_LIMIT:
        answer = interrupt({"order_id": oid, "item": order["item"],
                            "amount": order["amount"], "reason": reason})
        if answer != "yes":
            return {"status": "error", "message": "Not refunded: the manager declined. Offer the customer a support ticket."}

    # Side effect comes LAST, after every check and the approval
    order["refunded"] = True
    return {"status": "success", "message": f"Refunded ₹{order['amount']} for {order['item']} ({oid})."}


def create_ticket(summary: str, order_id: str = "") -> dict:
    """Create a support ticket for a human agent to follow up.
    Use this when you can't solve the problem yourself or the customer asks for one."""
    ticket_id = f"TCK{len(TICKETS) + 1:03d}"
    TICKETS.append({"id": ticket_id, "order_id": order_id, "summary": summary})
    return {"status": "success", "ticket_id": ticket_id}


tools = [lookup_order, get_refund_policy, issue_refund, create_ticket]

# ---------- Model and graph ----------
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=os.environ["GEMINI_API_KEY"])
llm_with_tools = llm.bind_tools(tools)

SYSTEM = SystemMessage(content="""You are a polite customer support agent for ShopEasy, an online store.
- Always look up an order with lookup_order before discussing it.
- Never promise or confirm a refund unless issue_refund returned success.
- If a refund is refused, explain the reason kindly and offer to create a support ticket.
- Show amounts in rupees (₹). Reply in plain text, briefly.""")


def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([SYSTEM] + state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile(checkpointer=InMemorySaver())


# ---------- Chat loop ----------
def show(msg):
    """Print a message compactly."""
    if msg.type == "ai":
        for tc in msg.tool_calls:
            print(f"  [tool call] {tc['name']}({tc['args']})")
        if msg.text:
            print(f"Agent: {msg.text}")
    elif msg.type == "tool":
        print(f"  [result] {msg.content}")


def run_turn(inputs, config):
    for update in graph.stream(inputs, config, stream_mode="updates"):
        for node_name, data in update.items():
            if node_name == "__interrupt__":
                return data[0].value
            for msg in data["messages"]:
                show(msg)
    return None


if __name__ == "__main__":
    print("ShopEasy support. Type 'quit' to exit, 'new' for a new customer.")
    thread = 1
    while True:
        text = input("\nCustomer: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        if text.lower() == "new":
            thread += 1
            print(f"--- New customer (conversation #{thread}) ---")
            continue
        config = {"configurable": {"thread_id": str(thread)}, "recursion_limit": 12}
        try:
            pending = run_turn({"messages": [HumanMessage(content=text)]}, config)
            while pending is not None:
                print(f"\n>>> MANAGER APPROVAL: refund ₹{pending['amount']} for "
                      f"{pending['item']} ({pending['order_id']}). Reason: {pending['reason']}")
                answer = input(">>> Approve? (yes/no): ").strip().lower()
                pending = run_turn(Command(resume=answer), config)
        except GraphRecursionError:
            print("Agent: Sorry, I couldn't finish that. Let me create a ticket for you.")