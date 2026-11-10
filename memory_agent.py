import os, re, sqlite3, uuid
from datetime import date
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from rag_agent import search_help_articles
from support_agent import lookup_order, create_ticket

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=os.environ["GEMINI_API_KEY"])

# ---------- Long-term memory store (a SQLite file) ----------
DB = "memory.db"
with sqlite3.connect(DB) as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS facts (customer_id TEXT, fact TEXT, saved_on TEXT)")


def load_facts(customer_id: str) -> list[str]:
    with sqlite3.connect(DB) as conn:
        rows = conn.execute("SELECT fact FROM facts WHERE customer_id = ?", (customer_id,)).fetchall()
    return [r[0] for r in rows]


@tool
def remember_fact(fact: str, config: RunnableConfig) -> dict:
    """Save a lasting fact or preference the customer told you, e.g.
    'prefers express delivery' or 'lives in Pune'.
    Use it only for things that will still matter in future conversations,
    not one-off questions. Never save payment details, passwords or ID numbers."""
    customer_id = config["configurable"]["customer_id"]   # from the session, NOT the model
    fact = fact.strip()
    if re.search(r"\d{8,}", fact):                         # enforced in code
        return {"status": "error", "message": "Not saved: facts must not contain long numbers such as card or ID numbers."}
    if fact in load_facts(customer_id):
        return {"status": "success", "message": "Already remembered."}
    with sqlite3.connect(DB) as conn:
        conn.execute("INSERT INTO facts VALUES (?, ?, ?)", (customer_id, fact, date.today().isoformat()))
    return {"status": "success", "saved": fact}


# ---------- Agent ----------
tools = [search_help_articles, lookup_order, create_ticket, remember_fact]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_TEXT = """You are a polite customer support agent for ShopEasy, an online store.
- For store policy questions, use search_help_articles and answer ONLY from its passages.
  If it returns not_found, say the help articles don't cover it.
- For a specific order, use lookup_order.
- When the customer tells you a lasting preference or fact about themselves, save it with remember_fact.
- Use known facts about the customer when they are relevant to the question.
- Reply briefly in plain text. Show amounts in rupees (₹)."""


def assistant(state: MessagesState, config: RunnableConfig):
    facts = load_facts(config["configurable"]["customer_id"])
    if facts:
        memory_note = "Known facts about this customer from earlier conversations:\n" + "\n".join(f"- {f}" for f in facts)
    else:
        memory_note = "No saved facts about this customer yet."
    system = SystemMessage(content=SYSTEM_TEXT + "\n\n" + memory_note)
    return {"messages": [llm_with_tools.invoke([system] + state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile(checkpointer=InMemorySaver())


def show(msg):
    if msg.type == "ai":
        for tc in msg.tool_calls:
            print(f"  [tool call] {tc['name']}({tc['args']})")
        if msg.text:
            print(f"Agent: {msg.text}")
    elif msg.type == "tool":
        print(f"  [result] {msg.content[:150]}")


if __name__ == "__main__":
    customer_id = input("Customer ID (e.g. C001): ").strip().upper() or "C001"
    print(f"Saved facts for {customer_id}: {load_facts(customer_id) or 'none'}")
    config = {
        "configurable": {
            "thread_id": f"{customer_id}-{uuid.uuid4().hex[:8]}",   # a new conversation every run
            "customer_id": customer_id,
        },
        "recursion_limit": 12,
    }
    while True:
        text = input("\nCustomer: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        try:
            for update in graph.stream({"messages": [HumanMessage(content=text)]}, config, stream_mode="updates"):
                for data in update.values():
                    for msg in data["messages"]:
                        show(msg)
        except GraphRecursionError:
            print("Agent: Sorry, I couldn't finish that.")