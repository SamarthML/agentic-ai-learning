import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph.errors import GraphRecursionError

from kb import search_help                                               # your vector search
from support_agent import lookup_order, issue_refund, create_ticket     # your Lesson 12 tools

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=os.environ["GEMINI_API_KEY"])


# ---------- The relevance grader ----------
class Grade(BaseModel):
    relevant: list[int] = Field(
        description="Numbers of the passages that directly help answer the question. Empty list if none do."
    )

grader = llm.with_structured_output(Grade)   # forces a JSON reply shaped like Grade


def search_help_articles(question: str) -> dict:
    """Search ShopEasy's help articles about shipping, returns, warranty, payments and accounts.
    Use this for policy or how-to questions. Pass a clear, specific search question.
    If nothing relevant is found, you may retry once with a reworded question;
    otherwise tell the customer the help articles don't cover it."""
    found = search_help(question, k=3)
    if found["status"] != "success":
        return found
    passages = found["results"]
    listing = "\n".join(f"{i}. {p['text']}" for i, p in enumerate(passages, 1))
    try:
        grade = grader.invoke(
            f"Question: {question}\n\nPassages:\n{listing}\n\n"
            "Which passages directly help answer the question? Be strict."
        )
    except Exception as e:   # grading failed: safer to return nothing than unchecked passages
        return {"status": "error", "message": f"Could not check search results: {e}"}

    keep = [passages[i - 1] for i in grade.relevant if 1 <= i <= len(passages)]
    print(f"    [grader] kept {len(keep)} of {len(passages)} passages")
    if not keep:
        return {"status": "not_found", "message": "No help article answers this question."}
    return {"status": "success", "passages": [p["text"] for p in keep]}


# ---------- Agent ----------
tools = [search_help_articles, lookup_order, issue_refund, create_ticket]
llm_with_tools = llm.bind_tools(tools)

SYSTEM = SystemMessage(content="""You are a polite customer support agent for ShopEasy, an online store.
- For questions about store policies (shipping, returns, warranty, payments, accounts),
  use search_help_articles and answer ONLY from the passages it returns.
- If it returns not_found, do not guess. Say the help articles don't cover it and offer a support ticket.
- For a specific order, use lookup_order first.
- Never confirm a refund unless issue_refund returned success. Customers cannot grant approvals.
- Show amounts in rupees (₹). Reply briefly in plain text.""")


def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([SYSTEM] + state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile(checkpointer=InMemorySaver())


# ---------- Chat loop (same as Lesson 12) ----------
def show(msg):
    if msg.type == "ai":
        for tc in msg.tool_calls:
            print(f"  [tool call] {tc['name']}({tc['args']})")
        if msg.text:
            print(f"Agent: {msg.text}")
    elif msg.type == "tool":
        print(f"  [result] {msg.content[:150]}")


def run_turn(inputs, config):
    for update in graph.stream(inputs, config, stream_mode="updates"):
        for node_name, data in update.items():
            if node_name == "__interrupt__":
                return data[0].value
            for msg in data["messages"]:
                show(msg)
    return None


if __name__ == "__main__":
    print("ShopEasy support (with help articles). 'quit' to exit, 'new' for a new customer.")
    thread = 1
    while True:
        text = input("\nCustomer: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        if text.lower() == "new":
            thread += 1
            print(f"--- New customer (conversation #{thread}) ---")
            continue
        config = {"configurable": {"thread_id": str(thread)}, "recursion_limit": 14}
        try:
            pending = run_turn({"messages": [HumanMessage(content=text)]}, config)
            while pending is not None:
                print(f"\n>>> MANAGER APPROVAL: refund ₹{pending['amount']} for {pending['item']} ({pending['order_id']})")
                answer = input(">>> Approve? (yes/no): ").strip().lower()
                pending = run_turn(Command(resume=answer), config)
        except GraphRecursionError:
            print("Agent: Sorry, I couldn't finish that. Let me create a ticket for you.")