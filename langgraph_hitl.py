import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langgraph.errors import GraphRecursionError

from tools import get_weather, save_note as raw_save_note

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=os.environ["GEMINI_API_KEY"])


def save_note(text: str) -> dict:
    """Append a note to notes.txt.

    Use this when the user asks to remember, save, or write down something.
    The user must approve each note before it is saved.
    """
    # PAUSE the whole graph here and wait for a human answer
    answer = interrupt({"action": "save_note", "text": text})
    if answer == "yes":
        return raw_save_note(text)
    return {"status": "error", "message": "The user did not approve saving this note."}


tools = [get_weather, save_note]
llm_with_tools = llm.bind_tools(tools)

SYSTEM = SystemMessage(content="""You are a helpful assistant that can use tools.
- Use tools for facts like weather. Never guess data a tool can provide.
- If a tool returns an error, explain the problem to the user.
- Reply in plain text. Do not use LaTeX.""")


def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([SYSTEM] + state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile(checkpointer=InMemorySaver())   # NEW: memory


def run_turn(inputs, config):
    """Stream one run. Returns the interrupt request if the graph paused, else None."""
    for update in graph.stream(inputs, config, stream_mode="updates"):
        for node_name, data in update.items():
            if node_name == "__interrupt__":       # the graph paused for a human
                return data[0].value
            print(f"\n--- node: {node_name} ---")
            for msg in data["messages"]:
                msg.pretty_print()
    return None


if __name__ == "__main__":
    print("Type 'quit' to exit, 'new' to start a new conversation.")
    thread = 1
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if question.lower() == "new":
            thread += 1
            print(f"Started conversation #{thread}")
            continue

        config = {"configurable": {"thread_id": str(thread)}, "recursion_limit": 10}
        try:
            pending = run_turn({"messages": [HumanMessage(content=question)]}, config)
            while pending is not None:   # keep asking until nothing is waiting
                print(f"\n>>> APPROVAL NEEDED: save note: {pending['text']!r}")
                answer = input(">>> Approve? (yes/no): ").strip().lower()
                pending = run_turn(Command(resume=answer), config)
        except GraphRecursionError:
            print("\nStopped: the agent took too many steps without finishing.")