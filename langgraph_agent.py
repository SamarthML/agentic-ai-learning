import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError

from tools import get_weather, save_note   # your own tools, unchanged!

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=os.environ["GEMINI_API_KEY"],
)
tools = [get_weather, save_note]
llm_with_tools = llm.bind_tools(tools)   # like tools=[...] in your config

SYSTEM = SystemMessage(content="""You are a helpful assistant that can use tools.
- Use tools for facts like weather. Never guess data a tool can provide.
- If a tool returns an error, fix your input and retry, or explain the problem.
- Reply in plain text. Do not use LaTeX.""")


# NODE 1: the model thinks (your generate_content call)
def assistant(state: MessagesState):
    response = llm_with_tools.invoke([SYSTEM] + state["messages"])
    return {"messages": [response]}   # appended to the state automatically


# BUILD THE GRAPH
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))                    # NODE 2: runs tools
builder.add_edge(START, "assistant")                          # start by thinking
builder.add_conditional_edges("assistant", tools_condition)   # tools, or END?
builder.add_edge("tools", "assistant")                        # results go back to the model
graph = builder.compile()


if __name__ == "__main__":
    question = input("You: ")
    try:
        for update in graph.stream(
            {"messages": [HumanMessage(content=question)]},
            {"recursion_limit": 10},
            stream_mode="updates",
        ):
            for node_name, data in update.items():
                print(f"\n--- node: {node_name} ---")
                for msg in data["messages"]:
                    msg.pretty_print()
    except GraphRecursionError:
        print("\nStopped: the agent took too many steps without finishing.")