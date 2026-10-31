import os
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors   # NEW: errors

from tools import get_weather, save_note

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash-lite"

TOOLS = {"get_weather": get_weather, "save_note": save_note}

SYSTEM = """You are a helpful assistant that can use tools.
- Use tools for facts like weather. Never guess data a tool can provide.
- If a tool returns an error, read the message and either fix your input and retry, or explain the problem to the user.
- Reply in plain text. Do not use LaTeX."""

config = types.GenerateContentConfig(
    system_instruction=SYSTEM,
    tools=list(TOOLS.values()),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def run_tool(call) -> dict:
    """Run one tool call safely. Never raises."""
    func = TOOLS.get(call.name)
    if func is None:
        return {"status": "error", "message": f"Unknown tool: {call.name}"}
    try:
        return func(**(call.args or {}))
    except TypeError as e:   # model sent wrong or missing arguments
        return {"status": "error", "message": f"Bad arguments: {e}"}

TRIM_NOTE = ("(Note: earlier parts of this conversation were removed to save space. "
             "If asked about them, say you no longer have that information.)")


def trim_history(history: list, max_turns: int = 10) -> None:
    """Keep only the last `max_turns` questions and everything after them."""
    starts = [i for i, msg in enumerate(history)
              if msg.role == "user" and msg.parts[0].text]
    if len(starts) > max_turns:
        del history[:starts[-max_turns]]
        # Tell the model something was removed, by adding the note
        # to the start of the first remaining question
        first_parts = history[0].parts
        if first_parts[0].text != TRIM_NOTE:
            first_parts.insert(0, types.Part(text=TRIM_NOTE))

def run_agent(question: str, history: list, max_steps: int = 5) -> str:   # NEW: history parameter
    # NEW: add the question to the SHARED history instead of making a new list
    history.append(types.Content(role="user", parts=[types.Part(text=question)]))
    trim_history(history, max_turns=10)
    for step in range(1, max_steps + 1):
        # THINK: the model decides what to do next
        try:   # NEW: a failed model call no longer crashes the program
            response = client.models.generate_content(model=MODEL, contents=history, config=config)
        except errors.APIError as e:
            return f"Model call failed (code {e.code}): {e.message}"

        # DONE: no tool calls means this is the final answer
        if not response.function_calls:
            history.append(response.candidates[0].content)   # NEW: remember its own answer
            return response.text

        history.append(response.candidates[0].content)

        # ACT + OBSERVE: run every tool it asked for, collect the results
        result_parts = []
        for call in response.function_calls:
            result = run_tool(call)
            print(f"  [step {step}] {call.name}({call.args}) -> {result}")
            result_parts.append(types.Part.from_function_response(name=call.name, response=result))
        history.append(types.Content(role="user", parts=result_parts))
        # ...and loop back so the model can THINK again with the new results

    return "Stopped: reached the step limit without a final answer."


if __name__ == "__main__":
    print("Agent ready. Type 'quit' to exit, 'reset' to clear memory.")
    history = []   # NEW: created ONCE, survives between questions
    while True:
        question = input("\nYou: ")
        if question.strip().lower() in {"quit", "exit"}:
            break
        if question.strip().lower() == "reset":   # NEW
            history.clear()
            print("Memory cleared.")
            continue
        print("Agent:", run_agent(question, history))
        print(f"  (history now has {len(history)} messages)")   # NEW