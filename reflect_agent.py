from google.genai import types, errors
from agent import client, MODEL, run_agent

CRITIC = """You are a strict reviewer of an AI assistant's answers.
You will see a conversation (including tool calls and tool results) and a DRAFT answer.
Check the draft:
1. Is every fact in it supported by a TOOL RESULT? Flag anything guessed,
   e.g. sky conditions, humidity or forecasts that no tool reported.
2. Does it answer everything the user asked?
If the draft is fully correct, reply with exactly: APPROVED
Otherwise, reply with ONLY the corrected final answer, in plain text."""

critic_config = types.GenerateContentConfig(system_instruction=CRITIC)


def to_transcript(messages: list) -> str:
    """Turn the message history into readable text for the critic."""
    lines = []
    for msg in messages:
        for part in msg.parts:
            if part.thought:          # skip the model's hidden reasoning parts
                continue
            if part.text:
                lines.append(f"{msg.role.upper()}: {part.text}")
            elif part.function_call:
                fc = part.function_call
                lines.append(f"TOOL CALL: {fc.name}({dict(fc.args or {})})")
            elif part.function_response:
                fr = part.function_response
                lines.append(f"TOOL RESULT from {fr.name}: {fr.response}")
    return "\n".join(lines)


def reflect(history: list, draft: str) -> str:
    """Ask the critic to review the draft. Returns the final answer."""
    transcript = to_transcript(history[:-1])   # everything except the draft itself
    prompt = f"CONVERSATION:\n{transcript}\n\nDRAFT ANSWER:\n{draft}"
    try:
        resp = client.models.generate_content(model=MODEL, contents=prompt, config=critic_config)
    except errors.APIError as e:
        print(f"  [review] skipped (API error {e.code})")
        return draft

    verdict = (resp.text or "").strip()
    if not verdict or verdict.upper().startswith("APPROVED"):
        print("  [review] APPROVED")
        return draft

    print("  [review] REVISED")
    # Keep memory consistent: replace the draft with the corrected answer
    history[-1] = types.Content(role="model", parts=[types.Part(text=verdict)])
    return verdict


if __name__ == "__main__":
    print("Reflective agent ready. Type 'quit' to exit.")
    history = []
    while True:
        question = input("\nYou: ")
        if question.strip().lower() in {"quit", "exit"}:
            break
        draft = run_agent(question, history)
        # Only review real answers, not error or step-limit messages
        if history and history[-1].role == "model":
            print(f"  [draft] {draft}")
            answer = reflect(history, draft)
        else:
            answer = draft
        print("Agent:", answer)