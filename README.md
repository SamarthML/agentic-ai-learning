# Agentic AI Learning

Hands-on journey into **agentic AI**: building AI agents from scratch in Python before moving to frameworks like LangGraph and Google ADK.

Everything here is written without an agent framework, so every part of the agent loop is visible and understood: tool design, tool calling, the ReAct loop, error handling and conversation memory.

**Stack:** Python 3.12 · Google Gemini API (`google-genai`) · Pydantic · Requests · Open-Meteo API

---

## What the agent can do

A terminal chat agent that decides on its own which tools to call, in what order, and when it is done.

```
You: Which is hotter right now, Pune or Delhi? Save the answer as a note.
  [step 1] get_weather({'city': 'Pune'})  -> 33.1°C
  [step 1] get_weather({'city': 'Delhi'}) -> 29.4°C
  [step 2] save_note({'text': 'Pune is hotter right now with 33.1°C compared to Delhi at 29.4°C.'})
Agent: Pune is hotter right now with a temperature of 33.1°C compared to Delhi at 29.4°C.
       I have saved this answer as a note for you.
```

Here the agent ran two weather lookups **in parallel** (they don't depend on each other), then saved the note **in a later step**, because it needed both results first.

---

## Features

- **Custom tools** (`tools.py`)
  - `get_weather(city)`: live weather via the free Open-Meteo API (geocoding + forecast)
  - `save_note(text)`: appends notes to a local file
  - Inputs validated with **Pydantic**; tools **never raise**, they return `{"status": "error", ...}` so the agent can recover
  - Network calls use timeouts and handle missing response fields safely
- **ReAct agent loop** (`agent.py`): think → act → observe → repeat, written by hand
  - Parallel and sequential tool calls
  - `max_steps` limit so a confused model can't loop forever or burn API quota
  - Graceful handling of unknown tools, bad arguments and API errors (quota limits, retired models)
  - System instruction to stop the model guessing data a tool can provide
- **Conversation memory**
  - Shared message history across turns, so follow-ups like "save that as a note" work
  - History trimming by **whole turns** (never splits a tool request from its result)
  - Tells the model when earlier context was removed, so it answers honestly instead of confidently wrong
  - `reset` command to clear memory

---

## Files

| File | What it shows |
| --- | --- |
| `tools.py` | The agent's tools: Pydantic validation, API calls, error dicts |
| `pydantic_basics.py` | Pydantic models, validation errors and the JSON schema an LLM reads |
| `api_basics.py` | Calling a web API safely with `requests` |
| `llm_basics.py` | First Gemini API call |
| `first_tool_call.py` | The model choosing a tool and its arguments (without running it) |
| `tool_round_trip.py` | One full cycle: model → tool → result → final answer |
| `agent.py` | The complete agent: ReAct loop, error handling, memory and trimming |
| `reflect_agent.py` | Reflection: a critic reviews each draft against the tool results |
| `test_critic.py` | Testing the critic with deliberately wrong drafts |
| `plan_agent.py` | Plan-and-execute: structured-output planner, step execution, tool-free final answer |
| `langgraph_agent.py` | The same agent rebuilt as a LangGraph graph: state, nodes, conditional edges, ToolNode |
---

## Run it yourself

1. Install the packages:
   ```
   pip install google-genai python-dotenv pydantic requests
   ```
2. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/) (no billing needed).
3. Create a `.env` file in the project folder (it is git-ignored):
   ```
   GEMINI_API_KEY=your-key-here
   ```
4. Start the agent:
   ```
   python agent.py
   ```
   Type questions at the `You:` prompt, `reset` to clear memory, `quit` to exit.

> The model name is set in `MODEL` at the top of each script. Google retires older models over time, so if you get a 404, check AI Studio for the current Flash model name.

---

## Key lessons learned

- **The model never runs code.** It only says which tool to call and with what arguments; my code executes it.
- **Docstrings and type hints are instructions for the model**, not just documentation.
- **An agent is only as good as its tools' data.** When a tool returned no rainfall data, the model confidently guessed "dry". Adding precipitation to the tool fixed the answer.
- **The model has no memory.** Every call resends the full history, so memory costs tokens and must be trimmed.
- **Lost context makes agents confidently wrong**, not confused, unless you tell them something was removed.

---

## Roadmap

- [x] Python for agents: tools, Pydantic, web APIs
- [x] LLM tool calling with Gemini
- [x] Agent loop from scratch (ReAct) with memory
- [x] Reflection and plan-and-execute patterns
- [ ] LangGraph and Google ADK
- [ ] Agentic RAG and long-term memory
- [ ] Agent protocols: MCP and A2A
- [ ] Multi-agent systems, guardrails and evaluation
- [ ] Deploying an agent on Google Cloud
