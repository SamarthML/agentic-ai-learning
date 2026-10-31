from pydantic import BaseModel, Field
from google.genai import types, errors
from agent import client, MODEL, run_agent, config


class Plan(BaseModel):
    steps: list[str] = Field(description="2 to 5 short steps, in order")


PLANNER = """You are a planner. Break the user's request into 2 to 5 short steps, in order.
Available tools: get_weather(city) returns current temperature, wind and precipitation;
save_note(text) saves a note.
Each step must be one clear action. Do NOT do the steps, only list them."""

planner_config = types.GenerateContentConfig(
    system_instruction=PLANNER,
    response_mime_type="application/json",   # reply must be JSON...
    response_schema=Plan,                    # ...shaped exactly like the Plan model
)

# Same as the agent's config, but the model is NOT ALLOWED to call tools
final_config = config.model_copy(update={
    "tool_config": types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="NONE")
    )
})


def make_plan(request: str) -> Plan | None:
    try:
        resp = client.models.generate_content(model=MODEL, contents=request, config=planner_config)
    except errors.APIError as e:
        print(f"Planning failed (code {e.code}): {e.message}")
        return None
    return resp.parsed   # already a Plan object, validated by Pydantic


def plan_and_execute(request: str) -> str:
    plan = make_plan(request)
    if plan is None or not plan.steps:
        return "Could not make a plan."

    print("PLAN:")
    for i, step in enumerate(plan.steps, 1):
        print(f"  {i}. {step}")

    history = []   # shared, so later steps can use earlier results
    for i, step in enumerate(plan.steps, 1):
        print(f"\nEXECUTING step {i}: {step}")
        result = run_agent(step, history, max_steps=3)
        print(f"  -> {result}")

    # Final answer: OUTSIDE the loop (same indentation as "for"), tools switched off
    history.append(types.Content(role="user", parts=[types.Part(
        text=f"Using only the results above, give a short final answer to: {request}")]))
    try:
        resp = client.models.generate_content(model=MODEL, contents=history, config=final_config)
    except errors.APIError as e:
        return f"Final answer failed (code {e.code}): {e.message}"
    return resp.text


if __name__ == "__main__":
    request = input("Request: ")
    print("\nFINAL:", plan_and_execute(request))