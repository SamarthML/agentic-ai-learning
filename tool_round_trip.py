import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import get_weather, save_note

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash"

# Tool registry: name the model uses -> your real function
TOOLS = {"get_weather": get_weather, "save_note": save_note}

config = types.GenerateContentConfig(
    tools=list(TOOLS.values()),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

question = "Check the weather in Pune and save it as a note"

# The conversation history starts with the user's message
contents = [types.Content(role="user", parts=[types.Part(text=question)])]

# --- Message 2: model decides ---
response = client.models.generate_content(model=MODEL, contents=contents, config=config)

if not response.function_calls:
    print("Answered directly:", response.text)
else:
    # Add the model's reply (its tool requests) to the history, exactly as it came
    contents.append(response.candidates[0].content)

    # --- Message 3: we run the tools ---
    result_parts = []
    for call in response.function_calls:
        func = TOOLS.get(call.name)
        if func is None:
            result = {"status": "error", "message": f"Unknown tool: {call.name}"}
        else:
            result = func(**call.args)
        print(f"Ran {call.name}({call.args}) -> {result}")
        result_parts.append(
            types.Part.from_function_response(name=call.name, response=result)
        )
    contents.append(types.Content(role="user", parts=result_parts))

    # --- Message 4: model answers using the results ---
    final = client.models.generate_content(model=MODEL, contents=contents, config=config)
    print("\nFinal answer:", final.text)