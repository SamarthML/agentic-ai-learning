import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import get_weather, save_note   # your own tools!

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash"

config = types.GenerateContentConfig(
    tools=[get_weather, save_note],
    # Turn OFF auto-running, so we can SEE what the model decides
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

question = "Compare the weather in London and Mumbai"
response = client.models.generate_content(model=MODEL, contents=question, config=config)

if response.function_calls:
    for call in response.function_calls:
        print("Model wants to call:", call.name)
        print("With arguments:     ", call.args)
else:
    print("No tool call. Model answered directly:")
    print(response.text)