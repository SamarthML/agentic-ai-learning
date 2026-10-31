from agent import run_agent
from reflect_agent import reflect

history = []
run_agent("What's the weather in Pune?", history)   # real tool data in history

bad_drafts = [
    # Obvious: invented temperature, sky and humidity
    "Pune is 35°C with clear sunny skies and 60% humidity. Great picnic weather!",
    # Subtle: turns a measurement into a forecast
    "Pune has light wind and there is a small chance of rain later today.",
]

for bad in bad_drafts:
    print("\nBAD DRAFT:", bad)
    print("CRITIC RESULT:", reflect(history, bad))