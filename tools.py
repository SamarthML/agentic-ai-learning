import json
import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class NoteInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=500, description="Note text, 1 to 500 characters")


def save_note(text: str) -> dict:
    """Append a note to notes.txt.

    Use this when the user asks to remember, save, or write down something.

    Args:
        text: The note content to save. 1 to 500 characters.

    Returns:
        {"status": "success", "saved": text} on success,
        {"status": "error", "message": ...} if the input is invalid or the file can't be written.
    """
    try:
        note = NoteInput(text=text)
    except ValidationError as e:
        err = e.errors()[0]
        return {"status": "error", "message": f"Invalid note text: {err['msg']}."}

    try:
        with open("notes.txt", "a", encoding="utf-8") as f:
            f.write(note.text + "\n")
    except OSError as e:
        return {"status": "error", "message": f"Could not save note: {e}"}

    return {"status": "success", "saved": note.text}


class WeatherInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    city: str = Field(min_length=2, description="City name, e.g. 'Pune'")


def get_weather(city: str) -> dict:
    """Get the current temperature and wind speed for a city.

    Use this when the user asks about the weather in a specific place.

    Args:
        city: The city name, e.g. "Pune". At least 2 characters.

    Returns:
        {"status": "success", "city": ..., "temperature_c": ..., "wind_kmh": ...} on success,
        {"status": "error", "message": ...} if the city is invalid or not found,
        or the weather service is unavailable.
    """
    # 1. Validate
    try:
        data = WeatherInput(city=city)
    except ValidationError as e:
        err = e.errors()[0]
        return {"status": "error", "message": f"Invalid city: {err['msg']}."}

    try:
        # 2. City -> coordinates
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": data.city, "count": 1},
            timeout=10,
        )
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            return {"status": "error", "message": f"City not found: {data.city}."}
        place = results[0]

        # 3. Coordinates -> current weather
        fc = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,wind_speed_10m,precipitation",
            },
            timeout=10,
        )
        fc.raise_for_status()
        weather = fc.json()

    except requests.RequestException as e:
        return {"status": "error", "message": f"Weather service unavailable: {e}"}

    # 4. Build a readable place name, e.g. "Pune, Maharashtra, India"
    parts = [place.get("name"), place.get("admin1"), place.get("country")]
    full_name = ", ".join(p for p in parts if p)

    return {
        "status": "success",
        "city": full_name,
        "temperature_c": weather["current"]["temperature_2m"],
        "wind_kmh": weather["current"]["wind_speed_10m"],
        "precipitation_mm": weather["current"]["precipitation"],
    }


# --- testing ---
if __name__ == "__main__":
    args = json.loads('{"text": "Learn LangGraph next week"}')
    print(save_note(**args))
    print(save_note("   "))
    print(save_note(123))
    print(save_note("a" * 501))
    print(get_weather("Pune"))
    print(get_weather("Xyzabc"))
    print(get_weather("   "))
    print(get_weather("London"))
