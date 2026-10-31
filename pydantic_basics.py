import json
from pydantic import BaseModel, Field, ValidationError, ConfigDict


class WeatherInput(BaseModel):
    city: str = Field(min_length=2, description="City name, e.g. 'Pune'")
    days: int = Field(default=1, ge=1, le=7, description="Forecast days, 1 to 7")

class NoteInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=500, description="Note text, 1 to 500 characters")

# 1. Valid input
w = WeatherInput(city="Pune", days=3)
print(w.city, w.days)                     # Pune 3
w2 = WeatherInput(city="Pune", days="3")
print(w2.days, type(w2.days))
# 2. Default value used when days is missing
print(WeatherInput(city="Mumbai").days)   # 1

# 3. Invalid input -> clear error, no crash
try:
    WeatherInput(city="P", days=10)
except ValidationError as e:
    print(e)

# 4. The schema an LLM reads
print(json.dumps(WeatherInput.model_json_schema(), indent=2))
# 5. NoteInput schema
print(json.dumps(NoteInput.model_json_schema(), indent=2))

# 6. Empty text -> too short
try:
    NoteInput(text="")
except ValidationError as e:
    print(e)

# 7. Number instead of text -> wrong type
try:
    NoteInput(text=123)
except ValidationError as e:
    print(e)
try:
    print(NoteInput(text="   "))
except ValidationError as e:
    print(e)
