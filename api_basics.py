import requests

resp = requests.get(
    "https://geocoding-api.open-meteo.com/v1/search",
    params={"name": "Aurangabad", "count": 1},
    timeout=10,
)
resp.raise_for_status()      # raises an error if status is 4xx/5xx
data = resp.json()           # JSON response -> Python dict
print(data)


results = data.get("results")     # returns None instead of crashing
if not results:
    print("City not found")
else:
    place = results[0]
    print(place["name"], place["latitude"], place["longitude"])