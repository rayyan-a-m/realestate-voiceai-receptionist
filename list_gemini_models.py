import json
import requests
from config import GOOGLE_API_KEY

if not GOOGLE_API_KEY:
    raise SystemExit("GOOGLE_API_KEY is missing. Set it in your .env file.")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GOOGLE_API_KEY}"
resp = requests.get(url, timeout=30)
print("HTTP", resp.status_code)

if not resp.ok:
    print(resp.text)
    raise SystemExit(1)

data = resp.json()
models = data.get("models", [])

# Print concise, useful info
rows = []
for m in models:
    rows.append({
        "name": m.get("name"),
        "displayName": m.get("displayName"),
        "supported": m.get("supportedGenerationMethods"),
    })

rows.sort(key=lambda r: r["name"] or "")
print(json.dumps(rows, indent=2))
