from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Models supporting content generation:\n")

for model in client.models.list():
    actions = getattr(model, "supported_actions", [])

    if "generateContent" in actions:
        print(model.name)