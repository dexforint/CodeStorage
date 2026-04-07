import requests

import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("APIFREELLM_TOKEN")

response = requests.post(
    "https://apifreellm.com/api/v1/chat",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}",
    },
    json={"message": "Hello, how are you?"},
)

print(response.json())
