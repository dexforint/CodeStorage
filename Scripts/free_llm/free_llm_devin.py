import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("DEVIN_TOKEN")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

json_data = {
    "prompt": "Привет! Напиши Python функцию для суммирования двух чисел.",
}

response = requests.post(
    "https://api.devin.ai/v1/sessions", headers=headers, json=json_data
)
print(response.text)
