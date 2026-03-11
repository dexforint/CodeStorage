import openai
import requests

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NAVY_TOKEN2")

headers = {"Authorization": f"Bearer {api_key}"}

response = requests.get("https://api.navy/v1/usage", headers=headers)
print(response.json())

# https://api.navy/v1/models

# client = openai.OpenAI(base_url="https://api.navy/v1", api_key=api_key)

# resp = client.chat.completions.create(
#     model="gemini-3.1-flash-lite-preview",
#     messages=[
#         {
#             "role": "user",
#             "content": "Кто ты?",
#         }
#     ],
# )

# print(resp)
# print("#####")
# print(resp.choices[0].message.content)
