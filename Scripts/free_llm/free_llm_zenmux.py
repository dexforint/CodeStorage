from openai import OpenAI
import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ZENMUX_TOKEN")

endpoint = "https://zenmux.ai/api/v1"
model = "xiaomi/mimo-v2-flash-free"
model = "kuaishou/kat-coder-pro-v1-free"
model = "z-ai/glm-4.7-flash-free"
model = "stepfun/step-3.5-flash-free"

# url = "https://zenmux.ai/api/v1/models"
# headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
# response = requests.get(url, headers=headers)
# print(response.json())

client = OpenAI(
    base_url=endpoint,
    api_key=api_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Кто ты?",
        },
    ],
    model=model,
)

print(response.choices[0].message.content)
