from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ONLYSQ_TOKEN")

client = OpenAI(
    base_url="https://api.onlysq.ru/ai/openai",
    api_key=api_key,
)

completion = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "user",
            "content": "Привет! Кто ты?",
        },
    ],
)

print(completion.choices[0].message.content)
