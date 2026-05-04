# https://freetheai.xyz/models/

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("FREETHEAI_TOKEN")

client = OpenAI(
    base_url="https://api.freetheai.xyz/v1",
    api_key=api_key,
)

completion = client.chat.completions.create(
    model="cat/claude-4-6-sonnet",
    messages=[
        {
            "role": "user",
            "content": "Привет! Кто ты?",
        },
    ],
)

print(completion.choices[0].message.content)
