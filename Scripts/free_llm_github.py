from openai import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("GITHUB_TOKEN")

endpoint = "https://models.github.ai/inference"
model = "meta/Llama-3.2-90B-Vision-Instruct"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
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
