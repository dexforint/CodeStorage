from openai import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OLLAMA_TOKEN")

# endpoint = "https://ollama.com/v1"
endpoint = "http://localhost:11434/v1/"
# model = "gpt-oss:120b-cloud"
model = "glm-4.7-flash"

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
