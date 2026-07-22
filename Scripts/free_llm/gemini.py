import openai

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_TOKEN")

# https://unity2.ai/v1/chat/completions
client = openai.OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=api_key
)


resp = client.chat.completions.create(
    model="gemini-3.5-flash",
    messages=[
        {
            "role": "user",
            "content": "Напиши Python функцию для суммирования двух чисел",
        }
    ],
)

print(resp)
print("#####")
print(resp.choices[0].message.content)
