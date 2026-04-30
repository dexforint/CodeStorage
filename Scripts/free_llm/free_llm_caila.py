import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("CAILA_TOKEN")

import openai

client = openai.OpenAI(base_url="https://caila.io/api/adapters/openai", api_key=api_key)

resp = client.chat.completions.create(
    model="just-ai/deepseek/deepseek-r1",
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
