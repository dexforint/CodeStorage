import openai

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("42GPU_TOKEN")


client = openai.OpenAI(base_url="https://api.42gpu.ru/v1", api_key=api_key)


resp = client.chat.completions.create(
    model="local/qwen3.5-397b",
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
