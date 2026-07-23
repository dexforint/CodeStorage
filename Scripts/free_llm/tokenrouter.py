from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TOKENROUTER_TOKEN")

client = OpenAI(
    base_url="https://api.tokenrouter.com/v1",
    api_key=api_key,
)

messages = [
    {
        "role": "system",
        "content": "You are an intelligent assistant, please reply concisely.",
    },
    {"role": "user", "content": "Hello, what kind of model are you?"},
]

resp = client.chat.completions.create(
    model="z-ai/glm-5.2-free",
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
