import openai
from prettytable import PrettyTable

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("AGENTPLATFORM_TOKEN")


client = openai.OpenAI(base_url="https://api.agentplatform.ru/v1", api_key=api_key)


resp = client.chat.completions.create(
    model="google/gemini-3-flash-preview",
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
