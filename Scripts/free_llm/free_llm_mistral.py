import openai
from prettytable import PrettyTable

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MISTRAL_TOKEN")


client = openai.OpenAI(base_url="https://api.mistral.ai/v1", api_key=api_key)


resp = client.chat.completions.create(
    model="codestral-latest",
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
