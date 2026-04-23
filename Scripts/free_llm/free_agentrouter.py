import openai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("AGENTROUTER_TOKEN")


client = openai.OpenAI(base_url="https://agentrouter.org/v1", api_key=api_key)


resp = client.chat.completions.create(
    model="claude-sonnet-4-5-20250929",
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
