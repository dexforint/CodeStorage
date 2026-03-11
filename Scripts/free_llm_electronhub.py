import openai

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ELECTRONHUB_TOKEN")


client = openai.OpenAI(base_url="https://api.electronhub.ai/v1", api_key=api_key)

resp = client.chat.completions.create(
    model="gpt-oss-120b:free",
    messages=[
        {
            "role": "user",
            "content": "Кто ты?",
        }
    ],
)

print(resp)
print("#####")
print(resp.choices[0].message.content)
