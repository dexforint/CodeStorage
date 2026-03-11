import openai


import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_TOKEN")


client = openai.OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=api_key
)


resp = client.chat.completions.create(
    model="gemini-3.1-flash-lite-preview",
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
