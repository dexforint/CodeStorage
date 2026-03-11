import openai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("VOIDAI_TOKEN")


client = openai.OpenAI(base_url="https://api.voidai.app/v1", api_key=api_key)

resp = client.chat.completions.create(
    model="gpt-5.4",
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
