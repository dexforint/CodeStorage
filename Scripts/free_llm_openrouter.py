import openai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_TOKEN")


client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


resp = client.chat.completions.create(
    model="arcee-ai/trinity-large-preview:free",
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
