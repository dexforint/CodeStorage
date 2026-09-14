# https://modal.com/glm-5-endpoint
import openai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("EXPERIENTIALLABS_TOKEN")


client = openai.OpenAI(base_url="https://api.experientiallabs.ai/v1", api_key=api_key)


resp = client.chat.completions.create(
    model="deepseek-v4.1-flash",
    messages=[
        {
            "role": "user",
            "content": "Привет! Напиши Python функцию для суммирования двух чисел.",
        }
    ],
)

print(resp)
print("#####")
print(resp.choices[0].message.content)
