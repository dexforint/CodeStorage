# https://modal.com/glm-5-endpoint
import openai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("AEROLINK_TOKEN")


client = openai.OpenAI(base_url="https://capi.aerolink.lat/", api_key=api_key)


resp = client.chat.completions.create(
    model="gpt-5.6-sol",
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
