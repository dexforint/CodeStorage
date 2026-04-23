# https://modal.com/glm-5-endpoint
import openai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("UPSTAGE_TOKEN")


client = openai.OpenAI(base_url="https://api.upstage.ai/v1", api_key=api_key)


resp = client.chat.completions.create(
    model="solar-pro3",
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
