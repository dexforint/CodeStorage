# https://modal.com/glm-5-endpoint
import openai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BAI_TOKEN")


client = openai.OpenAI(base_url="https://api.b.ai/v1", api_key=api_key)

# mimo-v2.5
# glm-5.3-flash
# hy3
# qwen3.8-flash
resp = client.chat.completions.create(
    model="glm-5.3-flash",
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
