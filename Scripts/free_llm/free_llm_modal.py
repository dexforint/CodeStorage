# https://modal.com/glm-5-endpoint
import openai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MODAL_TOKEN")


client = openai.OpenAI(
    base_url="https://api.us-west-2.modal.direct/v1", api_key=api_key
)


resp = client.chat.completions.create(
    model="zai-org/GLM-5.1-FP8",
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
