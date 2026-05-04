import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("YEPAPI_TOKEN")

# import openai

# client = openai.OpenAI(base_url="https://api.yepapi.com/v1/ai/chat", api_key=api_key)

# resp = client.chat.completions.create(
#     model="deepseek/deepseek-v4-pro",
#     messages=[
#         {
#             "role": "user",
#             "content": "Напиши Python функцию для суммирования двух чисел",
#         }
#     ],
# )

# print(resp)
# print("#####")
# print(resp.choices[0].message.content)


import requests

res = requests.post(
    "https://api.yepapi.com/v1/ai/chat",
    headers={"x-api-key": api_key},
    json={
        "model": "deepseek/deepseek-v4-pro",
        "messages": [
            {
                "role": "user",
                "content": "Напиши Python функцию для суммирования двух чисел.",
            }
        ],
        "maxTokens": 2048,
        "stream": True,
    },
    stream=True,
)

for line in res.iter_lines():
    line = line.decode()
    if not line.startswith("data: "):
        continue
    data = line[6:].strip()
    if data == "[DONE]":
        break
    import json

    parsed = json.loads(data)
    text = parsed.get("delta", {}).get("content", "")
    print(text, end="", flush=True)
