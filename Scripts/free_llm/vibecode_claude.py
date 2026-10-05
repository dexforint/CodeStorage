import openai

import os
from dotenv import load_dotenv

load_dotenv()

# @llm_api_model_router_bot
# Нужен VPN
api_key = os.getenv("LLM_API_MODEL_ROUTER_BOT")

client = openai.OpenAI(
    base_url="https://api.vibecode-claude.online/v1", api_key=api_key
)

# 20 запр./30 с
# gpt-5.6-sol - 272K
# claude-opus-5 - 500K
# claude-sonnet-5 - 1000K
# grok-4.6 - 500K
resp = client.chat.completions.create(
    model="claude-opus-5",
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
