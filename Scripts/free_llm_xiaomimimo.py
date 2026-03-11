# https://platform.xiaomimimo.com/#/docs/pricing
# https://platform.xiaomimimo.com/#/console/api-keys
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("AIRFORCE_TOKEN")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.xiaomimimo.com/v1",
)

completion = client.chat.completions.create(
    model="mimo-v2-flash",
    messages=[
        {
            "role": "system",
            "content": "You are MiMo, an AI assistant developed by Xiaomi. Today is date: Tuesday, December 16, 2025. Your knowledge cutoff date is December 2024.",
        },
        {"role": "user", "content": "please introduce yourself"},
    ],
    max_completion_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    stream=False,
    stop=None,
    frequency_penalty=0,
    presence_penalty=0,
    extra_body={"thinking": {"type": "disabled"}},
)

print(completion.model_dump_json())
