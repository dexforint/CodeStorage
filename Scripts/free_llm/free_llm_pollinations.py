# https://platform.xiaomimimo.com/#/docs/pricing
# https://platform.xiaomimimo.com/#/console/api-keys
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("POLLINATIONS_TOKEN")

client = OpenAI(
    api_key=api_key,
    base_url="https://gen.pollinations.ai/v1",
)

resp = client.chat.completions.create(
    model="openai-fast",
    messages=[
        {
            "role": "user",
            "content": "Напиши Python функцию (только код) для суммирования двух чисел",
        },
    ],
    # max_completion_tokens=1024,
    # temperature=0.3,
    # top_p=0.95,
    # stream=False,
    # stop=None,
    # frequency_penalty=0,
    # presence_penalty=0,
    # extra_body={"thinking": {"type": "disabled"}},
)
# claude-large

print(resp)
print("#####")
print(resp.choices[0].message.content)
