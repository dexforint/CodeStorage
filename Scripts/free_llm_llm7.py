import openai

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LLM7_TOKEN")


client = openai.OpenAI(base_url="https://api.llm7.io/v1", api_key=api_key)

resp = client.chat.completions.create(
    model="gpt-oss:20b",
    messages=[{"role": "user", "content": "Кто ты?"}],
)

print(resp)
print("#####")
print(resp.choices[0].message.content)

# https://api.llm7.io/v1/models

# codestral-latest
# ministral-8b-2512
# GLM-4.6V-Flash
# gpt-oss:20b
#

# meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
# deepseek-v3.1:671b-terminus
