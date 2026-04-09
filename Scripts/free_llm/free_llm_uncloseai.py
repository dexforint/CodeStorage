# Python Fizzbuzz Example with Hermes
from openai import OpenAI

# https://uncloseai.com/python-examples.html
client = OpenAI(base_url="https://hermes.ai.unturf.com/v1", api_key="choose-any-value")

MODEL = "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic"

messages = [{"role": "user", "content": "Привет! Кто ты?"}]

response = client.chat.completions.create(
    model=MODEL, messages=messages, temperature=0.5, max_tokens=150
)

print(response.choices[0].message.content)
