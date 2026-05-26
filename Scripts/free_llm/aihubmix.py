# https://console.aihubmix.com
import os
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("AIHUBMIX_TOKEN")

client = openai.OpenAI(
    api_key="api_key",  # Replace with the key generated in AIHubMix
    base_url="https://aihubmix.com/v1",
)

response = client.chat.completions.create(
    model="gpt-5.5-free",  # gpt-5.5-free The reasoning depth of the model defaults to none; the chat interface does not support modifying the reasoning intensity. Please switch to gpt-5.5-free-high/low or use the responses interface.
    messages=[
        {"role": "user", "content": "Привет! Кто ты? Дай полную информацию о себе"}
    ],
    temperature=0.7,  # Default is 1
)

print(response.choices[0].message.content)
