from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("POLZA_TOKEN")

client = OpenAI(
    base_url="https://polza.ai/api/v1",
    api_key=api_key,
)

# completion = client.chat.completions.create(
#     model="openai/gpt-4o",
#     messages=[{"role": "user", "content": "Что думаешь об этой жизни?"}],
# )

# print(completion.choices[0].message.content)

response = client.chat.completions.create(
    model="google/lyria-3-pro-preview",
    messages=[{"role": "user", "content": "Напиши спокойную музыку в стиле ambient"}],
    audio={"format": "mp3"},
    stream=True,
)

audio_chunks = []
for chunk in response:
    audio = getattr(chunk.choices[0].delta, "audio", None) if chunk.choices else None
    if audio and getattr(audio, "data", None):
        audio_chunks.append(audio.data)

if audio_chunks:
    mp3_data = base64.b64decode("".join(audio_chunks))
    with open("./data/generated_audio.mp3", "wb") as f:
        f.write(mp3_data)
    print("Audio saved to generated_audio.mp3")
