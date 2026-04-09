from openai import OpenAI

# https://github.com/alistaitsacle/free-llm-api-keys
client = OpenAI(
    base_url="https://aiapiv2.pekpik.com/v1",
    api_key="sk-EvBYW8DU4KzUMKtip56wuZqrrJPkNciEpW2L72FuRheZrhC0",
)

response = client.chat.completions.create(
    model="smart-chat",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)
print(response.choices[0].message.content)
