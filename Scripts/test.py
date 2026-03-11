from openai import OpenAI

api_key = "sk-ivZpJ5YjnS4DKRz6u4ET-A"
endpoint = "https://ai-gateway.vercel.sh/v1"
model = "gpt-oss:120b-cloud"

# url = "https://zenmux.ai/api/v1/models"
# headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
# response = requests.get(url, headers=headers)
# print(response.json())

client = OpenAI(
    base_url=endpoint,
    api_key=api_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Кто ты?",
        },
    ],
    model=model,
)

print(response.choices[0].message.content)
