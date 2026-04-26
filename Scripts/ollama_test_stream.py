from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # любое значение
)

stream = client.chat.completions.create(
    model="gemma4:latest",
    messages=[
        {"role": "system", "content": "Отвечай кратко."},
        {
            "role": "user",
            "content": "Напиши короткий пример кода на Python и поясни его.",
        },
    ],
    temperature=0.2,
    stream=True,
)

for event in stream:
    # В стриме приходят чанки; текст обычно лежит в delta.content
    delta = event.choices[0].delta
    if delta and delta.content:
        print(delta.content, end="", flush=True)

print()  # перевод строки в конце
