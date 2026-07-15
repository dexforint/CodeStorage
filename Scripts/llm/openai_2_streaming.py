from openai import OpenAI

client = OpenAI()

# ─── Базовый стриминг ─────────────────────────────────────────────────────────
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Напиши эссе о Python."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content is not None:
        print(delta.content, end="", flush=True)

print()  # Перенос строки в конце


# ─── Стриминг с накоплением полного текста ───────────────────────────────────
def stream_response(prompt: str) -> str:
    """Стримит в консоль и возвращает полный текст."""
    full_text = ""

    with client.chat.completions.stream(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_text += delta
            print(delta, end="", flush=True)

    print()
    return full_text


result = stream_response("Объясни ООП за 3 абзаца.")
print(f"\nДлина ответа: {len(result)} символов")
