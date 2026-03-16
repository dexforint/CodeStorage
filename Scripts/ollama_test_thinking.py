from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
)

stream = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[{"role": "user", "content": "Докажи, что √2 — иррациональное число"}],
    stream=True,
    extra_body={"think": True},
)

thinking_started = False
answer_started = False

for chunk in stream:
    print(chunk.choices[0].delta.reasoning, end="")
    # delta = chunk.choices[0].delta

    # # Ollama передаёт thinking через reasoning_content в delta
    # reasoning = getattr(delta, "reasoning_content", None)
    # content = delta.content

    # if reasoning:
    #     if not thinking_started:
    #         thinking_started = True
    #         print("🧠 Мысли модели:\n" + "=" * 40)
    #     print(reasoning, end="", flush=True)

    # if content:
    #     if not answer_started:
    #         answer_started = True
    #         if thinking_started:
    #             print("\n")
    #         print("💬 Ответ:\n" + "=" * 40)
    #     print(content, end="", flush=True)

print()  # финальный перенос строки

# from openai import OpenAI

# client = OpenAI(
#     base_url="http://localhost:11434/v1/",
#     api_key="ollama",  # обязателен, но игнорируется Ollama
# )

# response = client.chat.completions.create(
#     model="glm-4.7-flash",
#     messages=[{"role": "user", "content": "Сколько букв 'р' в слове 'клубника'?"}],
#     # Включаем thinking через extra_body
#     extra_body={"think": True},
# )

# message = response.choices[0].message

# # Финальный ответ модели
# print("=== Ответ ===")
# print(message.content)

# # "Мысли" модели (reasoning / thinking)
# # В OpenAI-совместимом API Ollama возвращает их
# # в поле reasoning_content (или через provider_specific_fields)
# if hasattr(message, "reasoning_content") and message.reasoning_content:
#     print("\n=== Мысли (reasoning_content) ===")
#     print(message.reasoning_content)

# # Альтернативно — через raw-ответ
# raw = response.model_dump()
# print("\n=== Raw response (для отладки) ===")
# import json

# print(json.dumps(raw, indent=2, ensure_ascii=False))
