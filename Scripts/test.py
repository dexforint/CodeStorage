from openai import OpenAI

# Подключаемся к локальной Ollama
# api_key может быть любым (например, "ollama"),
# так как локальная модель не требует авторизации,
# но параметр обязателен для клиента OpenAI.
client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")

# Имя модели, которую вы скачали (можно посмотреть командой ollama list)
model_name = "qwen3.6:latest"


print(f"Отправка запроса к модели {model_name}...\n")

try:
    # Делаем запрос к модели (Chat Completions)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "Почему небо голубое? Ответь в двух предложениях.",
            },
        ],
        temperature=0.7,
        max_tokens=150,
    )

    # Выводим ответ
    answer = response.choices[0].message.content
    print("Ответ модели:")
    print(answer)

except Exception as e:
    print(f"Произошла ошибка: {e}")
