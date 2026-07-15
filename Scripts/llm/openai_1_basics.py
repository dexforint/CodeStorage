import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Загружает .env файл

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Или просто OpenAI() — ключ читается автоматически
    timeout=30.0,  # Таймаут запроса в секундах
    max_retries=2,  # Количество автоматических повторов
)

# ─── Синхронная автопагинация ─────────────────────────────────────────────────
all_models = []
for model in client.models.list():  # Автоматически делает запросы на следующие страницы
    all_models.append(model.id)

print(f"Всего моделей: {len(all_models)}")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Ты полезный ассистент."},
        {"role": "user", "content": "Что такое Python?"},
    ],
    temperature=0.7,  # Креативность (0.0 — детерм., 2.0 — макс. случайность)
    max_tokens=500,  # Максимум токенов в ответе
    top_p=1.0,  # Nucleus sampling
    frequency_penalty=0.0,  # Штраф за повторение слов
    presence_penalty=0.0,  # Штраф за новые темы
    stop=["END"],  # Стоп-слова — модель остановится здесь
    n=1,  # Сколько вариантов ответа сгенерировать
)

# ─── Извлечение результата ────────────────────────────────────────────────────
text = response.choices[0].message.content
role = response.choices[0].message.role  # "assistant"
reason = response.choices[0].finish_reason  # "stop" / "length" / "tool_calls"

# ─── Метаданные использования ─────────────────────────────────────────────────
prompt_tokens = response.usage.prompt_tokens
completion_tokens = response.usage.completion_tokens
total_tokens = response.usage.total_tokens
