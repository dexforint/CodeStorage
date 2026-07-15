from openai import OpenAI
import openai
from pydantic import BaseModel, Field
from typing import Optional

# ─── Подключение к Ollama через OpenAI-совместимый API ───────────────────────
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Любая строка, Ollama не проверяет ключ
)


# ─── Определяем структуру ответа через Pydantic ──────────────────────────────
class Person(BaseModel):
    name: str = Field(description="Полное имя человека")
    age: int = Field(description="Возраст")
    occupation: str = Field(description="Профессия")
    city: str = Field(description="Город проживания")
    fun_fact: Optional[str] = Field(None, description="Интересный факт")


# ─── Запрос к модели ──────────────────────────────────────────────────────────
try:
    completion = client.beta.chat.completions.parse(
        model="gemma4:latest",
        temperature=0,  # Для детерминированного вывода
        messages=[
            {
                "role": "system",
                "content": "Ты помощник, который извлекает информацию о людях.",
            },
            {
                "role": "user",
                "content": (
                    "Расскажи мне о Аброусе Зарамовиче: его возраст на момент смерти, "
                    "профессия, город где он жил последние годы и интересный факт."
                ),
            },
        ],
        response_format=Person,
    )

    response = completion.choices[0].message

    if response.parsed:
        person: Person = response.parsed
        print(f"👤 Имя:       {person.name}")
        print(f"🎂 Возраст:   {person.age}")
        print(f"💼 Профессия: {person.occupation}")
        print(f"🏙️ Город:     {person.city}")
        print(f"💡 Факт:      {person.fun_fact}")
    elif response.refusal:
        print(f"⚠️ Модель отказала: {response.refusal}")

except openai.LengthFinishReasonError as e:
    print(f"❌ Слишком много токенов: {e}")
except Exception as e:
    print(f"❌ Ошибка: {e}")
