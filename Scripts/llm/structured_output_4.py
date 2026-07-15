from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


class ConfidenceLevel(str, Enum):
    HIGH = "high"  # Уверен на 100%, факт из реальной биографии
    MEDIUM = "medium"  # Вероятно верно, но не уверен
    LOW = "low"  # Догадка
    UNKNOWN = "unknown"  # Нет информации


class PersonField(BaseModel):
    value: Optional[str] = Field(None, description="Значение поля или null")
    confidence: ConfidenceLevel = Field(description="Уровень уверенности")


class Person(BaseModel):
    is_known: bool = Field(
        description="True если человек реально существовал и ты знаешь о нём"
    )
    name: str
    age: PersonField
    occupation: PersonField
    city: PersonField
    fun_fact: PersonField


SYSTEM_PROMPT = """
Ты помощник, который извлекает информацию о людях.

Для каждого поля укажи:
- value: реальное значение или null если не знаешь
- confidence: уровень уверенности
  * high    — 100% уверен, факт из биографии
  * medium  — вероятно верно
  * low     — предположение
  * unknown — нет информации

НИКОГДА не выдумывай факты. Если не знаешь — unknown + null.
"""


def get_person_info(query: str) -> Person:
    completion = client.beta.chat.completions.parse(
        model="gemma4:latest",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Информация о человеке: {query}"},
        ],
        response_format=Person,
    )
    return completion.choices[0].message.parsed


CONFIDENCE_ICON = {
    ConfidenceLevel.HIGH: "🟢",
    ConfidenceLevel.MEDIUM: "🟡",
    ConfidenceLevel.LOW: "🔴",
    ConfidenceLevel.UNKNOWN: "⬜",
}


def print_person(person: Person):
    if not person.is_known:
        print(f"❌ Человек '{person.name}' неизвестен или не существует")
        return

    def fmt(field: PersonField) -> str:
        icon = CONFIDENCE_ICON[field.confidence]
        val = field.value or "Неизвестно"
        return f"{val}  {icon} [{field.confidence.value}]"

    print(f"✅ Человек найден: {person.name}")
    print(f"   🎂 Возраст:   {fmt(person.age)}")
    print(f"   💼 Профессия: {fmt(person.occupation)}")
    print(f"   🏙️  Город:     {fmt(person.city)}")
    print(f"   💡 Факт:      {fmt(person.fun_fact)}")


# ─── Тесты ───────────────────────────────────────────────────────────────────
for query in ["Никола Тесла", "Алексей Смирнов из Твери", "Альберт Эйнштейн"]:
    print("=" * 55)
    print_person(get_person_info(query))
