from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


class Person(BaseModel):
    name: str
    age: Optional[int] = Field(None, description="Возраст. None если неизвестно")
    occupation: Optional[str] = Field(
        None, description="Профессия. None если неизвестно"
    )
    city: Optional[str] = Field(None, description="Город. None если неизвестно")
    fun_fact: Optional[str] = Field(None, description="Факт. None если неизвестно")


SYSTEM_PROMPT = """
Ты помощник, который извлекает информацию о людях.

ВАЖНЫЕ ПРАВИЛА:
- Если ты не знаешь значение поля — верни null, НЕ выдумывай
- Если человек вымышленный или ты не уверен — верни null во всех полях
- Лучше вернуть null, чем написать неверную информацию
"""


def get_person_info(name: str) -> Person:
    completion = client.beta.chat.completions.parse(
        model="gemma4:latest",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Расскажи мне о человеке: {name}"},
        ],
        response_format=Person,
    )
    return completion.choices[0].message.parsed


def print_person(person: Person):
    def fmt(val):
        return val if val is not None else "❓ Неизвестно"

    print(f"👤 Имя:       {person.name}")
    print(f"🎂 Возраст:   {fmt(person.age)}")
    print(f"💼 Профессия: {fmt(person.occupation)}")
    print(f"🏙️  Город:     {fmt(person.city)}")
    print(f"💡 Факт:      {fmt(person.fun_fact)}")


# ─── Тест ────────────────────────────────────────────────────────────────────
print("=" * 40)
print_person(get_person_info("Никола Тесла"))  # Известный человек

print("=" * 40)
print_person(get_person_info("Иван Петров 1987"))  # Неизвестный человек
