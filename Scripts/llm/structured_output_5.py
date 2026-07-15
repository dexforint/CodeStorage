from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, TypeVar, Generic, Type

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

T = TypeVar("T", bound=BaseModel)


# ─── Универсальная обёртка ────────────────────────────────────────────────────
class KnowledgeWrapper(BaseModel, Generic[T]):
    is_known: bool = Field(
        description="True — данные найдены и достоверны. False — неизвестно"
    )
    reason: Optional[str] = Field(
        None,
        description="Причина, если is_known=False (например: 'частное лицо', 'вымышленный персонаж')",
    )
    data: Optional[T] = Field(None, description="Данные если is_known=True, иначе null")


# ─── Ваша схема данных ────────────────────────────────────────────────────────
class Person(BaseModel):
    name: str
    age: Optional[int] = None
    occupation: Optional[str] = None
    city: Optional[str] = None
    fun_fact: Optional[str] = None


class Movie(BaseModel):
    title: str
    director: Optional[str] = None
    year: Optional[int] = None
    rating: Optional[float] = None


# ─── Универсальная функция ────────────────────────────────────────────────────
def ask_llm(prompt: str, schema: Type[T]) -> KnowledgeWrapper:
    """
    Универсальная функция: работает с любой Pydantic-схемой.
    Если LLM не знает — вернёт is_known=False.
    """

    # Динамически создаём обёртку для конкретного типа
    class WrappedSchema(BaseModel):
        is_known: bool = Field(
            description="True — данные достоверны. False — нет информации"
        )
        reason: Optional[str] = Field(None, description="Причина если is_known=False")
        data: Optional[schema] = Field(  # type: ignore
            None, description="Данные если is_known=True, иначе null"
        )

    SYSTEM_PROMPT = f"""
Ты помощник, извлекающий структурированные данные.

Правила:
1. Если у тебя достоверная информация — заполни data и установи is_known=true
2. Если человек/объект неизвестен или ты не уверен — is_known=false, data=null
3. В поле reason объясни почему is_known=false
4. НИКОГДА не выдумывай данные
"""

    completion = client.beta.chat.completions.parse(
        model="gemma4:latest",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format=WrappedSchema,
    )
    return completion.choices[0].message.parsed


# ─── Тесты ───────────────────────────────────────────────────────────────────
queries = [
    ("Расскажи о Николе Тесле", Person),
    ("Расскажи о Vasya Pupkin 2000 Москва", Person),
    ("Информация о фильме Inception", Movie),
    ("Информация о фильме 'Космический огурец 3'", Movie),
]

for prompt, schema in queries:
    print("=" * 50)
    result = ask_llm(prompt, schema)

    if result.is_known:
        print(f"✅ Данные найдены:")
        for key, val in result.data.model_dump().items():
            if val is not None:
                print(f"   {key}: {val}")
    else:
        print(f"🤷 Данных нет")
        print(f"   Причина: {result.reason}")
