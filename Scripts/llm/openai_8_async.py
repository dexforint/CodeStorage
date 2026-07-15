import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()


# ─── Базовый async запрос ─────────────────────────────────────────────────────
async def ask(question: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


# ─── Параллельные запросы ─────────────────────────────────────────────────────
async def ask_parallel(questions: list[str]) -> list[str]:
    """Отправляет все вопросы ОДНОВРЕМЕННО — многократное ускорение!"""
    tasks = [ask(q) for q in questions]
    return await asyncio.gather(*tasks)


# ─── Async стриминг ──────────────────────────────────────────────────────────
async def stream_async(prompt: str) -> str:
    full_text = ""
    async with client.chat.completions.stream(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_text += delta
            print(delta, end="", flush=True)
    print()
    return full_text


# ─── Запуск ──────────────────────────────────────────────────────────────────
async def main():
    # Параллельно 3 запроса
    questions = [
        "Столица Франции?",
        "Что такое рекурсия?",
        "Напиши Hello World на Rust.",
    ]

    print("🚀 Отправляем запросы параллельно...")
    answers = await ask_parallel(questions)

    for q, a in zip(questions, answers):
        print(f"\n❓ {q}\n💬 {a[:100]}...")


asyncio.run(main())
