import openai
from openai import OpenAI

client = OpenAI(max_retries=3)  # Авто-ретрай


# ─── Полная обработка всех ошибок ────────────────────────────────────────────
def safe_request(prompt: str) -> str | None:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            timeout=30.0,
        )
        return response.choices[0].message.content

    except openai.AuthenticationError as e:
        # 401 — Неверный API-ключ
        print(f"🔑 Ошибка аутентификации: {e}")

    except openai.RateLimitError as e:
        # 429 — Превышен лимит запросов
        print(f"⏱️  Лимит запросов: {e}")

    except openai.NotFoundError as e:
        # 404 — Модель не найдена
        print(f"🔍 Не найдено: {e}")

    except openai.BadRequestError as e:
        # 400 — Неверный запрос (например, слишком длинный prompt)
        print(f"📛 Неверный запрос: {e}")

    except openai.APIConnectionError as e:
        # Проблема с соединением
        print(f"🌐 Ошибка соединения: {e}")

    except openai.APITimeoutError as e:
        # Таймаут
        print(f"⏰ Таймаут: {e}")

    except openai.APIStatusError as e:
        # Любая другая HTTP ошибка
        print(f"❌ HTTP {e.status_code}: {e.message}")
        print(f"   Request ID: {e.request_id}")  # Для отчёта в OpenAI

    return None


result = safe_request("Привет!")
