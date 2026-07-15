import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ✅ 1. Переменные окружения вместо хардкода ключей
client = OpenAI()  # Автоматически читает OPENAI_API_KEY из env

# ✅ 2. Переиспользуйте один клиент (connection pooling)
# НЕ создавайте клиент в каждой функции!
GLOBAL_CLIENT = OpenAI(
    max_retries=3,
    timeout=60.0,
)


# ✅ 3. Следите за токенами
def estimate_cost(response) -> float:
    """Примерная стоимость для gpt-4o."""
    input_cost = response.usage.prompt_tokens * 0.0000025  # $2.50/1M
    output_cost = response.usage.completion_tokens * 0.0000100  # $10/1M
    return input_cost + output_cost


response = GLOBAL_CLIENT.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Привет!"}],
)
print(f"💰 Стоимость запроса: ${estimate_cost(response):.6f}")

# ✅ 4. Request ID для дебаггинга
print(f"🆔 Request ID: {response._request_id}")

# ✅ 5. Типизация через TypedDict
from typing import TypedDict, Literal


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


def build_messages(system: str, user: str) -> list[Message]:
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]
