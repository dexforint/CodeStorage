from openai import OpenAI
import json

client = OpenAI()

# ─── Описываем инструменты ────────────────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Получить текущую погоду в городе",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Название города, например 'Москва'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Единица измерения температуры",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Поиск информации в базе данных",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
]


# ─── Реальные функции ─────────────────────────────────────────────────────────
def get_weather(city: str, unit: str = "celsius") -> dict:
    # Здесь реальный вызов API погоды
    return {"city": city, "temperature": 22, "unit": unit, "condition": "Солнечно"}


def search_database(query: str, limit: int = 10) -> dict:
    # Здесь реальный запрос к БД
    return {"results": [f"Результат {i} для '{query}'" for i in range(limit)]}


AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "search_database": search_database,
}


# ─── Агентный цикл ────────────────────────────────────────────────────────────
def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # "auto" / "none" / {"type": "function", "function": {"name": "..."}}
        )

        message = response.choices[0].message
        messages.append(message)

        # Если модель решила не вызывать функции — возвращаем ответ
        if response.choices[0].finish_reason == "stop":
            return message.content

        # Модель хочет вызвать функции
        if response.choices[0].finish_reason == "tool_calls":
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"🔧 Вызов: {func_name}({func_args})")

                # Вызываем реальную функцию
                result = AVAILABLE_FUNCTIONS[func_name](**func_args)

                # Возвращаем результат модели
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )


# ─── Тест ─────────────────────────────────────────────────────────────────────
answer = run_agent("Какая погода в Москве сейчас?")
print(f"\n🤖 {answer}")
