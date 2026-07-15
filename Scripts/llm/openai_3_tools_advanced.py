import inspect
import json
from typing import Any, Callable, Literal, Optional, Union, get_args, get_origin
from docstring_parser import parse as parse_docstring


# ─── Маппинг Python-типов → JSON Schema типы ─────────────────────────────────
def python_type_to_json_schema(annotation) -> dict:
    """Конвертирует Python type hint в JSON Schema."""

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[X] → тип X (просто не обязательный)
    if origin is Union and type(None) in args:
        inner = [a for a in args if a is not type(None)][0]
        return python_type_to_json_schema(inner)

    # Literal["a", "b", "c"] → enum
    if origin is Literal:
        values = list(args)
        base_type = type(values[0]).__name__
        type_map = {"str": "string", "int": "integer", "float": "number"}
        return {
            "type": type_map.get(base_type, "string"),
            "enum": values,
        }

    # list[X] / List[X]
    if origin is list:
        result = {"type": "array"}
        if args:
            result["items"] = python_type_to_json_schema(args[0])
        return result

    # dict[K, V] / Dict[K, V]
    if origin is dict:
        return {"type": "object"}

    # Примитивные типы
    TYPE_MAP = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        Any: {},
    }

    return TYPE_MAP.get(annotation, {"type": "string"})


# ─── Главная функция: func → OpenAI tool schema ───────────────────────────────
def function_to_tool_schema(func: Callable) -> dict:
    """
    Конвертирует Python-функцию в OpenAI tool schema.
    Читает: аннотации типов, docstring (описание + параметры).
    """
    sig = inspect.signature(func)
    doc = parse_docstring(func.__doc__ or "")

    # Описания параметров из docstring
    param_docs = {p.arg_name: p.description for p in doc.params}

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        annotation = param.annotation
        has_default = param.default is not inspect.Parameter.empty

        # Тип параметра
        schema = (
            {}
            if annotation is inspect.Parameter.empty
            else python_type_to_json_schema(annotation)
        )

        # Описание из docstring
        if name in param_docs:
            schema["description"] = param_docs[name]

        # Значение по умолчанию
        if has_default and param.default is not None:
            schema["default"] = param.default

        properties[name] = schema

        # Обязательный параметр — нет default и не Optional
        origin = get_origin(annotation)
        args = get_args(annotation)
        is_optional = origin is Union and type(None) in args

        if not has_default and not is_optional:
            required.append(name)

    # Описание функции из docstring
    description = ""
    if doc.short_description:
        description = doc.short_description
    if doc.long_description:
        description += f"\n{doc.long_description}"

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ─── Утилита: список функций → список tools ───────────────────────────────────
def functions_to_tools(funcs: list[Callable]) -> list[dict]:
    """Конвертирует список функций в список OpenAI tools."""
    return [function_to_tool_schema(f) for f in funcs]


# ─── Декоратор @tool (опциональный) ──────────────────────────────────────────
def tool(func: Callable) -> Callable:
    """
    Декоратор-маркер. Добавляет .schema атрибут к функции.
    Использование: @tool
    """
    func.schema = function_to_tool_schema(func)
    func.is_tool = True
    return func


#!!!

# ─── Просто пишем функции как обычно! ────────────────────────────────────────


@tool
def get_weather(
    city: str,
    unit: Literal["celsius", "fahrenheit"] = "celsius",
    include_forecast: bool = False,
) -> dict:
    """
    Получить текущую погоду в указанном городе.

    Args:
        city: Название города (например, 'Москва', 'London')
        unit: Единица температуры — celsius или fahrenheit
        include_forecast: Включить прогноз на 5 дней
    """
    # Реальная логика...
    return {"city": city, "temperature": 22, "condition": "Sunny"}


@tool
def search_web(
    query: str,
    max_results: int = 5,
    language: Optional[str] = None,
) -> list[dict]:
    """
    Поиск информации в интернете.

    Args:
        query: Поисковый запрос
        max_results: Максимальное количество результатов (1-20)
        language: Язык результатов, например 'ru', 'en'. None — любой язык
    """
    return [
        {"title": f"Result {i}", "url": f"https://example.com/{i}"}
        for i in range(max_results)
    ]


@tool
def send_email(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    priority: Literal["low", "normal", "high"] = "normal",
) -> bool:
    """
    Отправить email-сообщение.

    Args:
        to: Email получателя
        subject: Тема письма
        body: Текст письма
        cc: Email для копии (необязательно)
        priority: Приоритет письма
    """
    print(f"📧 Отправка письма на {to}...")
    return True


@tool
def calculate(
    expression: str,
    precision: int = 2,
) -> dict:
    """
    Вычислить математическое выражение.

    Args:
        expression: Математическое выражение, например '2 + 2 * 10'
        precision: Количество знаков после запятой в результате
    """
    try:
        result = eval(expression)  # В продакшене используйте безопасный парсер!
        return {"expression": expression, "result": round(float(result), precision)}
    except Exception as e:
        return {"error": str(e)}


# ─── Смотрим что сгенерировалось ──────────────────────────────────────────────
if __name__ == "__main__":
    for func in [get_weather, search_web, send_email, calculate]:
        schema = func.schema
        fn = schema["function"]
        params = fn["parameters"]["properties"]
        required = fn["parameters"]["required"]

        print(f"\n🔧 {fn['name']}")
        print(f"   📝 {fn['description']}")
        for param_name, param_schema in params.items():
            req_mark = "✅" if param_name in required else "⬜"
            ptype = param_schema.get("type", "any")
            enum = param_schema.get("enum", [])
            default = param_schema.get("default", "—")
            desc = param_schema.get("description", "")
            enum_str = f" [{', '.join(map(str, enum))}]" if enum else ""
            print(f"   {req_mark} {param_name}: {ptype}{enum_str} = {default} | {desc}")

#!!

import json
from openai import OpenAI

client = OpenAI()


# ─── ToolRegistry — реестр всех инструментов ──────────────────────────────────
class ToolRegistry:
    """Хранит функции и их схемы. Вызывает нужную по имени."""

    def __init__(self):
        self._tools: dict[str, Callable] = {}

    def register(self, *funcs: Callable) -> "ToolRegistry":
        """Регистрирует функции в реестре."""
        for func in funcs:
            if not hasattr(func, "is_tool"):
                # Если забыли поставить @tool — регистрируем всё равно
                func = tool(func)
            self._tools[func.__name__] = func
        return self

    def get_schemas(self) -> list[dict]:
        """Возвращает список схем для OpenAI API."""
        return [f.schema for f in self._tools.values()]

    def call(self, name: str, arguments: str) -> str:
        """Вызывает функцию по имени с JSON-аргументами."""
        if name not in self._tools:
            return json.dumps({"error": f"Функция '{name}' не найдена"})
        try:
            args = json.loads(arguments)
            result = self._tools[name](**args)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def __repr__(self) -> str:
        return f"ToolRegistry({list(self._tools.keys())})"


# ─── Создаём реестр из наших функций ─────────────────────────────────────────
registry = ToolRegistry().register(
    get_weather,
    search_web,
    send_email,
    calculate,
)

print(
    registry
)  # ToolRegistry(['get_weather', 'search_web', 'send_email', 'calculate'])


# ─── Агентный цикл ────────────────────────────────────────────────────────────
def run_agent(user_message: str, verbose: bool = True) -> str:
    """Агент с автоматическим вызовом инструментов."""
    messages = [
        {
            "role": "system",
            "content": "Ты полезный ассистент с доступом к инструментам.",
        },
        {"role": "user", "content": user_message},
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=registry.get_schemas(),  # ← Схемы генерируются автоматически!
            tool_choice="auto",
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        messages.append(message)

        # Финальный ответ — возвращаем
        if finish_reason == "stop":
            return message.content

        # Вызов инструментов
        if finish_reason == "tool_calls":
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                arguments = tool_call.function.arguments

                if verbose:
                    args_preview = json.loads(arguments)
                    print(f"  🔧 {name}({args_preview})")

                # Вызываем через реестр — автоматический диспетчинг!
                result = registry.call(name, arguments)

                if verbose:
                    print(f"  ✅ Результат: {result[:80]}...")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )


# ─── Тесты ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("❓ Какая погода в Токио?")
print("=" * 55)
answer = run_agent("Какая погода сейчас в Токио?")
print(f"\n🤖 {answer}")

print("\n" + "=" * 55)
print("❓ Посчитай выражение и напиши результат")
print("=" * 55)
answer = run_agent("Посчитай (123 * 456) + (789 / 3) и объясни результат")
print(f"\n🤖 {answer}")

#!!

from pydantic import BaseModel, Field
from typing import Literal, Optional
import inspect, json


def pydantic_tool(model: type[BaseModel]):
    """
    Декоратор: принимает Pydantic-модель как описание аргументов.
    Функция получает уже провалидированный объект модели.
    """

    def decorator(func: Callable) -> Callable:
        schema = model.model_json_schema()

        # Убираем лишние поля из схемы Pydantic (title и т.д.)
        def clean_schema(s: dict) -> dict:
            return {k: v for k, v in s.items() if k not in ("title",)}

        func.schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        k: clean_schema(v)
                        for k, v in schema.get("properties", {}).items()
                    },
                    "required": schema.get("required", []),
                },
            },
        }
        func.is_tool = True
        func.arg_model = model

        # Обёртка: автоматически валидирует аргументы через Pydantic
        def wrapper(**kwargs):
            validated = model(**kwargs)  # Валидация!
            return func(validated)

        wrapper.__name__ = func.__name__
        wrapper.schema = func.schema
        wrapper.is_tool = True
        wrapper.arg_model = model
        return wrapper

    return decorator


# ─── Описываем аргументы как Pydantic-модель ──────────────────────────────────
class FlightSearchArgs(BaseModel):
    origin: str = Field(description="Код аэропорта вылета (IATA), например 'SVO'")
    destination: str = Field(
        description="Код аэропорта назначения (IATA), например 'JFK'"
    )
    date: str = Field(description="Дата вылета в формате YYYY-MM-DD")
    cabin: Literal["economy", "business", "first"] = Field(
        "economy", description="Класс обслуживания"
    )
    passengers: int = Field(1, ge=1, le=9, description="Количество пассажиров (1-9)")


@pydantic_tool(FlightSearchArgs)
def search_flights(args: FlightSearchArgs) -> dict:
    """Поиск доступных авиарейсов между двумя городами."""
    return {
        "flights": [
            {
                "flight": "SU 100",
                "from": args.origin,
                "to": args.destination,
                "date": args.date,
                "cabin": args.cabin,
                "price": 450 * args.passengers,
                "currency": "USD",
            }
        ]
    }


# ─── Проверяем схему ──────────────────────────────────────────────────────────
print(json.dumps(search_flights.schema, ensure_ascii=False, indent=2))
