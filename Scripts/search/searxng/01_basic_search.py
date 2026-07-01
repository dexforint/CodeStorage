import requests
import json

# ─────────────────────────────────────────────
# Базовая конфигурация
# ─────────────────────────────────────────────
SEARXNG_URL = "http://localhost:8080"  # Адрес локального экземпляра SearXNG
SEARCH_ENDPOINT = f"{SEARXNG_URL}/search"


def basic_search(query: str) -> dict:
    """
    Выполняет простой поисковый запрос к SearXNG API.
    Возвращает словарь с результатами поиска.
    """
    # Параметры GET-запроса
    params = {
        "q": query,  # Поисковый запрос
        "format": "json",  # Формат ответа — JSON
    }

    # Заголовки запроса, имитирующие обычный браузер
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        # Выполняем GET-запрос к API
        response = requests.get(
            SEARCH_ENDPOINT,
            params=params,
            headers=headers,
            timeout=10,  # Таймаут 10 секунд
        )

        # Проверяем, что сервер вернул успешный статус (200 OK)
        response.raise_for_status()

        # Парсим JSON-ответ и возвращаем его
        return response.json()

    except requests.exceptions.ConnectionError:
        print("❌ Ошибка подключения. Убедитесь, что SearXNG запущен!")
        return {}
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP ошибка: {e}")
        print("   Проверьте, что в settings.yml включён JSON формат.")
        return {}
    except requests.exceptions.Timeout:
        print("❌ Превышено время ожидания ответа.")
        return {}


def print_results(data: dict) -> None:
    """Красиво выводит результаты поиска в консоль."""
    if not data:
        print("Нет данных для отображения.")
        return

    # Список основных результатов поиска
    results = data.get("results", [])
    print(f"\n🔍 Найдено результатов: {len(results)}\n")
    print("=" * 60)

    for i, result in enumerate(results, start=1):
        # Каждый результат содержит: title, url, content
        title = result.get("title", "Без заголовка")
        url = result.get("url", "")
        snippet = result.get("content", "Описание отсутствует")
        engine = result.get("engine", "неизвестно")  # Поисковик-источник

        print(f"[{i}] {title}")
        print(f"    🔗 {url}")
        print(f"    📄 {snippet[:120]}...")  # Ограничиваем вывод сниппета
        print(f"    ⚙️  Источник: {engine}")
        print()


# ─────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────
if __name__ == "__main__":
    query = "Python программирование"
    print(f"🔎 Поиск: «{query}»")

    data = basic_search(query)
    print_results(data)
