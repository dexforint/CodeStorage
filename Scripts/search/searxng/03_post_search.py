import requests

SEARXNG_URL = "http://localhost:8080"
SEARCH_ENDPOINT = f"{SEARXNG_URL}/search"


def post_search(query: str, **kwargs) -> dict:
    """
    Выполняет поиск через POST-запрос.
    POST отправляет данные в теле запроса как form data,
    а не в строке URL — удобно для длинных или сложных запросов.
    """
    # Данные формы для POST-запроса
    form_data = {
        "q": query,
        "format": "json",
        **kwargs,  # Дополнительные параметры (categories, language и т.д.)
    }

    headers = {
        # Для POST важно указать правильный Content-Type
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0 (compatible; SearXNG-Python-Client/1.0)",
    }

    try:
        # Отправляем POST-запрос
        response = requests.post(
            SEARCH_ENDPOINT,
            data=form_data,  # Передаём как form data, не JSON!
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка POST-запроса: {e}")
        return {}


if __name__ == "__main__":
    print("📤 Поиск через POST-запрос")
    data = post_search(
        "machine learning", categories="science", language="en-US", time_range="month"
    )

    results = data.get("results", [])
    print(f"\n✅ Получено результатов: {len(results)}")
    for r in results[:3]:
        print(f"  • {r.get('title')}")
        print(f"    {r.get('url')}\n")
