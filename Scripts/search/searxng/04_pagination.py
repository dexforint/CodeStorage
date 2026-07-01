import requests
import time

SEARXNG_URL = "http://localhost:8080"
SEARCH_ENDPOINT = f"{SEARXNG_URL}/search"


def search_page(query: str, page: int) -> dict:
    """Получает одну страницу результатов поиска."""
    params = {
        "q": query,
        "format": "json",
        "pageno": page,  # Номер страницы (начиная с 1)
    }
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SearXNG-Python-Client/1.0)"}

    try:
        r = requests.get(SEARCH_ENDPOINT, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка на странице {page}: {e}")
        return {}


def collect_all_results(query: str, max_pages: int = 3) -> list:
    """
    Собирает результаты с нескольких страниц.

    Параметры:
        query     — поисковый запрос
        max_pages — максимальное количество страниц для сбора
    """
    all_results = []

    for page in range(1, max_pages + 1):
        print(f"  📄 Загрузка страницы {page}/{max_pages}...")
        data = search_page(query, page)
        page_results = data.get("results", [])

        if not page_results:
            # Если страница пустая — дальше нет смысла идти
            print(f"  ⚠️  Страница {page} пустая, останавливаемся.")
            break

        all_results.extend(page_results)

        # Пауза между запросами — важна, чтобы не перегружать сервер
        if page < max_pages:
            time.sleep(1.0)

    return all_results


if __name__ == "__main__":
    query = "open source software"
    print(f"📚 Сбор результатов для: «{query}»")

    results = collect_all_results(query, max_pages=3)
    print(f"\n✅ Всего собрано: {len(results)} результатов\n")

    # Выводим первые 5 для проверки
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. {r.get('title')}")
        print(f"     {r.get('url')}\n")
