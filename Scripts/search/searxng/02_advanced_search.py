import requests

SEARXNG_URL = "http://localhost:8080"
SEARCH_ENDPOINT = f"{SEARXNG_URL}/search"


def advanced_search(
    query: str,
    categories: str = "general",
    engines: str = "",
    language: str = "ru-RU",
    time_range: str = "",
    safesearch: int = 0,
    pageno: int = 1,
) -> dict:
    """
    Расширенный поиск с поддержкой всех параметров SearXNG API.

    Параметры:
        query      — поисковый запрос
        categories — категория поиска: general, news, images, videos,
                     science, it, map, music, social media, files
        engines    — список движков через запятую: google,bing,duckduckgo
        language   — язык результатов: ru-RU, en-US, de-DE и т.д.
        time_range — фильтр по времени: day, week, month, year
        safesearch — фильтрация контента: 0 = выкл, 1 = умеренно, 2 = строго
        pageno     — номер страницы результатов (начиная с 1)
    """
    params = {
        "q": query,
        "format": "json",
        "categories": categories,
        "language": language,
        "safesearch": safesearch,
        "pageno": pageno,
    }

    # Добавляем необязательные параметры, только если они заданы
    if engines:
        params["engines"] = engines
    if time_range:
        params["time_range"] = time_range

    headers = {"User-Agent": "Mozilla/5.0 (compatible; SearXNG-Python-Client/1.0)"}

    try:
        response = requests.get(
            SEARCH_ENDPOINT, params=params, headers=headers, timeout=15
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка запроса: {e}")
        return {}


def print_full_response(data: dict) -> None:
    """
    Выводит полный разбор JSON-ответа SearXNG:
    — основные результаты
    — предложения (suggestions)
    — прямые ответы (answers)
    — инфобоксы (infoboxes)
    — неответившие движки (unresponsive_engines)
    """
    if not data:
        return

    # ── Основные результаты ──────────────────
    results = data.get("results", [])
    print(f"\n📋 Основные результаты ({len(results)} шт.):")
    print("-" * 50)
    for i, r in enumerate(results[:5], 1):  # Показываем первые 5
        print(f"  {i}. {r.get('title', 'N/A')}")
        print(f"     URL: {r.get('url', 'N/A')}")
        # Дата публикации (если доступна)
        if r.get("publishedDate"):
            print(f"     📅 Дата: {r['publishedDate']}")
        # Релевантность — числовой балл от движка
        if r.get("score"):
            print(f"     ⭐ Рейтинг: {r['score']:.2f}")
        print()

    # ── Предложения альтернативных запросов ──
    suggestions = data.get("suggestions", [])
    if suggestions:
        print(f"\n💡 Похожие запросы:")
        for s in suggestions:
            print(f"   → {s}")

    # ── Прямые ответы (например, калькулятор) ──
    answers = data.get("answers", [])
    if answers:
        print(f"\n✅ Прямые ответы:")
        for a in answers:
            print(f"   ➜ {a}")

    # ── Инфобоксы (карточки знаний, Wikipedia) ──
    infoboxes = data.get("infoboxes", [])
    if infoboxes:
        print(f"\n📦 Инфобоксы ({len(infoboxes)} шт.):")
        for box in infoboxes:
            print(f"   📌 {box.get('infobox', 'N/A')}: {box.get('content', '')[:100]}")

    # ── Исправления запроса ──
    corrections = data.get("corrections", [])
    if corrections:
        print(f"\n🔤 Исправления запроса:")
        for c in corrections:
            print(f"   ✏️  {c}")

    # ── Движки, не ответившие на запрос ──
    unresponsive = data.get("unresponsive_engines", [])
    if unresponsive:
        print(f"\n⚠️  Не ответили: {', '.join(unresponsive)}")


# ─────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Пример 1: поиск новостей за последнюю неделю
    print("=" * 60)
    print("📰 Пример 1: Поиск свежих новостей")
    data = advanced_search(
        query="искусственный интеллект",
        categories="news",
        language="ru-RU",
        time_range="week",
    )
    print_full_response(data)

    # Пример 2: IT-поиск по конкретным движкам
    print("\n" + "=" * 60)
    print("💻 Пример 2: IT-поиск по GitHub и StackOverflow")
    data = advanced_search(
        query="Python async await tutorial",
        categories="it",
        engines="github,stackoverflow",
        language="en-US",
    )
    print_full_response(data)
