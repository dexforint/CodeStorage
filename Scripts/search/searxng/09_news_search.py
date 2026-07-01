import requests
from datetime import datetime

SEARXNG_URL = "http://localhost:8080"
SEARCH_ENDPOINT = f"{SEARXNG_URL}/search"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SearXNG-Python-Client/1.0)"}


def search_news(
    query: str,
    engines: str = "google news,bing news,duckduckgo news",
    language: str = "ru-RU",
    time_range: str = "day",
    pageno: int = 1,
) -> list:
    """
    Поиск новостей через SearXNG API.

    Поля результата:
        title        — заголовок новости
        url          — ссылка на статью
        content      — краткое содержание
        publishedDate— дата/время публикации (ISO 8601)
        engine       — источник новости
        thumbnail    — превью (если есть)
    """
    params = {
        "q": query,
        "categories": "news",  # Категория новостей
        "format": "json",
        "engines": engines,
        "language": language,
        "time_range": time_range,
        "pageno": pageno,
    }

    try:
        response = requests.get(
            SEARCH_ENDPOINT, params=params, headers=HEADERS, timeout=15
        )
        response.raise_for_status()
        return response.json().get("results", [])

    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка: {e}")
        return []


def format_date(date_str: str) -> str:
    """Форматирует дату публикации в читаемый вид."""
    if not date_str:
        return "дата неизвестна"
    try:
        # Парсим дату в формате ISO 8601
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%d.%m.%Y %H:%M")
    except (ValueError, AttributeError):
        return date_str


def print_news(results: list, max_items: int = 10) -> None:
    """Выводит новости в удобном формате."""
    print(f"\n📰 Найдено новостей: {len(results)}\n")
    print("=" * 60)

    for i, r in enumerate(results[:max_items], 1):
        title = r.get("title", "Без заголовка")
        url = r.get("url", "")
        content = r.get("content", "")[:150]
        pub_date = format_date(r.get("publishedDate", ""))
        engine = r.get("engine", "")

        print(f"[{i}] 📰 {title}")
        print(f"     📅 {pub_date}  |  ⚙️  {engine}")
        print(f"     🔗 {url}")
        if content:
            print(f"     📄 {content}...")
        print()


def get_news_digest(topics: list, time_range: str = "day") -> dict:
    """
    Собирает дайджест новостей по нескольким темам сразу.

    Параметры:
        topics     — список тем для поиска
        time_range — период: 'day', 'week', 'month'
    Возвращает:
        словарь {тема: список_новостей}
    """
    import time

    digest = {}

    for topic in topics:
        print(f"  🔍 Загружаем новости по теме: «{topic}»...")
        news = search_news(topic, time_range=time_range)
        digest[topic] = news[:5]  # Берём топ-5 по каждой теме
        time.sleep(0.5)  # Пауза между запросами

    return digest


if __name__ == "__main__":
    # Пример 1: Поиск свежих новостей за день
    print("📡 Пример 1: Новости за сегодня")
    results = search_news(
        query="искусственный интеллект", time_range="day", language="ru-RU"
    )
    print_news(results, max_items=5)

    # Пример 2: Технические новости за неделю
    print("💻 Пример 2: Тех-новости за неделю")
    results = search_news(
        query="Python programming",
        engines="google news,bing news",
        language="en-US",
        time_range="week",
    )
    print_news(results, max_items=3)

    # Пример 3: Дайджест по нескольким темам
    print("\n📋 Пример 3: Ежедневный дайджест")
    topics = ["технологии", "наука", "космос"]
    digest = get_news_digest(topics, time_range="day")

    for topic, news_list in digest.items():
        print(f"\n🗂️  Тема: {topic} ({len(news_list)} новостей)")
        for n in news_list:
            pub_date = format_date(n.get("publishedDate", ""))
            print(f"  • [{pub_date}] {n.get('title', 'N/A')}")
