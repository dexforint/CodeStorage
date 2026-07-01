import requests
from datetime import timedelta

SEARXNG_URL = "http://localhost:8080"
SEARCH_ENDPOINT = f"{SEARXNG_URL}/search"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SearXNG-Python-Client/1.0)"}


def search_videos(
    query: str,
    engines: str = "youtube,dailymotion,vimeo",
    language: str = "ru-RU",
    time_range: str = "",
    pageno: int = 1,
) -> list:
    """
    Поиск видео через SearXNG API.

    Поля результата:
        title        — название видео
        url          — ссылка на видео
        content      — описание
        thumbnail    — URL превью
        length       — длительность (строка или секунды)
        publishedDate— дата публикации
        iframe_src   — embed-ссылка для встраивания (если есть)
        engine       — движок-источник (youtube, dailymotion и т.д.)
    """
    params = {
        "q": query,
        "categories": "videos",  # Категория поиска видео
        "format": "json",
        "engines": engines,
        "language": language,
        "pageno": pageno,
    }

    if time_range:
        params["time_range"] = time_range

    try:
        response = requests.get(
            SEARCH_ENDPOINT, params=params, headers=HEADERS, timeout=15
        )
        response.raise_for_status()
        return response.json().get("results", [])

    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка: {e}")
        return []


def format_duration(length) -> str:
    """
    Форматирует длительность видео в читаемый вид.
    length может быть строкой '10:35' или числом секунд.
    """
    if not length:
        return "неизвестно"

    # Если уже строка — возвращаем как есть
    if isinstance(length, str):
        return length

    # Если число секунд — конвертируем
    try:
        td = timedelta(seconds=int(length))
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"
    except (ValueError, TypeError):
        return str(length)


def print_video_results(results: list) -> None:
    """Красиво выводит результаты поиска видео."""
    print(f"\n🎬 Найдено видео: {len(results)}\n")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        title = r.get("title", "Без названия")
        url = r.get("url", "")
        content = r.get("content", "")[:100]
        length = format_duration(r.get("length"))
        pub_date = r.get("publishedDate", "неизвестно")
        engine = r.get("engine", "неизвестно")
        thumbnail = r.get("thumbnail", "")

        print(f"[{i}] {title}")
        print(f"    🔗 URL      : {url}")
        print(f"    ⏱️  Длина    : {length}")
        print(f"    📅 Дата     : {pub_date}")
        print(f"    ⚙️  Движок   : {engine}")
        if content:
            print(f"    📄 Описание : {content}...")
        if thumbnail:
            print(f"    🖼️  Превью   : {thumbnail}")
        print()


def get_embed_links(results: list) -> list:
    """
    Извлекает embed-ссылки из результатов поиска видео.
    Полезно для встраивания видео в веб-страницы.
    """
    embeds = []
    for r in results:
        iframe = r.get("iframe_src")
        if iframe:
            embeds.append(
                {
                    "title": r.get("title"),
                    "iframe_src": iframe,
                    "url": r.get("url"),
                }
            )
    return embeds


if __name__ == "__main__":
    # Пример 1: Поиск обучающих видео на YouTube
    print("🎓 Пример 1: Обучающие видео по Python")
    results = search_videos(
        query="Python tutorial for beginners", engines="youtube", language="en-US"
    )
    print_video_results(results[:5])

    # Пример 2: Поиск свежих новостных видео
    print("📰 Пример 2: Свежие новости (последняя неделя)")
    results = search_videos(
        query="технологические новости",
        engines="youtube,dailymotion",
        language="ru-RU",
        time_range="week",
    )
    print_video_results(results[:3])

    # Пример 3: Получение embed-ссылок
    print("🔗 Пример 3: Embed-ссылки для встраивания")
    results = search_videos("music concert live", engines="youtube,vimeo")
    embeds = get_embed_links(results)
    for e in embeds[:3]:
        print(f"  📺 {e['title']}")
        print(f"     {e['iframe_src']}\n")
