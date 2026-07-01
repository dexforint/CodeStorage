import requests
import os
import urllib.request

SEARXNG_URL = "http://localhost:8080"
SEARCH_ENDPOINT = f"{SEARXNG_URL}/search"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SearXNG-Python-Client/1.0)"}


def search_images(
    query: str,
    engines: str = "google images,bing images,duckduckgo images",
    safesearch: int = 1,
    pageno: int = 1,
) -> list:
    """
    Поиск изображений через SearXNG API.

    Возвращает список словарей с полями:
        title        — заголовок/описание
        url          — страница-источник изображения
        img_src      — прямая ссылка на файл изображения
        thumbnail_src— ссылка на миниатюру
        resolution   — разрешение: '1920 x 1080'
        engine       — движок-источник
    """
    params = {
        "q": query,
        "categories": "images",  # Категория поиска изображений
        "format": "json",
        "engines": engines,
        "safesearch": safesearch,
        "pageno": pageno,
    }

    try:
        response = requests.get(
            SEARCH_ENDPOINT, params=params, headers=HEADERS, timeout=15
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])

    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка: {e}")
        return []


def print_image_results(results: list) -> None:
    """Выводит результаты поиска изображений."""
    print(f"\n🖼️  Найдено изображений: {len(results)}\n")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        title = r.get("title", "Без названия")
        img_src = r.get("img_src", "")
        resolution = r.get("resolution", "неизвестно")
        engine = r.get("engine", "неизвестно")
        source_url = r.get("url", "")

        print(f"[{i}] {title}")
        print(f"    📐 Разрешение : {resolution}")
        print(f"    🔗 Источник   : {source_url}")
        print(f"    🖼️  Изображение: {img_src}")
        print(f"    ⚙️  Движок     : {engine}")
        print()


def download_images(
    results: list, folder: str = "downloaded_images", max_count: int = 5
) -> None:
    """
    Скачивает изображения из результатов поиска.

    Параметры:
        results   — список результатов от search_images()
        folder    — папка для сохранения файлов
        max_count — максимальное количество изображений
    """
    os.makedirs(folder, exist_ok=True)

    downloaded = 0
    for i, r in enumerate(results):
        if downloaded >= max_count:
            break

        img_url = r.get("img_src", "")
        title = r.get("title", f"image_{i}")

        if not img_url:
            continue

        # Определяем расширение файла из URL
        ext = img_url.split("?")[0].rsplit(".", 1)[-1]
        if ext.lower() not in ("jpg", "jpeg", "png", "gif", "webp", "bmp"):
            ext = "jpg"  # Расширение по умолчанию

        # Формируем безопасное имя файла
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:40]
        filename = os.path.join(folder, f"{i+1:02d}_{safe_name}.{ext}")

        try:
            # Скачиваем изображение
            urllib.request.urlretrieve(img_url, filename)
            print(f"  ✅ Скачано: {filename}")
            downloaded += 1
        except Exception as e:
            print(f"  ⚠️  Не удалось скачать [{i+1}]: {e}")

    print(f"\n📁 Всего скачано: {downloaded} изображений в папку '{folder}'")


if __name__ == "__main__":
    query = "northern lights landscape photography"

    print(f"🔎 Поиск изображений: «{query}»")
    results = search_images(query, safesearch=1)
    print_image_results(results)

    # Скачиваем первые 5 изображений
    if results:
        print("⬇️  Скачиваем изображения...")
        download_images(results, max_count=5)
