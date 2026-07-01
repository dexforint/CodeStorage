import requests
import base64
import os

SEARXNG_URL = "http://localhost:8080"
SEARCH_ENDPOINT = f"{SEARXNG_URL}/search"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SearXNG-Python-Client/1.0)"}


def reverse_image_search_by_url(image_url: str, pageno: int = 1) -> dict:
    """
    Обратный поиск изображения по его URL через TinEye (в SearXNG).

    TinEye — движок типа online_url_search: принимает URL изображения,
    ищет по всему интернету где ещё встречается это изображение.

    Параметры:
        image_url — публичная ссылка на изображение
        pageno    — страница результатов

    Возвращает полный ответ API со списком совпадений.
    """
    params = {
        "q": image_url,  # URL изображения — это и есть запрос
        "engines": "tineye",  # Используем движок TinEye
        "format": "json",
        "pageno": pageno,
    }

    try:
        response = requests.get(
            SEARCH_ENDPOINT, params=params, headers=HEADERS, timeout=20
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка: {e}")
        return {}


def upload_and_search(image_path: str) -> dict:
    """
    Обратный поиск по ЛОКАЛЬНОМУ файлу изображения.

    Поскольку TinEye через SearXNG принимает только URL,
    мы конвертируем файл в base64 Data URL — это позволяет
    передать изображение напрямую без внешнего хостинга.

    Поддерживаемые форматы: JPEG, PNG, GIF, BMP, TIFF, WebP.
    """
    if not os.path.exists(image_path):
        print(f"❌ Файл не найден: {image_path}")
        return {}

    # Определяем MIME-тип по расширению файла
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "bmp": "image/bmp",
        "webp": "image/webp",
        "tiff": "image/tiff",
    }
    mime = mime_types.get(ext, "image/jpeg")

    # Читаем файл и конвертируем в base64 Data URL
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    data_url = f"data:{mime};base64,{image_data}"

    print(f"  📦 Изображение конвертировано в Data URL ({len(data_url)//1024} KB)")

    # Отправляем POST-запрос с Data URL
    params = {
        "q": data_url,
        "engines": "tineye",
        "format": "json",
    }

    try:
        response = requests.get(
            SEARCH_ENDPOINT, params=params, headers=HEADERS, timeout=30
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка загрузки: {e}")
        return {}


def print_tineye_results(data: dict) -> None:
    """
    Выводит результаты обратного поиска TinEye.

    Поля каждого результата:
        url       — ссылка на страницу с изображением
        img_src   — прямая ссылка на найденное изображение
        score     — процент совпадения (0–100)
        domain    — домен источника
        width     — ширина найденного изображения
        height    — высота найденного изображения
        filesize  — размер файла в байтах
        format    — формат (JPEG, PNG и т.д.)
        backlinks — список страниц, использующих изображение
    """
    results = data.get("results", [])

    if not results:
        print("❌ Совпадений не найдено (изображение уникально или TinEye недоступен).")
        return

    print(f"\n🔍 Найдено совпадений: {len(results)}\n")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        url = r.get("url", "")
        img_src = r.get("img_src", "")
        score = r.get("score", 0)
        domain = r.get("domain", "")
        width = r.get("img_format", {}).get("width", r.get("width", "?"))
        height = r.get("img_format", {}).get("height", r.get("height", "?"))
        filesize = r.get("img_format", {}).get("filesize", r.get("filesize", "?"))
        fmt = r.get("img_format", {}).get("format", r.get("format", "?"))
        crawl_date = r.get("publishedDate", "")

        print(f"[{i}] 🌐 {domain}")
        print(f"     🔗 Страница    : {url}")
        print(f"     🖼️  Изображение : {img_src}")
        print(f"     ⭐ Совпадение  : {score}%")
        print(f"     📐 Размер      : {width} × {height} px | {fmt}")
        if filesize and filesize != "?":
            size_kb = int(filesize) // 1024 if str(filesize).isdigit() else filesize
            print(f"     💾 Файл        : {size_kb} KB")
        if crawl_date:
            print(f"     📅 Дата индекс.: {crawl_date}")

        # Выводим обратные ссылки (сайты, использующие изображение)
        backlinks = r.get("backlinks", [])
        if backlinks:
            print(f"     🔗 Backlinks ({len(backlinks)}):")
            for bl in backlinks[:3]:  # Показываем не более 3
                bl_url = bl.get("backlink", bl.get("url", ""))
                bl_date = bl.get("crawl_date", "")
                print(f"        • {bl_url}")
                if bl_date:
                    print(f"          📅 {bl_date}")
        print()

    # Статистика
    answers = data.get("answers", [])
    if answers:
        print(f"📊 Статистика TinEye: {answers}")


def multi_engine_reverse_search(image_url: str) -> None:
    """
    Генерирует ссылки для обратного поиска сразу в нескольких системах:
    Google Images, Bing, Yandex.

    Полезно когда хочется проверить изображение сразу во всех движках.
    """
    import urllib.parse

    encoded_url = urllib.parse.quote(image_url, safe="")

    print(f"\n🔗 Ссылки для ручного обратного поиска: «{image_url[:60]}...»\n")

    links = {
        "TinEye (через SearXNG)": f"{SEARXNG_URL}/search?q={encoded_url}&engines=tineye",
        "Google Images": f"https://www.google.com/searchbyimage?image_url={encoded_url}",
        "Bing Visual Search": f"https://www.bing.com/images/searchbyimage?imgurl={encoded_url}",
        "Yandex Images": f"https://yandex.ru/images/search?url={encoded_url}&rpt=imageview",
    }

    for engine, link in links.items():
        print(f"  🔍 {engine}:")
        print(f"     {link}\n")


if __name__ == "__main__":
    # Пример 1: Обратный поиск по URL известного изображения
    test_image_url = (
        "https://i.pinimg.com/originals/6d/3a/68/6d3a68efe558025a1c797ea91fe5c6ca.png"
    )

    print("🔍 Пример 1: Обратный поиск по URL (TinEye через SearXNG)")
    print(f"   URL: {test_image_url}")
    data = reverse_image_search_by_url(test_image_url)
    print_tineye_results(data)

    # Пример 2: Генерация ссылок для всех поисковиков
    print("🔍 Пример 2: Ссылки для поиска во всех системах")
    multi_engine_reverse_search(test_image_url)

    # Пример 3: Поиск по локальному файлу (раскомментируйте если нужно)
    # print("🔍 Пример 3: Поиск по локальному файлу")
    # data = upload_and_search("C:/my_image.jpg")
    # print_tineye_results(data)
