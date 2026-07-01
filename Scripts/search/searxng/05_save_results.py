import requests
import json
import csv
import os
from datetime import datetime

SEARXNG_URL = "http://localhost:8080"
SEARCH_ENDPOINT = f"{SEARXNG_URL}/search"


def fetch_results(query: str, **params) -> list:
    """Получает список результатов для заданного запроса."""
    default_params = {"q": query, "format": "json"}
    default_params.update(params)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; SearXNG-Python-Client/1.0)"}

    try:
        r = requests.get(
            SEARCH_ENDPOINT, params=default_params, headers=headers, timeout=10
        )
        r.raise_for_status()
        return r.json().get("results", [])
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка: {e}")
        return []


def save_to_json(results: list, filename: str) -> None:
    """Сохраняет результаты в JSON-файл с отступами для читаемости."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON сохранён: {filename} ({len(results)} записей)")


def save_to_csv(results: list, filename: str) -> None:
    """
    Сохраняет результаты в CSV-файл.
    Поля: title, url, content, engine, publishedDate.
    """
    # Определяем заголовки столбцов
    fieldnames = ["title", "url", "content", "engine", "publishedDate"]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()  # Записываем строку заголовков

        for r in results:
            # Извлекаем только нужные поля, остальные заполняем пустой строкой
            writer.writerow(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "").replace("\n", " "),
                    "engine": r.get("engine", ""),
                    "publishedDate": r.get("publishedDate", ""),
                }
            )

    print(f"📊 CSV сохранён: {filename} ({len(results)} записей)")


if __name__ == "__main__":
    query = "Python web scraping"

    # Создаём папку для результатов, если её нет
    os.makedirs("search_results", exist_ok=True)

    # Метка времени для уникальных имён файлов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"🔎 Поиск: «{query}»")
    results = fetch_results(query, categories="it", language="en-US")

    if results:
        # Сохраняем в оба формата
        save_to_json(results, f"search_results/{timestamp}_results.json")
        save_to_csv(results, f"search_results/{timestamp}_results.csv")
    else:
        print("⚠️  Результаты не получены.")
