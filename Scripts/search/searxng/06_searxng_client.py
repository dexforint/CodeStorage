import requests
import json
import time
from typing import Optional


class SearXNGClient:
    """
    Удобный клиент для работы с SearXNG API.
    Инкапсулирует все параметры и методы для поиска.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: int = 15,
        delay: float = 0.5,
    ):
        """
        Инициализация клиента.

        Параметры:
            base_url — адрес экземпляра SearXNG
            timeout  — таймаут запросов в секундах
            delay    — задержка между запросами (защита от перегрузки)
        """
        self.base_url = base_url.rstrip("/")
        self.search_url = f"{self.base_url}/search"
        self.timeout = timeout
        self.delay = delay
        self._last_request = 0.0  # Время последнего запроса

        # Стандартные заголовки для всех запросов
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; SearXNG-Python-Client/1.0)"
        }

    def _wait(self) -> None:
        """Выдерживает минимальную задержку между запросами."""
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()

    def search(
        self,
        query: str,
        categories: str = "general",
        engines: str = "",
        language: str = "ru-RU",
        time_range: str = "",
        safesearch: int = 0,
        pageno: int = 1,
    ) -> dict:
        """
        Универсальный метод поиска.
        Возвращает полный словарь ответа API.
        """
        self._wait()  # Соблюдаем задержку

        params = {
            "q": query,
            "format": "json",
            "categories": categories,
            "language": language,
            "safesearch": safesearch,
            "pageno": pageno,
        }
        if engines:
            params["engines"] = engines
        if time_range:
            params["time_range"] = time_range

        try:
            response = requests.get(
                self.search_url,
                params=params,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка запроса: {e}")
            return {}

    def search_news(self, query: str, time_range: str = "week") -> list:
        """Поиск новостей. Возвращает только список результатов."""
        data = self.search(query, categories="news", time_range=time_range)
        return data.get("results", [])

    def search_images(self, query: str) -> list:
        """Поиск изображений. Возвращает список URL изображений."""
        data = self.search(query, categories="images")
        results = data.get("results", [])
        # Для изображений доступно поле img_src — прямая ссылка на файл
        return [
            {"title": r.get("title"), "img_src": r.get("img_src"), "url": r.get("url")}
            for r in results
            if r.get("img_src")
        ]

    def search_it(self, query: str, engines: str = "") -> list:
        """IT-поиск: GitHub, StackOverflow, документация и т.д."""
        data = self.search(query, categories="it", engines=engines, language="en-US")
        return data.get("results", [])

    def get_suggestions(self, query: str) -> list:
        """Возвращает список предложений альтернативных запросов."""
        data = self.search(query)
        return data.get("suggestions", [])

    def is_available(self) -> bool:
        """Проверяет доступность экземпляра SearXNG."""
        try:
            r = requests.get(self.base_url, timeout=5, headers=self.headers)
            return r.status_code == 200
        except requests.exceptions.RequestException:
            return False


# ─────────────────────────────────────────────
# Точка входа — демонстрация всех методов
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Создаём экземпляр клиента
    client = SearXNGClient(base_url="http://localhost:8080")

    # Проверка доступности перед началом работы
    if not client.is_available():
        print("❌ SearXNG недоступен! Запустите его с помощью start.bat")
        exit(1)

    print("✅ SearXNG доступен!\n")

    # ── Тест 1: Общий поиск ──────────────────
    print("─" * 50)
    print("1️⃣  Общий поиск")
    data = client.search("нейронные сети", language="ru-RU")
    for r in data.get("results", [])[:3]:
        print(f"  • {r['title']}")
        print(f"    {r['url']}\n")

    # ── Тест 2: Поиск новостей ──────────────
    print("─" * 50)
    print("2️⃣  Свежие новости")
    news = client.search_news("технологии", time_range="week")
    for n in news[:3]:
        print(f"  📰 {n.get('title')}")

    # ── Тест 3: IT-поиск ────────────────────
    print("\n─" * 50)
    print("3️⃣  IT-поиск (GitHub)")
    it_results = client.search_it("FastAPI tutorial", engines="github")
    for r in it_results[:3]:
        print(f"  💻 {r.get('title')}")
        print(f"     {r.get('url')}\n")

    # ── Тест 4: Подсказки ───────────────────
    print("─" * 50)
    print("4️⃣  Похожие запросы")
    suggestions = client.get_suggestions("Python")
    print(f"  💡 {', '.join(suggestions[:5])}")
