import requests

# Nominatim — официальный геокодер OpenStreetMap
NOMINATIM_URL = "https://nominatim.openstreetmap.org"

HEADERS = {
    # Nominatim требует указать корректный User-Agent с контактом
    "User-Agent": "SearXNG-Python-Client/1.0 (your_email@example.com)"
}


def geocode(query: str, limit: int = 5) -> list:
    """
    Прямое геокодирование: адрес/название → координаты.
    Например: 'Эйфелева башня' → [48.8584, 2.2945]
    """
    params = {
        "q": query,
        "format": "json",
        "limit": limit,
        "addressdetails": 1,  # Включить детали адреса
        "extratags": 1,  # Доп. теги (телефон, сайт и т.д.)
        "namedetails": 1,  # Названия на разных языках
    }

    try:
        response = requests.get(
            f"{NOMINATIM_URL}/search", params=params, headers=HEADERS, timeout=10
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка геокодирования: {e}")
        return []


def reverse_geocode(lat: float, lon: float) -> dict:
    """
    Обратное геокодирование: координаты → адрес.
    Например: [48.8584, 2.2945] → 'Эйфелева башня, Париж, Франция'
    """
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "addressdetails": 1,
        "zoom": 18,  # Уровень детализации (18 = здание)
    }

    try:
        response = requests.get(
            f"{NOMINATIM_URL}/reverse", params=params, headers=HEADERS, timeout=10
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка обратного геокодирования: {e}")
        return {}


def search_nearby(query: str, lat: float, lon: float, radius_km: float = 5.0) -> list:
    """
    Поиск объектов рядом с заданной точкой.

    Параметры:
        query     — что ищем (например, 'кафе', 'музей', 'аптека')
        lat, lon  — координаты центра поиска
        radius_km — радиус поиска в километрах
    """
    # Конвертируем радиус в градусы (приблизительно: 1° ≈ 111 км)
    delta = radius_km / 111.0

    # Формируем ограничивающий прямоугольник (bounding box)
    viewbox = f"{lon - delta},{lat + delta},{lon + delta},{lat - delta}"

    params = {
        "q": query,
        "format": "json",
        "limit": 10,
        "viewbox": viewbox,
        "bounded": 1,  # Искать только в рамках viewbox
        "addressdetails": 1,
    }

    try:
        response = requests.get(
            f"{NOMINATIM_URL}/search", params=params, headers=HEADERS, timeout=10
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка: {e}")
        return []


def print_geo_results(results: list) -> None:
    """Выводит результаты геопоиска."""
    print(f"\n📍 Найдено мест: {len(results)}\n")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        name = r.get("display_name", "Без названия")
        lat = r.get("lat", "?")
        lon = r.get("lon", "?")
        place_type = r.get("type", "")
        place_class = r.get("class", "")
        importance = float(r.get("importance", 0))

        # Извлекаем адресные компоненты
        address = r.get("address", {})
        city = address.get("city") or address.get("town") or address.get("village", "")
        country = address.get("country", "")

        # OpenStreetMap ссылка
        osm_type = r.get("osm_type", "")
        osm_id = r.get("osm_id", "")
        osm_link = (
            f"https://www.openstreetmap.org/{osm_type}/{osm_id}" if osm_id else ""
        )

        print(f"[{i}] 📍 {name[:80]}")
        print(f"     🌐 Координаты  : {lat}, {lon}")
        print(f"     🏷️  Тип         : {place_class} / {place_type}")
        print(f"     🏙️  Город/страна: {city}, {country}")
        print(f"     ⭐ Важность    : {importance:.4f}")
        if osm_link:
            print(f"     🗺️  На карте    : {osm_link}")
        print()


if __name__ == "__main__":
    # Пример 1: Найти Эйфелеву башню
    print("🗺️  Пример 1: Геокодирование — Эйфелева башня")
    results = geocode("Эйфелева башня Париж")
    print_geo_results(results[:3])

    # Пример 2: Обратное геокодирование — что находится в центре Москвы?
    print("🗺️  Пример 2: Обратное геокодирование — центр Москвы")
    # Координаты Красной площади
    result = reverse_geocode(lat=55.7539, lon=37.6208)
    if result:
        print(f"  📍 {result.get('display_name', 'N/A')}")
        addr = result.get("address", {})
        print(f"  🏙️  Улица  : {addr.get('road', 'N/A')}")
        print(f"  🌆 Город  : {addr.get('city', 'N/A')}")
        print(f"  🌍 Страна : {addr.get('country', 'N/A')}")

    # Пример 3: Найти кафе рядом с Эйфелевой башней (радиус 1 км)
    print("\n🗺️  Пример 3: Кафе рядом с Эйфелевой башней (1 км)")
    nearby = search_nearby(query="кафе", lat=48.8584, lon=2.2945, radius_km=1.0)
    print_geo_results(nearby[:5])
