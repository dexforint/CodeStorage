import requests


def get_ip_and_location():
    try:
        response = requests.get("http://ip-api.com/json/", timeout=10)
        data = response.json()

        if data["status"] == "success":
            print(f"{'='*40}")
            print(f"  IP адрес:       {data['query']}")
            print(f"{'='*40}")
            print(f"  Страна:         {data['country']}")
            print(f"  Регион:         {data['regionName']}")
            print(f"  Город:          {data['city']}")
            print(f"  Почтовый код:   {data['zip']}")
            print(f"  Широта:         {data['lat']}")
            print(f"  Долгота:        {data['lon']}")
            print(f"  Часовой пояс:   {data['timezone']}")
            print(f"  Провайдер:      {data['isp']}")
            print(f"  Организация:    {data['org']}")
            print(f"{'='*40}")
        else:
            print("Не удалось определить местоположение.")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка сети: {e}")


if __name__ == "__main__":
    get_ip_and_location()
