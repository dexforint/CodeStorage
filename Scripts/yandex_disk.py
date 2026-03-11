import requests
import json

import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("YANDEX_APP_TOKEN")

BASE_URL = "https://cloud-api.yandex.net/v1/disk"

HEADERS = {"Authorization": f"OAuth {TOKEN}"}

# Все пути начинаются с "app:/" — это корень папки приложения
# Физически на диске: "Приложения/<имя_приложения>/"
APP_ROOT = "app:/"


def list_files(path=APP_ROOT):
    """Получить список файлов и папок."""
    r = requests.get(
        f"{BASE_URL}/resources", headers=HEADERS, params={"path": path, "limit": 100}
    )
    r.raise_for_status()
    data = r.json()

    items = data.get("_embedded", {}).get("items", [])
    for item in items:
        icon = "📁" if item["type"] == "dir" else "📄"
        size = item.get("size", "")
        print(f"  {icon} {item['name']}  {size}")

    return items


def delete(remote_path, permanently=False):
    """Удалить файл или папку."""
    r = requests.delete(
        f"{BASE_URL}/resources",
        headers=HEADERS,
        params={"path": remote_path, "permanently": str(permanently).lower()},
    )
    r.raise_for_status()
    print(f"Удалён: {remote_path}")


def upload_file(local_path, remote_path=None):
    """Загрузить файл в папку приложения."""
    if remote_path is None:
        filename = os.path.basename(local_path)
        remote_path = f"app:/{filename}"

    # 1. Получить URL для загрузки
    r = requests.get(
        f"{BASE_URL}/resources/upload",
        headers=HEADERS,
        params={"path": remote_path, "overwrite": "true"},
    )
    r.raise_for_status()
    upload_url = r.json()["href"]

    # 2. Загрузить файл по полученному URL
    with open(local_path, "rb") as f:
        r = requests.put(upload_url, files={"file": f})
    r.raise_for_status()

    print(f"Загружен: {local_path} → {remote_path}")


# Примеры:
# upload_file("test_file.txt")
# upload_file('report.xlsx', 'app:/data/report.xlsx')


def upload_text(content: str, remote_path: str):
    """Записать строку как файл."""
    r = requests.get(
        f"{BASE_URL}/resources/upload",
        headers=HEADERS,
        params={"path": remote_path, "overwrite": "true"},
    )
    r.raise_for_status()
    upload_url = r.json()["href"]

    r = requests.put(upload_url, data=content.encode("utf-8"))
    r.raise_for_status()
    print(f"Записано: {remote_path}")


def upload_json(data: dict, remote_path: str):
    """Записать dict как JSON-файл."""
    content = json.dumps(data, ensure_ascii=False, indent=2)
    upload_text(content, remote_path)


# Примеры:
# upload_text("Привет, мир!", "app:/hello.txt")
# upload_json({"key": "value", "число": 42}, "app:/config.json")


def read_text(remote_path) -> str:
    """Прочитать текстовый файл."""
    r = requests.get(
        f"{BASE_URL}/resources/download", headers=HEADERS, params={"path": remote_path}
    )
    r.raise_for_status()
    download_url = r.json()["href"]

    r = requests.get(download_url)
    r.raise_for_status()
    return r.text


def read_json(remote_path) -> dict:
    """Прочитать JSON-файл."""
    return json.loads(read_text(remote_path))


# Примеры:
# text = read_text('app:/hello.txt')
# config = read_json('app:/config.json')
# print(config)


def download_file(remote_path, save_path):
    """Скачать файл с Яндекс Диска."""
    # 1. Получить URL для скачивания
    r = requests.get(
        f"{BASE_URL}/resources/download", headers=HEADERS, params={"path": remote_path}
    )
    r.raise_for_status()
    download_url = r.json()["href"]

    # 2. Скачать
    r = requests.get(download_url)
    r.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(r.content)

    print(f"Скачан: {remote_path} → {save_path}")


download_file("app:/test_file.txt", "test_file_downloaded.txt")

# print(list_files())
