import yt_dlp


def get_video_info(url):
    """Получить информацию о видео без скачивания"""
    ydl_opts = {"quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        print(f"Название: {info.get('title')}")
        print(f"Длительность: {info.get('duration')} сек.")
        print(f"Доступные форматы:")
        for f in info.get("formats", []):
            print(
                f"  {f.get('format_id'):>10} | {f.get('ext'):>5} | "
                f"{f.get('width', '?')}x{f.get('height', '?')} | "
                f"{f.get('format_note', '')}"
            )
        return info


def download_vk_video(url, output_path="./data", quality="best"):
    ydl_opts = {
        "outtmpl": f"{output_path}/%(title)s.%(ext)s",
        "format": quality,
        # Для авторизованного доступа:
        # 'cookiesfrombrowser': ('chrome',),
        # Или через логин/пароль:
        # 'username': 'ваш_email',
        # 'password': 'ваш_пароль',
        "writethumbnail": True,
        "writedescription": True,
        "writeinfojson": True,
        "progress_hooks": [progress_hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def progress_hook(d):
    if d["status"] == "downloading":
        percent = d.get("_percent_str", "N/A")
        speed = d.get("_speed_str", "N/A")
        print(f"\rСкачивание: {percent} | Скорость: {speed}", end="")
    elif d["status"] == "finished":
        print(f"\nЗагрузка завершена: {d['filename']}")


# Пример
url = "https://vkvideo.ru/playlist/-52620949_52572507/video-52620949_456296741"

# Сначала смотрим информацию
# get_video_info(url)

# Затем скачиваем
download_vk_video(url, output_path="./data", quality="best")
