import os
from yt_dlp import YoutubeDL


def download_youtube_video(url: str, output_dir: str = "./data/video"):
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "outtmpl": os.path.join(output_dir, "%(title)s [%(id)s].%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        # "writethumbnail": True,  # скачать thumbnail
        # "writedescription": True,  # скачать описание
        "noplaylist": True,  # не скачивать весь плейлист
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def main():
    url = input("Введите URL видео на YouTube: ").strip()
    if not url:
        print("Ошибка: URL не может быть пустым.")
        return

    output_dir = input(
        "Введите папку для сохранения (по умолчанию ./data/video): "
    ).strip()
    if not output_dir:
        output_dir = "./data/video"

    print(f"Скачивание видео: {url}")
    download_youtube_video(url, output_dir)
    print("Готово!")


if __name__ == "__main__":
    main()
