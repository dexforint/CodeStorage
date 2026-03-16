import os
from yt_dlp import YoutubeDL


def download_youtube_video(url: str, output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "outtmpl": os.path.join(output_dir, "%(title)s [%(id)s].%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "writethumbnail": True,  # скачать thumbnail
        "writedescription": True,  # скачать описание
        "noplaylist": True,  # не скачивать весь плейлист
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=qRyLEOj5VOs"
    download_youtube_video(video_url)
