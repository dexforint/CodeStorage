import yt_dlp

url = "https://www.youtube.com/watch?v=9xeSBvzU6j8"

# Только субтитры (без видео)
ydl_opts = {
    "skip_download": True,  # не скачивать видео
    "writesubtitles": True,  # ручные субтитры
    "writeautomaticsub": True,  # автогенерированные
    "subtitleslangs": ["en"],
    "subtitlesformat": "srt",  # формат: srt, vtt, json3
    "outtmpl": "%(title)s.%(ext)s",
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
