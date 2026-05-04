import yt_dlp

url = "https://vkvideo.ru/playlist/-52620949_52572507/video-52620949_456296741"

ydl_opts = {
    "outtmpl": "./audio/%(title)s.%(ext)s",
    # Выбрать лучший аудиопоток
    "format": "bestaudio/best",
    # Извлечь и конвертировать в MP3
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",  # битрейт в kbps
        },
    ],
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
