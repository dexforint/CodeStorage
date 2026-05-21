import yt_dlp
import re
import os
import glob

url = "https://www.youtube.com/watch?v=SKI6pf0gkNo"

ydl_opts = {
    "skip_download": True,
    "writesubtitles": True,
    "writeautomaticsub": True,
    "subtitleslangs": ["ru"],
    "subtitlesformat": "vtt",  # Используем VTT, так как его легко парсить
    "outtmpl": "%(title)s.%(ext)s",
}


def clean_vtt_to_text(vtt_filepath):
    """Функция для очистки VTT файла до чистого текста"""
    with open(vtt_filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Удаляем заголовок WEBVTT
    text = re.sub(r"^WEBVTT.*\n", "", text, flags=re.MULTILINE)
    # Удаляем таймкоды (например: 00:00:00.000 --> 00:00:02.000)
    text = re.sub(
        r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*\n", "", text
    )
    text = re.sub(
        r"\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}\.\d{3}.*\n", "", text
    )  # Короткие таймкоды
    # Удаляем HTML-подобные теги (например <c> или </c>, которые бывают в автосабах)
    text = re.sub(r"<[^>]+>", "", text)
    # Удаляем пустые строки и лишние переносы, превращая в сплошной текст
    text = re.sub(r"\n+", " ", text)

    # Перезаписываем тот же файл, но уже с чистым текстом
    new_filepath = vtt_filepath.replace(".vtt", ".txt")
    with open(new_filepath, "w", encoding="utf-8") as f:
        f.write(text.strip())

    # Удаляем оригинальный грязный vtt файл
    os.remove(vtt_filepath)
    print(f"Сохранен чистый текст: {new_filepath}")


# Скачиваем с помощью yt-dlp
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

# Ищем скачанный .vtt файл и очищаем его
vtt_files = glob.glob("*.ru.vtt")
for vtt_file in vtt_files:
    clean_vtt_to_text(vtt_file)
