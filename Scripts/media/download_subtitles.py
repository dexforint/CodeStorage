import os
import re
import yt_dlp

URL = "https://www.youtube.com/watch?v=8rYLZ5V_b3M"
LANG = "ru"


def srt_to_plain_text(srt_path: str) -> str:
    """
    Превращает .srt в чистый текст:
    - удаляет номера блоков
    - удаляет строки с таймкодами
    - склеивает текст
    - убирает подряд идущие дубликаты строк (часто бывает из-за перекрытия субтитров)
    """
    timecode_re = re.compile(
        r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s-->\s\d{2}:\d{2}:\d{2}[,.]\d{3}"
    )
    index_re = re.compile(r"^\d+$")

    lines_out = []
    last_line = None

    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            if not line:
                continue
            if index_re.match(line):
                continue
            if timecode_re.match(line):
                continue

            # часто встречаются служебные теги в некоторых субтитрах
            line = re.sub(r"<[^>]+>", "", line).strip()
            if not line:
                continue

            # убираем подряд идущие дубликаты
            if line == last_line:
                continue

            lines_out.append(line)
            last_line = line

    # Склеиваем в один текст (можно заменить на '\n' если нужны абзацы)
    text = " ".join(lines_out)
    text = re.sub(r"\s+", " ", text).strip()
    return text


ydl_opts = {
    "skip_download": True,
    "writesubtitles": True,  # ручные
    "writeautomaticsub": True,  # авто
    "subtitleslangs": [LANG],
    "subtitlesformat": "srt",  # важно: srt/vtt удобнее парсить, чем "txt"
    "outtmpl": "./data/subtitles/%(title)s.%(ext)s",
    "quiet": True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(URL, download=True)

    # Базовое имя, которое yt-dlp использует для файлов
    base = os.path.splitext(ydl.prepare_filename(info))[0]

    # yt-dlp обычно сохраняет как "<base>.<lang>.srt"
    srt_path = f"{base}.{LANG}.srt"
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"Не найден файл субтитров: {srt_path}")

    plain_text = srt_to_plain_text(srt_path)

    txt_path = f"{base}.{LANG}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(plain_text)

print("Готово:", txt_path)
