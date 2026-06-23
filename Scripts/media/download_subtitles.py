import os
import re
import yt_dlp


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

            line = re.sub(r"<[^>]+>", "", line).strip()
            if not line:
                continue

            if line == last_line:
                continue

            lines_out.append(line)
            last_line = line

    text = " ".join(lines_out)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    url = input("Введите URL видео на YouTube: ").strip()
    if not url:
        print("Ошибка: URL не может быть пустым.")
        return

    lang = input("Введите код языка субтитров (по умолчанию ru): ").strip()
    if not lang:
        lang = "ru"

    outdir = "./data/subtitles"
    os.makedirs(outdir, exist_ok=True)

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "srt",
        "outtmpl": os.path.join(outdir, "%(title)s.%(ext)s"),
        "quiet": True,
    }

    print(f"Получение субтитров для: {url} (язык: {lang})...")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        base = os.path.splitext(ydl.prepare_filename(info))[0]

        srt_path = f"{base}.{lang}.srt"
        if not os.path.exists(srt_path):
            raise FileNotFoundError(
                f"Не найден файл субтитров: {srt_path}. "
                "Возможно, для этого видео нет субтитров на выбранном языке."
            )

        plain_text = srt_to_plain_text(srt_path)

        txt_path = f"{base}.{lang}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(plain_text)

    print("Готово! Текст сохранён в:", txt_path)


if __name__ == "__main__":
    main()
