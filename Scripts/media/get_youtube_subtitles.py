import re
import sys
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import (
    TextFormatter,
    JSONFormatter,
    WebVTTFormatter,
)


def extract_video_id(url: str) -> str:
    """
    Извлекает video_id из различных форматов YouTube URL:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
    """
    patterns = [
        r"(?:v=)([0-9A-Za-z_-]{11})",  # ?v=VIDEO_ID
        r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",  # youtu.be/VIDEO_ID
        r"(?:shorts/)([0-9A-Za-z_-]{11})",  # /shorts/VIDEO_ID
        r"(?:embed/)([0-9A-Za-z_-]{11})",  # /embed/VIDEO_ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Не удалось извлечь video_id из URL: {url}")


def get_transcript(
    url: str, languages: list[str] = None, output_format: str = "text"
) -> str:
    """
    Получает субтитры по ссылке на YouTube-видео.

    :param url:           Ссылка на YouTube-видео
    :param languages:     Список языков по приоритету, например ['ru', 'en']
                          По умолчанию — автоопределение (сначала ручные, потом авто)
    :param output_format: Формат вывода: 'text' | 'json' | 'vtt' | 'raw'
    :return:              Субтитры в виде строки (или список словарей для 'raw')
    """
    video_id = extract_video_id(url)
    api = YouTubeTranscriptApi()

    # --- Получаем транскрипт ---
    fetch_kwargs = {}
    if languages:
        fetch_kwargs["languages"] = languages

    transcript = api.fetch(video_id, **fetch_kwargs)

    print(f"[INFO] Видео ID   : {transcript.video_id}")
    print(f"[INFO] Язык       : {transcript.language} ({transcript.language_code})")
    print(f"[INFO] Авто-генер.: {transcript.is_generated}")
    print("-" * 50)

    # --- Форматирование ---
    if output_format == "raw":
        return transcript.to_raw_data()  # list[dict]

    elif output_format == "json":
        formatter = JSONFormatter()
        return formatter.format_transcript(transcript, indent=2)

    elif output_format == "vtt":
        formatter = WebVTTFormatter()
        return formatter.format_transcript(transcript)

    else:  # "text" по умолчанию
        formatter = TextFormatter()
        return formatter.format_transcript(transcript)


def list_available_languages(url: str):
    """Выводит все доступные языки субтитров для видео."""
    video_id = extract_video_id(url)
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    print(f"Доступные субтитры для видео [{video_id}]:\n")
    for t in transcript_list:
        kind = "авто" if t.is_generated else "ручные"
        print(f"  [{t.language_code}] {t.language} ({kind})")


# ============================================================
#  ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================
if __name__ == "__main__":
    VIDEO_URL = "https://www.youtube.com/watch?v=TLcdzeqBLpM"

    # # 1) Посмотреть доступные языки
    # print("=== Доступные языки ===")
    # list_available_languages(VIDEO_URL)
    # print()

    # # 2) Получить субтитры (приоритет: русский → английский)
    # print("=== Субтитры (текст) ===")
    # text = get_transcript(VIDEO_URL, languages=["en", "ru"], output_format="text")
    # print(text)

    # # 3) Получить субтитры в формате JSON (с таймкодами)
    # print("\n=== Субтитры (JSON) ===")
    # json_data = get_transcript(VIDEO_URL, languages=["ru", "en"], output_format="json")
    # print(json_data)

    # # 4) Получить сырые данные (list of dict: text, start, duration)
    # print("\n=== Субтитры (raw) ===")
    # raw = get_transcript(VIDEO_URL, languages=["en"], output_format="raw")
    # for snippet in raw[:5]:  # первые 5 строк
    #     print(f"  [{snippet['start']:.2f}s] {snippet['text']}")

    # 5) Сохранить субтитры в файл (.vtt)
    print("\n=== Сохранение в файл ===")
    vtt = get_transcript(VIDEO_URL, languages=["en"], output_format="text")
    with open("./data/subtitles.txt", "w", encoding="utf-8") as f:
        f.write(vtt)
    print("Субтитры сохранены в subtitles.txt")
