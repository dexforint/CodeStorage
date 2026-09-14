import json
import os
import torch
import whisperx

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ==================== НАСТРОЙКИ (ИЗМЕНИТЕ ПОД СВОИ НУЖДЫ) ====================
AUDIO_PATH = (
    input("Введите путь до аудио файла: ").strip().replace('"', "")
)  # Путь к аудиофайлу
MODEL_NAME = "large-v3"  # Размер модели Whisper: tiny, base, small, medium, large, large-v2, large-v3
LANGUAGE = "ru"  # Язык (например "ru", "en"). None — автоопределение.
COMPUTE_TYPE = "auto"  # "auto", "float16", "float32", "int8"
OUTPUT_DIR = "./data/transcribtions"  # Директория для сохранения JSON и SRT. None — только консоль.
BATCH_SIZE = 16  # Размер батча для транскрипции
# =============================================================================


def transcribe_audio(
    audio_path: str,
    model_name: str = "base",
    language: str | None = None,
    compute_type: str = "auto",
    output_dir: str | None = None,
    batch_size: int = 16,
):
    """
    Транскрибирует аудиофайл с временными метками, используя WhisperX.

    Параметры:
        audio_path (str): путь к аудиофайлу
        model_name (str): размер модели Whisper (tiny, base, small, medium, large, large-v2, large-v3)
        language (str, optional): код языка (например, "ru", "en"). Если None, определяется автоматически.
        compute_type (str): тип вычислений ("auto", "float16", "float32", "int8").
                            По умолчанию "auto": float16 для GPU, float32 для CPU.
        output_dir (str, optional): директория для сохранения результатов (JSON и SRT).
                                    Если не указана, вывод только в консоль.
        batch_size (int): размер батча для транскрипции.
    """
    # Определяем устройство (GPU/CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "float32"

    print(f"Используется устройство: {device}, compute_type: {compute_type}")

    # 1. Загрузка модели Whisper
    model = whisperx.load_model(model_name, device, compute_type=compute_type)

    # 2. Загрузка аудио
    audio = whisperx.load_audio(audio_path)

    # 3. Транскрипция
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    # 4. Выравнивание временных меток (forced alignment)
    detected_language = result["language"]
    print(f"Определён язык: {detected_language}")

    try:
        # Загружаем модель выравнивания и метаданные для данного языка
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language, device=device
        )
        result_aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception as e:
        print(
            f"Предупреждение: не удалось выполнить выравнивание слов ({e}). Будут использованы только сегменты."
        )
        result_aligned = result

    # 5. Вывод результатов в консоль
    # print("\n--- Результат транскрибации с временными метками ---")
    # for segment in result_aligned["segments"]:
    #     start = segment["start"]
    #     end = segment["end"]
    #     text = segment["text"]
    #     print(f"[{start:.2f} - {end:.2f}] {text}")
    #     # Если доступны слова с таймкодами, выводим их
    #     if "words" in segment:
    #         for word in segment["words"]:
    #             w_start = word.get("start", 0.0)
    #             w_end = word.get("end", 0.0)
    #             w_text = word["word"]
    #             print(f"    {w_start:.2f} - {w_end:.2f}: {w_text}")

    # 6. Сохранение результатов (если указана выходная директория)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        json_path = os.path.join(output_dir, f"{base_name}_transcript.json")
        srt_path = os.path.join(output_dir, f"{base_name}_transcript.srt")

        # Сохранение JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_aligned, f, ensure_ascii=False, indent=2)

        # Формирование SRT
        def format_timestamp(seconds: float) -> str:
            ms = int((seconds % 1) * 1000)
            s = int(seconds) % 60
            m = int(seconds // 60) % 60
            h = int(seconds // 3600)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result_aligned["segments"], 1):
                start_ts = format_timestamp(segment["start"])
                end_ts = format_timestamp(segment["end"])
                f.write(f"{i}\n{start_ts} --> {end_ts}\n{segment['text'].strip()}\n\n")

        print(f"\nРезультаты сохранены в:\n  JSON: {json_path}\n  SRT: {srt_path}")

    return result_aligned


if __name__ == "__main__":
    # Проверка существования файла
    if not os.path.exists(AUDIO_PATH):
        print(f"Ошибка: файл {AUDIO_PATH} не найден")
    else:
        transcribe_audio(
            audio_path=AUDIO_PATH,
            model_name=MODEL_NAME,
            language=LANGUAGE,
            compute_type=COMPUTE_TYPE,
            output_dir=OUTPUT_DIR,
            batch_size=BATCH_SIZE,
        )
