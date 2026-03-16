from pathlib import Path
import ctranslate2
from faster_whisper import WhisperModel
from tqdm.auto import tqdm

# Автовыбор устройства
# device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
device = "cpu"
compute_type = "float16" if device == "cuda" else "float32"

model = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    device=device,
    compute_type="int8",
)


def _ms_to_srt_time(ms: int) -> str:
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    seconds = ms // 1_000
    ms %= 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def create_english_subtitles(video_path: str) -> str:
    video = Path(video_path).expanduser().resolve()

    if not video.is_file():
        raise FileNotFoundError(f"Видео не найдено: {video}")

    srt_path = video.with_suffix(".srt")

    segments, info = model.transcribe(
        str(video),
        language="en",
        task="transcribe",
        beam_size=5,
    )

    total_seconds = getattr(info, "duration", None)

    with open(srt_path, "w", encoding="utf-8-sig", newline="\r\n") as f:
        subtitle_index = 1
        processed_until = 0.0

        with tqdm(
            total=total_seconds,
            unit="sec",
            desc="Transcribing",
            dynamic_ncols=True,
        ) as pbar:
            for segment in segments:
                text = " ".join(segment.text.split()).strip()

                if text:
                    start_ms = max(0, int(segment.start * 1000))
                    end_ms = max(start_ms + 1, int(segment.end * 1000))

                    f.write(f"{subtitle_index}\n")
                    f.write(
                        f"{_ms_to_srt_time(start_ms)} --> {_ms_to_srt_time(end_ms)}\n"
                    )
                    f.write(f"{text}\n\n")

                    subtitle_index += 1

                current_end = float(segment.end)

                if total_seconds is not None:
                    current_end = min(current_end, total_seconds)
                    delta = max(0.0, current_end - processed_until)
                    pbar.update(delta)
                    processed_until = current_end
                    pbar.set_postfix_str(f"{current_end:.1f}/{total_seconds:.1f} sec")
                else:
                    pbar.update(1)

            if total_seconds is not None and processed_until < total_seconds:
                pbar.update(total_seconds - processed_until)

    return str(srt_path)


if __name__ == "__main__":
    from glob import glob

    paths = glob(r"C:\Users\user\Downloads\archive-2026-03-16_20-17-56\archive\*.mp4")
    paths = [path.replace("\\", "/") for path in paths]
    paths.sort()

    for i, path in enumerate(paths):
        print(i + 1, "/", len(paths))
        create_english_subtitles(video_path=path)

    # path = 1. Welcome to the Course.mp4"
    # result =
    # print("Субтитры созданы:", result)
