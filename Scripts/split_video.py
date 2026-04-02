#!/usr/bin/env python3
"""
Разделение MP4-видео на N равных по продолжительности частей.
Использует ffmpeg с копированием потоков (без перекодирования) — быстро и без потери качества.

Использование:
    python split_video.py video.mp4 5
    python split_video.py video.mp4 3 -o ./output
    python split_video.py video.mp4 4 --reencode   # точная нарезка с перекодированием
"""

import subprocess
import sys
import os
import json
import shutil
import argparse
from pathlib import Path


def check_ffmpeg():
    """Проверяет наличие ffmpeg и ffprobe в системе."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            print(f"Ошибка: '{tool}' не найден в PATH.")
            print("Установите FFmpeg: https://ffmpeg.org/download.html")
            sys.exit(1)


def get_video_duration(input_file: str) -> float:
    """Получает длительность видео в секундах через ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_entries", "format=duration",
        input_file,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe завершился с ошибкой:\n{result.stderr.strip()}"
        )

    info = json.loads(result.stdout)
    duration = info.get("format", {}).get("duration")

    if duration is None:
        raise RuntimeError("Не удалось определить длительность видео.")

    return float(duration)


def format_time(seconds: float) -> str:
    """Форматирует секунды в HH:MM:SS.ms."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def split_video(
    input_file: str,
    n: int,
    output_dir: str | None = None,
    reencode: bool = False,
):
    """
    Разбивает видео на n равных частей.

    Args:
        input_file:  путь к исходному видеофайлу
        n:           количество частей
        output_dir:  папка для сохранения (по умолчанию — рядом с исходным файлом)
        reencode:    True — перекодировать (точная нарезка по кадрам),
                     False — копировать потоки (быстро, нарезка по ключевым кадрам)
    """
    input_path = Path(input_file).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    if n < 1:
        raise ValueError("Количество частей должно быть ≥ 1.")

    if n == 1:
        print("n = 1 — нечего разделять.")
        return

    # ---------- параметры ----------
    duration = get_video_duration(str(input_path))
    segment_duration = duration / n

    if output_dir is None:
        out_path = input_path.parent
    else:
        out_path = Path(output_dir).resolve()

    out_path.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    ext = input_path.suffix  # .mp4

    mode = "с перекодированием (точная нарезка)" if reencode else "без перекодирования (copy)"

    print(f"Исходный файл : {input_path}")
    print(f"Длительность  : {format_time(duration)}  ({duration:.3f} с)")
    print(f"Частей        : {n}")
    print(f"Длина части   : {format_time(segment_duration)}  ({segment_duration:.3f} с)")
    print(f"Режим         : {mode}")
    print(f"Выходная папка: {out_path}")
    print("-" * 60)

    # ---------- нарезка ----------
    for i in range(n):
        start = i * segment_duration
        part_num = i + 1
        output_file = out_path / f"{stem}_part{part_num:03d}{ext}"

        # Формируем команду ffmpeg
        if reencode:
            # Точная нарезка: -ss после -i, перекодирование
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(input_path),
                "-ss", str(start),
                "-t", str(segment_duration),
                "-map", "0",                     # все потоки (видео + аудио + субтитры)
                "-avoid_negative_ts", "make_zero",
                str(output_file),
            ]
        else:
            # Быстрая нарезка: -ss перед -i, копирование потоков
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(start),
                "-i", str(input_path),
                "-t", str(segment_duration),
                "-map", "0",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(output_file),
            ]

        print(
            f"  [{part_num}/{n}]  "
            f"{format_time(start)} → {format_time(min(start + segment_duration, duration))}  "
            f"=> {output_file.name}"
        )

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  ⚠  Ошибка ffmpeg для части {part_num}:")
            # Показываем последние строки stderr (самые информативные)
            err_lines = result.stderr.strip().splitlines()
            for line in err_lines[-5:]:
                print(f"      {line}")

    print("-" * 60)
    print("Готово!")


def main():
    parser = argparse.ArgumentParser(
        description="Разделение видео на N равных по продолжительности частей.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python split_video.py movie.mp4 5\n"
            "  python split_video.py movie.mp4 3 -o ./parts\n"
            "  python split_video.py movie.mp4 10 --reencode\n"
        ),
    )
    parser.add_argument("input", help="Путь к исходному видеофайлу")
    parser.add_argument("n", type=int, help="Количество частей")
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Папка для сохранения (по умолчанию — рядом с исходным файлом)",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Перекодировать видео для точной нарезки (медленнее, но точнее)",
    )

    args = parser.parse_args()

    check_ffmpeg()
    split_video(args.input, args.n, args.output_dir, args.reencode)


if __name__ == "__main__":
    main()