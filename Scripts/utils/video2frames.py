#!/usr/bin/env python3
"""
Скрипт для конвертации видео <-> кадры.

Использование:
    python video_frames.py <путь>

- Если путь указывает на видеофайл → создаёт папку с кадрами.
- Если путь указывает на папку с изображениями → собирает из них видео.
"""

import os
import sys
import glob
import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

DEFAULT_FPS = 30


def is_video_file(path: str) -> bool:
    return (
        os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS
    )


def video_to_frames(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.path.dirname(video_path) or ".", f"{base_name}_frames")
    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, frame)
        frame_idx += 1

    cap.release()
    print(f"Извлечено {frame_idx} кадров в папку: {output_dir}")


def frames_to_video(folder_path: str) -> None:
    patterns = [os.path.join(folder_path, f"*{ext}") for ext in IMAGE_EXTENSIONS]
    images = []
    for p in patterns:
        images.extend(glob.glob(p))
    images = sorted(images)

    if not images:
        print(f"Ошибка: в папке {folder_path} не найдено изображений")
        sys.exit(1)

    first = cv2.imread(images[0])
    if first is None:
        print(f"Ошибка: не удалось прочитать {images[0]}")
        sys.exit(1)

    height, width = first.shape[:2]
    fps = DEFAULT_FPS

    output_path = os.path.join(
        os.path.dirname(folder_path) or ".",
        f"{os.path.basename(folder_path.rstrip(os.sep))}_video.mp4",
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Пропуск повреждённого файла: {img_path}")
            continue
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        writer.write(img)

    writer.release()
    print(f"Видео сохранено: {output_path} ({len(images)} кадров, {fps} fps)")


def main() -> None:
    if len(sys.argv) != 2:
        print("Использование: python video_frames.py <путь_к_видео_или_папке>")
        sys.exit(1)

    path = os.path.abspath(sys.argv[1])

    if not os.path.exists(path):
        print(f"Ошибка: путь не существует: {path}")
        sys.exit(1)

    if is_video_file(path):
        video_to_frames(path)
    elif os.path.isdir(path):
        frames_to_video(path)
    else:
        print("Ошибка: укажите путь к видеофайлу или к папке с изображениями")
        sys.exit(1)


if __name__ == "__main__":
    main()
