#!/usr/bin/env python3
"""
Визуализатор результатов детекции YOLO

Принимает:
    - исходное видео
    - JSON-файл с результатами детекции (из MaximumQualityDetector)

Создаёт:
    - новое видео с отрисованными bounding box, track_id и confidence
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


def load_detections(json_path: Path) -> Dict[int, list[dict]]:
    """
    Загружает результаты детекции из JSON-файла.
    Возвращает словарь: {frame_id: [list of detections]}
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    detections_per_frame = data.get("detections_per_frame", {})

    # Приводим строковые ключи к int
    result: Dict[int, list[dict]] = {}
    for frame_str, dets in detections_per_frame.items():
        result[int(frame_str)] = dets

    print(f"Загружено детекций для {len(result)} кадров из файла {json_path.name}")
    return result


def get_unique_color(track_id: int) -> tuple[int, int, int]:
    """
    Генерирует уникальный яркий цвет для каждого track_id.
    Один и тот же трек всегда будет одного цвета.
    """
    # Используем HSV цветовое пространство для хорошего распределения цветов
    hue = (track_id * 137) % 360  # 137 — простое число, даёт хорошее распределение
    saturation = 255
    value = 255
    color_bgr = cv2.cvtColor(
        np.array([[[hue, saturation, value]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )
    return int(color_bgr[0][0][0]), int(color_bgr[0][0][1]), int(color_bgr[0][0][2])


def draw_detection(
    frame: np.ndarray,
    det: dict,
    color: tuple[int, int, int],
    thickness: int = 3,
    show_confidence: bool = True,
    font_scale: float = 0.7,
) -> None:
    """
    Рисует один bounding box с подписью на кадре.
    """
    bbox = det["bbox"]
    x1, y1, x2, y2 = map(int, bbox)
    track_id = det.get("track_id")
    confidence = det.get("confidence", 0.0)

    # Рамка
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Подпись
    label_parts = []
    if track_id is not None:
        label_parts.append(f"ID:{track_id}")
    if show_confidence:
        label_parts.append(f"{confidence:.2f}")

    label = " | ".join(label_parts)

    # Добавляем фон под текст для читаемости
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
    )

    # Позиция текста (над боксом)
    text_x = x1
    text_y = max(y1 - 10, text_height + 5)

    # Полупрозрачный фон
    cv2.rectangle(
        frame,
        (text_x - 2, text_y - text_height - 5),
        (text_x + text_width + 2, text_y + 5),
        color,
        -1,
    )

    # Сам текст (белый)
    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def visualize(
    video_path: Path,
    json_path: Path,
    output_path: Path,
    thickness: int = 4,
    font_scale: float = 0.8,
    show_confidence: bool = True,
    fourcc: str = "mp4v",
) -> None:
    """
    Основная функция визуализации.
    """
    # Загружаем результаты детекции
    detections_dict = load_detections(json_path)

    # Открываем видео
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Настройка записи видео
    fourcc_codec = cv2.VideoWriter_fourcc(*fourcc)
    out = cv2.VideoWriter(str(output_path), fourcc_codec, fps, (width, height))

    print(f"Начало визуализации: {video_path.name}")
    print(f"Выходной файл: {output_path.name}")
    print(f"Разрешение: {width}x{height}, FPS: {fps:.2f}, Кадров: {total_frames}")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Получаем детекции для текущего кадра
        current_detections = detections_dict.get(frame_id, [])

        for det in current_detections:
            track_id = det.get("track_id")
            color = get_unique_color(track_id) if track_id is not None else (0, 255, 0)

            draw_detection(
                frame=frame,
                det=det,
                color=color,
                thickness=thickness,
                show_confidence=show_confidence,
                font_scale=font_scale,
            )

        out.write(frame)
        frame_id += 1

        if frame_id % 500 == 0:
            print(f"Обработано кадров: {frame_id}/{total_frames}")

    cap.release()
    out.release()
    print("Визуализация успешно завершена!")


def main():
    parser = argparse.ArgumentParser(
        description="Визуализатор результатов YOLO детекции",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="Путь к исходному видео")
    parser.add_argument("json", type=Path, help="Путь к JSON-файлу с детекциями")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Путь к выходному видео (по умолчанию: video_detected.mp4)",
    )
    parser.add_argument("--thickness", type=int, default=4, help="Толщина рамки")
    parser.add_argument("--font-scale", type=float, default=0.75, help="Размер шрифта")
    parser.add_argument(
        "--no-confidence", action="store_true", help="Не показывать confidence"
    )

    args = parser.parse_args()

    if not args.video.exists():
        parser.error(f"Видео не найдено: {args.video}")
    if not args.json.exists():
        parser.error(f"JSON-файл не найден: {args.json}")

    output = args.output or args.video.with_name(f"{args.video.stem}_detected.mp4")

    visualize(
        video_path=args.video,
        json_path=args.json,
        output_path=output,
        thickness=args.thickness,
        font_scale=args.font_scale,
        show_confidence=not args.no_confidence,
    )


if __name__ == "__main__":
    main()
