"""
Визуализация результатов детекции из нескольких источников.
Каждая папка с разметкой рисуется своим цветом.
Поддерживает изображения и видео.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
import cv2

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# Цвета по умолчанию
DEFAULT_COLORS = ["lime", "red", "cyan", "yellow", "magenta", "blue", "white"]


def parse_color(color_str: str) -> Tuple[int, int, int]:
    """Преобразует строку цвета в BGR."""
    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "lime": (0, 255, 0),
        "cyan": (255, 255, 0),
        "yellow": (0, 255, 255),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    if color_str.lower() in color_map:
        return color_map[color_str.lower()]
    # Если RGB в формате "255,0,0"
    try:
        r, g, b = map(int, color_str.split(","))
        return (b, g, r)  # OpenCV использует BGR
    except:
        return (0, 255, 0)  # fallback


def visualize_image(
    image_path: Path,
    label_sources: List[Tuple[Path, Tuple[int, int, int]]],
    output_path: Path,
):
    """Рисует боксы из нескольких источников на одном изображении."""
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Не удалось прочитать {image_path.name}")
        return

    h, w = img.shape[:2]

    for label_dir, color in label_sources:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                conf = float(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    img,
                    f"{conf:.2f}",
                    (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

    cv2.imwrite(str(output_path), img)


def visualize_video(
    video_path: Path,
    label_sources: List[Tuple[Path, Tuple[int, int, int]]],
    output_path: Path,
):
    """Создаёт видео с боксами из нескольких источников."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    logger.info(f"Визуализация видео ({len(label_sources)} источников)...")

    for frame_idx in tqdm(range(frame_count), desc="Кадры"):
        ret, frame = cap.read()
        if not ret:
            break

        for label_dir, color in label_sources:
            label_path = label_dir / f"frame_{frame_idx:06d}.txt"
            if not label_path.exists():
                continue
            h, w = frame.shape[:2]
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    conf = float(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(
                        frame,
                        f"{conf:.2f}",
                        (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

        out.write(frame)

    cap.release()
    out.release()
    logger.info(f"Видео сохранено: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Визуализация нескольких источников детекции"
    )
    parser.add_argument(
        "--source", type=Path, required=True, help="Папка с изображениями или видеофайл"
    )
    parser.add_argument(
        "--label-sources",
        type=str,
        nargs="+",
        required=True,
        help="Список в формате: folder_path:color (например: results_yolo:lime results_dino:red)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Папка для изображений или путь к видео (.mp4)",
    )

    args = parser.parse_args()

    # Парсим label-sources
    label_sources: List[Tuple[Path, Tuple[int, int, int]]] = []
    for item in args.label_sources:
        if ":" not in item:
            logger.error(f"Неверный формат: {item}. Ожидается path:color")
            continue
        path_str, color_str = item.split(":", 1)
        label_sources.append((Path(path_str), parse_color(color_str)))

    if not label_sources:
        logger.error("Не указано ни одного источника разметки")
        return

    logger.info(f"Визуализируем {len(label_sources)} источников разметки.")

    if args.source.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        img_paths = list(args.source.glob("*.jpg")) + list(args.source.glob("*.png"))
        for img_path in tqdm(img_paths, desc="Визуализация изображений"):
            visualize_image(img_path, label_sources, args.output / img_path.name)
    else:
        visualize_video(args.source, label_sources, args.output)

    logger.info("Визуализация завершена.")


if __name__ == "__main__":
    main()
