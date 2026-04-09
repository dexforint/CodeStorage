"""
Скрипт для детекции людей на изображениях и видео.
Поддерживает как папки с картинками, так и видеофайлы.
"""

import logging
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# ====================== НАСТРОЙКИ ======================
CONF_THRESHOLD = 0.25
IMGSZ = 1280
FRAME_NAME_FORMAT = "frame_{:06d}.txt"  # формат именования кадров видео
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
# ======================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def initialize_model() -> YOLO:
    """Инициализирует модель YOLOv11x (меняйте только эту функцию при смене модели)."""
    logger.info("Инициализация модели YOLOv11x (imgsz=%d)...", IMGSZ)

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        logger.info("Используется GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning("CUDA недоступен. Работаем на CPU.")

    # model = YOLO("yolo11x.pt")
    model = YOLO("./data/yolov8n.pt")
    model.to(device)
    logger.info("Модель успешно загружена.")
    return model


def run_inference(
    model: YOLO, source: np.ndarray | Path | str
) -> List[Tuple[float, float, float, float, float]]:
    """
    Выполняет детекцию на одном кадре (numpy) или изображении (path).
    Возвращает список: (confidence, cx, cy, w, h) — все значения нормализованы [0, 1].
    """
    try:
        results = model.predict(
            source=source,
            conf=CONF_THRESHOLD,
            iou=0.45,
            imgsz=IMGSZ,
            augment=True,
            classes=[0],  # person
            verbose=False,
            device=model.device,
        )

        detections = []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        img_h, img_w = results[0].orig_shape[:2]

        for box in boxes:
            conf = float(box.conf.item())
            x, y, w, h = box.xywh[0].tolist()

            detections.append((conf, x / img_w, y / img_h, w / img_w, h / img_h))
        return detections

    except Exception as e:
        logger.error("Ошибка инференса: %s", e)
        return []


def process_video(model: YOLO, video_path: Path, output_dir: Path) -> None:
    """Обрабатывает видео файл и сохраняет детекции по кадрам."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Не удалось открыть видео: %s", video_path)
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        "Обработка видео: %s (%.1f FPS, %d кадров)", video_path.name, fps, frame_count
    )

    frame_idx = 0
    pbar = tqdm(total=frame_count, desc="Обработка кадров")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_inference(model, frame)

        txt_filename = FRAME_NAME_FORMAT.format(frame_idx)
        txt_path = output_dir / txt_filename

        with open(txt_path, "w", encoding="utf-8") as f:
            for conf, cx, cy, w, h in detections:
                f.write(f"{conf:.6f} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    logger.info("Видео обработано. Сохранено %d файлов меток.", frame_idx)


def process_folder(model: YOLO, input_dir: Path, output_dir: Path) -> None:
    """Обрабатывает папку с изображениями (существующая логика)."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions
    ]

    logger.info("Найдено изображений: %d", len(image_files))

    for img_path in tqdm(image_files, desc="Детекция"):
        detections = run_inference(model, img_path)
        txt_path = output_dir / f"{img_path.stem}.txt"

        with open(txt_path, "w", encoding="utf-8") as f:
            for conf, cx, cy, w, h in detections:
                f.write(f"{conf:.6f} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Детекция людей на изображениях и видео"
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Путь к папке с изображениями или к видеофайлу",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Папка для сохранения результатов (.txt файлов)",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = initialize_model()

    if args.source.is_dir():
        logger.info("Режим обработки папки с изображениями")
        process_folder(model, args.source, args.output_dir)
    elif args.source.suffix.lower() in VIDEO_EXTENSIONS:
        logger.info("Режим обработки видео")
        process_video(model, args.source, args.output_dir)
    else:
        logger.error("Неподдерживаемый тип источника. Укажите папку или видеофайл.")
        return

    logger.info("Готово! Результаты сохранены в: %s", args.output_dir)


if __name__ == "__main__":
    main()
