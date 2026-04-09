"""
Скрипт для детекции людей на изображениях и видео с использованием Grounding DINO.
Архитектура позволяет легко менять модель, изменяя только initialize_model() и run_inference().
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# ====================== НАСТРОЙКИ ======================
BOX_THRESHOLD = 0.35  # Порог для bounding box
TEXT_THRESHOLD = 0.25  # Порог для текстового промпта
TEXT_PROMPT = "person"  # Zero-shot промпт для детекции людей
# =======================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DetectorComponents:
    """Контейнер для компонентов Grounding DINO."""

    model: Any
    processor: Any
    text_labels: List[List[str]]
    device: torch.device


def initialize_model() -> DetectorComponents:
    """
    Инициализирует модель Grounding DINO (zero-shot object detection).

    Примечание: Чтобы сменить модель (например, на YOLO, RT-DETR,
    Florence-2 или другую версию Grounding DINO) — измените только эту функцию.
    """
    logger.info("Инициализация Grounding DINO (base)...")

    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Используется устройство: {device}")

    if device.type == "cuda":
        logger.info("CUDA доступен. Модель будет работать на GPU.")
    else:
        logger.warning(
            "CUDA недоступен! Работа на CPU будет очень медленной (особенно на видео)."
        )

    model_id = (
        "IDEA-Research/grounding-dino-base"  # Можно поменять на "tiny" или "large"
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # Для Grounding DINO текстовая метка передаётся в виде списка списков
    text_labels = [[TEXT_PROMPT]]

    logger.info("Модель Grounding DINO успешно инициализирована.")
    return DetectorComponents(
        model=model, processor=processor, text_labels=text_labels, device=device
    )


def run_inference(
    detector: DetectorComponents, source: np.ndarray | Path | str
) -> List[Tuple[float, float, float, float, float]]:
    """
    Выполняет zero-shot детекцию людей на одном кадре/изображении.

    Args:
        detector: компоненты модели (model, processor, text_labels, device)
        source: либо путь к изображению (Path), либо numpy-массив кадра (BGR)

    Returns:
        Список кортежей: (confidence, center_x, center_y, width, height) — всё нормализовано [0.0, 1.0]
    """
    try:
        # Конвертация источника в PIL Image
        if isinstance(source, (Path, str)):
            image = Image.open(str(source)).convert("RGB")
        else:
            # Если пришёл кадр из OpenCV (BGR) — конвертируем в RGB PIL
            frame_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

        # Подготовка входных данных
        inputs = detector.processor(
            images=image, text=detector.text_labels, return_tensors="pt"
        ).to(detector.device)

        # Инференс
        with torch.no_grad():
            outputs = detector.model(**inputs)

        # Постобработка
        results = detector.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            # box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],  # (width, height)
        )

        # 026-04-09 22:11:43,158 | ERROR | Ошибка инференса: GroundingDinoProcessor.post_process_grounded_object_detection() got an unexpected keyword argument 'box_threshold'
        # Обработка кадров видео:   1%|█▋                                                                                                                                                                                                                       | 252/31261 [00:41<1:21:34,  6.34it/s]2026-04-09 22:11:43,315 | ERROR | Ошибка инференса: GroundingDinoProcessor.post_process_grounded_object_detection() got an unexpected keyword argument 'box_threshol

        result = results[0]
        detections = []
        orig_w, orig_h = image.size

        for box, score in zip(result["boxes"], result["scores"]):
            # box приходит в формате [x1, y1, x2, y2] (абсолютные координаты)
            x1, y1, x2, y2 = box.tolist()

            cx = (x1 + x2) / 2 / orig_w
            cy = (y1 + y2) / 2 / orig_h
            w = (x2 - x1) / orig_w
            h = (y2 - y1) / orig_h
            conf = float(score.item())

            detections.append((conf, cx, cy, w, h))

        return detections

    except Exception as e:
        logger.error("Ошибка инференса: %s", e)
        return []


# ====================== ОСНОВНАЯ ЛОГИКА (НЕ МЕНЯТЬ ПРИ СМЕНЕ МОДЕЛИ) ======================


def process_video(
    detector: DetectorComponents, video_path: Path, output_dir: Path
) -> None:
    """Обрабатывает видео и сохраняет детекции по каждому кадру."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Не удалось открыть видео: %s", video_path.name)
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        "Обработка видео: %s (%.1f FPS, %d кадров). Grounding DINO работает медленно.",
        video_path.name,
        fps,
        frame_count,
    )

    frame_idx = 0
    pbar = tqdm(total=frame_count, desc="Обработка кадров видео")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_inference(detector, frame)

        txt_path = output_dir / f"frame_{frame_idx:06d}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for conf, cx, cy, w, h in detections:
                f.write(f"{conf:.6f} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    logger.info("Видео обработано. Сохранено %d файлов меток.", frame_idx)


def process_folder(
    detector: DetectorComponents, input_dir: Path, output_dir: Path
) -> None:
    """Обрабатывает папку с изображениями."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions
    ]

    logger.info("Найдено изображений: %d", len(image_files))

    for img_path in tqdm(image_files, desc="Детекция изображений"):
        detections = run_inference(detector, img_path)
        txt_path = output_dir / f"{img_path.stem}.txt"

        with open(txt_path, "w", encoding="utf-8") as f:
            for conf, cx, cy, w, h in detections:
                f.write(f"{conf:.6f} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Детекция людей с помощью Grounding DINO"
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Путь к папке с изображениями или видеофайлу",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Папка для сохранения .txt файлов с детекциями",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detector = initialize_model()

    if args.source.is_dir():
        logger.info("Режим: обработка папки с изображениями")
        process_folder(detector, args.source, args.output_dir)
    elif args.source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}:
        logger.info("Режим: обработка видео")
        process_video(detector, args.source, args.output_dir)
    else:
        logger.error("Неподдерживаемый источник. Укажите папку или видеофайл.")
        return

    logger.info("Обработка завершена! Результаты сохранены в: %s", args.output_dir)


if __name__ == "__main__":
    main()
