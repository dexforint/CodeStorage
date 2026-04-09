"""
Скрипт для детекции людей на изображениях и видео с использованием RF-DETR.
Архитектура позволяет легко менять модель, изменяя только initialize_model() и run_inference().
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# RF-DETR импорты
from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES

# ====================== НАСТРОЙКИ ======================
MODEL_SIZE = "large"  # "base" (быстрее) или "large" (точнее)
CONFIDENCE_THRESHOLD = 0.35
OPTIMIZE_FOR_INFERENCE = True  # JIT-компиляция для ускорения на GPU
# =======================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


PERSON_CLASS_ID = 0  # COCO class_id для "person"


@dataclass
class DetectorComponents:
    """Контейнер для компонентов модели RF-DETR."""

    model: Any
    confidence_threshold: float


def load_model(
    device: str = "cuda",
    model_size: str = "large",
    confidence_threshold: float = 0.35,
    optimize: bool = True,
) -> Any:
    """
    Загружает и возвращает модель RF-DETR (base или large).
    """
    if model_size == "large":
        model_cls = RFDETRLarge
        logger.info(
            "Загружается RF-DETR Large (128M параметров) — максимальное качество"
        )
    else:
        model_cls = RFDETRBase
        logger.info("Загружается RF-DETR Base (29M параметров)")

    model = model_cls(device=device)

    # JIT-компиляция значительно ускоряет инференс на GPU
    if optimize and "cuda" in device:
        logger.info("Выполняется оптимизация модели для инференса (compile=True)")
        model.optimize_for_inference(compile=True, batch_size=1)

    model._person_conf_threshold = confidence_threshold
    return model


def initialize_model() -> DetectorComponents:
    """
    Инициализирует модель RF-DETR.

    Примечание: Чтобы сменить модель — измените только эту функцию
    (и необходимые импорты в начале файла).
    """
    logger.info("Инициализация RF-DETR...")

    if not torch.cuda.is_available():
        logger.warning("CUDA недоступен! Модель будет работать на CPU (медленно).")
        device = "cpu"
    else:
        device = "cuda"
        logger.info("CUDA доступен. Используется GPU.")

    model = load_model(
        device=device,
        model_size=MODEL_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        optimize=OPTIMIZE_FOR_INFERENCE,
    )

    logger.info("Модель RF-DETR успешно инициализирована.")
    return DetectorComponents(model=model, confidence_threshold=CONFIDENCE_THRESHOLD)


def run_inference(
    detector: DetectorComponents, source: np.ndarray | Path | str
) -> List[Tuple[float, float, float, float, float]]:
    """
    Выполняет детекцию людей на одном кадре или изображении.

    Args:
        detector: компоненты модели
        source: путь к изображению (Path/str) или numpy-массив кадра в формате BGR

    Returns:
        Список кортежей (confidence, cx, cy, w, h) — все значения нормализованы [0, 1]
    """
    try:
        # Приводим source к numpy BGR (требование detect_persons)
        if isinstance(source, (Path, str)):
            image_bgr = cv2.imread(str(source))
            if image_bgr is None:
                logger.error("Не удалось прочитать изображение: %s", source)
                return []
        else:
            image_bgr = source

        # Основная функция детекции из предоставленного вами кода
        return detect_persons(
            image_bgr=image_bgr,
            model=detector.model,
            confidence_threshold=detector.confidence_threshold,
        )

    except Exception as e:
        logger.error("Ошибка инференса: %s", e)
        return []


def detect_persons(
    image_bgr: np.ndarray,
    model: Any,
    confidence_threshold: float | None = None,
) -> List[Tuple[float, float, float, float, float]]:
    """
    Детектирует людей с помощью RF-DETR (адаптированная версия вашего кода).
    """
    if confidence_threshold is None:
        confidence_threshold = getattr(model, "_person_conf_threshold", 0.35)

    h_img, w_img = image_bgr.shape[:2]
    image_rgb_pil = Image.fromarray(image_bgr[:, :, ::-1])  # BGR → RGB → PIL

    detections = model.predict(image_rgb_pil, threshold=confidence_threshold)
    results: List[Tuple[float, float, float, float, float]] = []

    if detections is None or len(detections.xyxy) == 0:
        return results

    for box_xyxy, conf, class_id in zip(
        detections.xyxy, detections.confidence, detections.class_id
    ):
        if int(class_id) != PERSON_CLASS_ID:
            continue

        x1, y1, x2, y2 = box_xyxy
        x1_n, y1_n = float(x1) / w_img, float(y1) / h_img
        x2_n, y2_n = float(x2) / w_img, float(y2) / h_img

        cx = (x1_n + x2_n) / 2.0
        cy = (y1_n + y2_n) / 2.0
        w = x2_n - x1_n
        h = y2_n - y1_n

        # Клампинг
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        results.append((float(conf), cx, cy, w, h))

    return results


# ====================== ОСНОВНАЯ ЛОГИКА (НЕ МЕНЯТЬ ПРИ СМЕНЕ МОДЕЛИ) ======================


def process_video(
    detector: DetectorComponents, video_path: Path, output_dir: Path
) -> None:
    """Обрабатывает видео файл."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Не удалось открыть видео: %s", video_path.name)
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Обработка видео: %s (%d кадров)", video_path.name, frame_count)

    pbar = tqdm(total=frame_count, desc="Обработка кадров")
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_inference(detector, frame)
        txt_path = output_dir / f"frame_{frame_idx:06d}.txt"

        with open(txt_path, "w", encoding="utf-8") as f:
            for conf, cx, cy, w, h in detections:
                f.write(f"{conf:.6f} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        pbar.update(1)

    cap.release()
    pbar.close()
    logger.info("Видео обработано. Сохранено %d файлов меток.", frame_count)


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

    parser = argparse.ArgumentParser(description="Детекция людей с помощью RF-DETR")
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
    elif args.source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}:
        logger.info("Режим: обработка видео")
        process_video(detector, args.source, args.output_dir)
    else:
        logger.error("Неподдерживаемый источник. Укажите папку или видеофайл.")

    logger.info("Обработка завершена! Результаты сохранены в: %s", args.output_dir)


if __name__ == "__main__":
    main()
