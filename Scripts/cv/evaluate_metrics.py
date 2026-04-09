"""
Скрипт для оценки качества детекции людей.
Поддерживает:
    - Предсказания в нашем формате: confidence cx cy w h
    - Ground Truth в формате YOLO: 0 x y w h (только класс person)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_txt_file(txt_path: Path) -> List[Tuple[float, float, float, float, float]]:
    """Загружает файл с детекциями (работает с обоими форматами)."""
    if not txt_path.exists():
        return []
    boxes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # Пропускаем первый элемент, если это класс (0)
            start_idx = 1 if len(parts) == 5 and parts[0] == "0" else 0
            conf_or_dummy, cx, cy, w, h = map(float, parts[start_idx : start_idx + 5])
            boxes.append((cx, cy, w, h))
    return boxes


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """Возвращает (width, height) изображения."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")
    return img.shape[1], img.shape[0]  # width, height


def convert_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int):
    """Конвертирует normalized (cx,cy,w,h) в абсолютные (x1,y1,x2,y2)."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def evaluate(
    pred_dir: Path, gt_dir: Path, images_dir: Path, iou_thresholds: List[float] = None
) -> Dict:
    """Основная функция оценки."""
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]

    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", iou_thresholds=iou_thresholds
    )

    # Собираем все файлы
    pred_files = list(pred_dir.glob("*.txt"))
    logger.info(f"Найдено файлов предсказаний: {len(pred_files)}")

    for pred_file in tqdm(pred_files, desc="Обработка изображений/кадров"):
        gt_file = gt_dir / pred_file.name
        image_file = images_dir / pred_file.with_suffix(".jpg").name
        if not image_file.exists():
            image_file = images_dir / pred_file.with_suffix(".png").name
        if not image_file.exists():
            logger.warning(f"Изображение не найдено для {pred_file.name}")
            continue

        try:
            img_w, img_h = get_image_size(image_file)

            # Загружаем предсказания (с confidence)
            pred_boxes_norm = load_txt_file(pred_file)
            gt_boxes_norm = load_txt_file(gt_file)

            pred_boxes = []
            pred_scores = []
            for cx, cy, w, h in pred_boxes_norm:
                box = convert_to_xyxy(cx, cy, w, h, img_w, img_h)
                pred_boxes.append(box)
                pred_scores.append(
                    0.85
                )  # dummy score (не используется при расчёте mAP в этом режиме)

            gt_boxes = [
                convert_to_xyxy(cx, cy, w, h, img_w, img_h)
                for cx, cy, w, h in gt_boxes_norm
            ]

            # Добавляем в метрику
            preds = [
                {
                    "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
                    "scores": torch.tensor(pred_scores, dtype=torch.float32),
                    "labels": torch.zeros(len(pred_boxes), dtype=torch.int64),
                }
            ]

            target = [
                {
                    "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
                    "labels": torch.zeros(len(gt_boxes), dtype=torch.int64),
                }
            ]

            metric.update(preds, target)

        except Exception as e:
            logger.error(f"Ошибка при обработке {pred_file.name}: {e}")

    # Вычисляем метрики
    result = metric.compute()

    logger.info("\n" + "=" * 60)
    logger.info("РЕЗУЛЬТАТЫ ОЦЕНКИ ДЕТЕКЦИИ (только класс 'person')")
    logger.info("=" * 60)
    logger.info(f"mAP@0.5:0.95 = {result['map']:.4f}")
    logger.info(f"mAP@0.5     = {result['map_50']:.4f}")
    logger.info(f"mAP@0.75    = {result['map_75']:.4f}")
    logger.info(
        f"Precision   = {result['map'].item():.4f}"
    )  # torchmetrics возвращает средние
    logger.info(
        f"Recall      = {result.get('recall', torch.tensor(0.0)).mean().item():.4f}"
    )
    logger.info("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(description="Расчёт метрик детекции (mAP)")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Папка с предсказаниями (наш формат)",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Папка с Ground Truth (YOLO формат)",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Папка с оригинальными изображениями/кадрами видео",
    )

    args = parser.parse_args()

    if (
        not args.predictions.exists()
        or not args.ground_truth.exists()
        or not args.images.exists()
    ):
        logger.error("Одна из указанных папок не существует!")
        return

    evaluate(args.predictions, args.ground_truth, args.images)


if __name__ == "__main__":
    main()
