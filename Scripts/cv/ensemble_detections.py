"""
Скрипт для объединения результатов детекций из нескольких папок (ensemble).
Используется Weighted Boxes Fusion (WBF) — один из лучших методов для повышения качества детекции.

Формат входных и выходных файлов:
    confidence cx cy w h    (все значения нормализованы в диапазоне [0, 1])
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ensemble_boxes import weighted_boxes_fusion, nms
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_detections(txt_path: Path) -> Tuple[List[List[float]], List[float]]:
    """
    Загружает детекции из .txt файла.
    Возвращает:
        boxes: список [x1, y1, x2, y2] (normalized)
        scores: список confidence
    """
    if not txt_path.exists():
        return [], []

    boxes = []
    scores = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            conf = float(parts[0])
            cx, cy, w, h = map(float, parts[1:5])

            # Конвертация из (cx, cy, w, h) в (x1, y1, x2, y2)
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)

    return boxes, scores


def save_detections(
    output_path: Path, boxes: List[List[float]], scores: List[float]
) -> None:
    """Сохраняет финальные детекции в исходном формате: confidence cx cy w h"""
    with open(output_path, "w", encoding="utf-8") as f:
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            f.write(f"{score:.6f} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def ensemble_detections(
    input_folders: List[Path],
    output_folder: Path,
    weights: List[float] | None = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.001,
    method: str = "wbf",
) -> None:
    """
    Основная функция объединения результатов из нескольких папок.

    Args:
        input_folders: список папок с результатами разных моделей
        output_folder: папка для сохранения итоговых .txt файлов
        weights: веса для каждой папки (модели). Если None — все веса = 1.0
        iou_thr: порог IoU для suppression/fusion
        skip_box_thr: минимальный confidence для учёта бокса
        method: "wbf" (рекомендуется) или "nms"
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    if weights is None:
        weights = [1.0] * len(input_folders)

    if len(weights) != len(input_folders):
        raise ValueError("Количество весов должно совпадать с количеством папок.")

    logger.info(
        "Запуск ensemble (%s) из %d источников", method.upper(), len(input_folders)
    )
    logger.info("Веса моделей: %s", weights)

    # Собираем все уникальные изображения
    all_txt_files = set()
    for folder in input_folders:
        all_txt_files.update(f.name for f in folder.glob("*.txt"))

    logger.info("Найдено уникальных изображений: %d", len(all_txt_files))

    for txt_name in tqdm(all_txt_files, desc="Ensemble детекций"):
        all_boxes = []
        all_scores = []
        all_labels = []

        for folder, weight in zip(input_folders, weights):
            boxes, scores = load_detections(folder / txt_name)

            if boxes:
                all_boxes.append(np.array(boxes, dtype=np.float32))
                all_scores.append(np.array(scores, dtype=np.float32))
                all_labels.append(np.zeros(len(scores), dtype=np.int32))  # один класс
            else:
                # Добавляем пустые списки, чтобы сохранить соответствие весов
                all_boxes.append(np.empty((0, 4), dtype=np.float32))
                all_scores.append(np.empty(0, dtype=np.float32))
                all_labels.append(np.empty(0, dtype=np.int32))

        if not any(len(s) > 0 for s in all_scores):
            # Нет ни одной детекции
            (output_folder / txt_name).touch()
            continue

        # Применяем ensemble
        if method.lower() == "wbf":
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list=all_boxes,
                scores_list=all_scores,
                labels_list=all_labels,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
        else:  # nms
            fused_boxes, fused_scores, fused_labels = nms(
                boxes_list=all_boxes,
                scores_list=all_scores,
                labels_list=all_labels,
                weights=weights,
                iou_thr=iou_thr,
            )

        save_detections(
            output_folder / txt_name, fused_boxes.tolist(), fused_scores.tolist()
        )

    logger.info("Ensemble успешно завершён! Результаты сохранены в: %s", output_folder)


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble детекций людей из нескольких моделей (WBF/NMS)"
    )
    parser.add_argument(
        "--input_folders",
        type=str,
        nargs="+",
        required=True,
        help="Пути к папкам с результатами детекций (минимум 2)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Папка для сохранения объединённых результатов",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=None,
        help="Веса для каждой модели (по умолчанию все 1.0)",
    )
    parser.add_argument(
        "--iou_thr", type=float, default=0.55, help="IoU threshold (по умолчанию 0.55)"
    )
    parser.add_argument(
        "--skip_box_thr",
        type=float,
        default=0.001,
        help="Минимальный confidence (по умолчанию 0.001)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["wbf", "nms"],
        default="wbf",
        help="Метод объединения: wbf (рекомендуется) или nms",
    )

    args = parser.parse_args()

    input_folders = [Path(p) for p in args.input_folders]
    output_folder = Path(args.output_folder)

    for folder in input_folders:
        if not folder.exists():
            logger.error("Папка не найдена: %s", folder)
            return

    ensemble_detections(
        input_folders=input_folders,
        output_folder=output_folder,
        weights=args.weights,
        iou_thr=args.iou_thr,
        skip_box_thr=args.skip_box_thr,
        method=args.method,
    )


if __name__ == "__main__":
    main()
