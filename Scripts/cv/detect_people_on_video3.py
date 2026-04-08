"""
Профессиональная детекция людей на видео с разделением детекции и трекинга.

Архитектура позволяет легко заменять модель детекции (из разных библиотек),
не трогая основной пайплайн. Трекинг вынесен отдельно.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from ultralytics import YOLO

# ====================== НАСТРОЙКА ЛОГИРОВАНИЯ ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ====================== СТРУКТУРЫ ДАННЫХ ======================
@dataclass
class RawDetection:
    """Результат чистой детекции (без трекинга)."""

    bbox: List[float]  # [x1, y1, x2, y2] в абсолютных координатах
    confidence: float
    class_id: int = 0


@dataclass
class TrackedDetection:
    """Финальная детекция с идентификатором трека."""

    track_id: int
    bbox: List[float]
    confidence: float


# ====================== ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ======================
def initialize_detector(model_name: str = "yolo11x.pt", device: str = "cuda") -> YOLO:
    """
    Инициализирует модель детекции.
    Эту функцию вы будете менять при переходе на другую библиотеку/модель.
    """
    logger.info(f"Инициализация детектора: {model_name} на {device}")

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA недоступен. Используется CPU.")
        device = "cpu"

    detector = YOLO(model_name)
    detector.to(device)

    logger.info("Детектор успешно инициализирован")
    return detector


def initialize_tracker() -> sv.ByteTrack:
    """
    Инициализирует трекер.
    Можно заменить на sv.BoTSORT(), sv.DeepSORT() или собственную реализацию.
    """
    logger.info("Инициализация трекера BoT-SORT (supervision)")
    # Можно поэкспериментировать с параметрами: track_thresh, lost_track_buffer и т.д.
    return sv.ByteTrack(
        track_activation_threshold=0.35,
        lost_track_buffer=60,
        minimum_matching_threshold=0.4,
        frame_rate=30,
    )


# ====================== ОСНОВНЫЕ ФУНКЦИИ (ЗАМЕНИМЫЕ) ======================
def run_inference(
    detector: YOLO, frame: np.ndarray, conf_threshold: float = 0.30
) -> List[RawDetection]:
    """
    Выполняет только детекцию людей.
    НЕ занимается трекингом.

    При смене библиотеки (Detectron2, MMDetection и т.д.) меняете только эту функцию.
    """
    results = detector(
        source=frame,
        conf=conf_threshold,
        classes=[0],  # 0 = person в COCO
        imgsz=1280,
        verbose=False,
    )

    raw_detections: List[RawDetection] = []
    result = results[0]

    if not result.boxes:
        return raw_detections

    boxes = result.boxes

    for box in boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        raw_detections.append(
            RawDetection(
                bbox=[x1, y1, x2, y2], confidence=round(conf, 4), class_id=cls_id
            )
        )

    return raw_detections


def convert_to_supervision_format(
    detections: List[RawDetection],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Преобразует наши детекции в формат supervision."""
    if not detections:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=int),
        )

    xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
    confidence = np.array([d.confidence for d in detections], dtype=np.float32)
    class_id = np.array([d.class_id for d in detections], dtype=int)

    return xyxy, confidence, class_id


def update_tracker(
    tracker: sv.ByteTrack, raw_detections: List[RawDetection], frame: np.ndarray
) -> List[TrackedDetection]:
    """
    Обновляет трекер и возвращает детекции с track_id.
    """
    xyxy, confidence, class_id = convert_to_supervision_format(raw_detections)

    # supervision.Detections — основной формат библиотеки
    sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    # Применяем трекинг
    tracked_sv_detections = tracker.update_with_detections(detections=sv_detections)

    tracked: List[TrackedDetection] = []

    for i, track_id in enumerate(tracked_sv_detections.tracker_id):
        if track_id is None:
            continue
        x1, y1, x2, y2 = tracked_sv_detections.xyxy[i].tolist()
        conf = float(tracked_sv_detections.confidence[i])

        tracked.append(
            TrackedDetection(
                track_id=int(track_id), bbox=[x1, y1, x2, y2], confidence=round(conf, 4)
            )
        )

    return tracked


# ====================== ОСНОВНОЙ ПАЙПЛАЙН ======================
def process_video(
    video_path: str | Path,
    output_json: str = "detections.json",
    conf_threshold: float = 0.30,
    skip_frames: int = 0,
    device: str = "cuda",
) -> None:
    """
    Основная функция обработки видео.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    detector = initialize_detector(model_name="yolo11x.pt", device=device)
    tracker = initialize_tracker()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        f"Запуск обработки: {video_path.name} | Разрешение: {width}x{height} | "
        f"FPS: {fps:.2f} | Кадров: {total_frames}"
    )

    detections_per_frame: List[Dict[str, Any]] = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc="Обработка видео", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        # === ДЕТЕКЦИЯ ===
        raw_detections = run_inference(detector, frame, conf_threshold)

        # === ТРЕКИНГ ===
        tracked_detections = update_tracker(tracker, raw_detections, frame)

        detections_per_frame.append(
            {
                "frame_idx": frame_idx,
                "detections": [d.__dict__ for d in tracked_detections],
            }
        )

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    result_data = {
        "metadata": {
            "video_path": str(video_path.absolute()),
            "fps": float(fps),
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "detector": "yolo11x",
            "tracker": "BoT-SORT (supervision)",
            "imgsz": 1280,
            "conf_threshold": conf_threshold,
        },
        "detections_per_frame": detections_per_frame,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Обработка завершена. Результат сохранён: {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Детекция людей на видео (детекция + трекинг раздельно)"
    )
    parser.add_argument("video_path", type=str, help="Путь к видеофайлу")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="detections.json",
        help="Путь к JSON результату",
    )
    parser.add_argument("--conf", type=float, default=0.30, help="Порог уверенности")
    parser.add_argument("--skip", type=int, default=0, help="Пропускать N кадров")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    try:
        process_video(
            video_path=args.video_path,
            output_json=args.output,
            conf_threshold=args.conf,
            skip_frames=args.skip,
            device=args.device,
        )
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Необходимые зависимости:
    # pip install ultralytics supervision opencv-python tqdm torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    main()
