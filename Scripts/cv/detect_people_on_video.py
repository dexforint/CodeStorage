#!/usr/bin/env python3
"""
Улучшенная версия: YOLO + Трекинг + Постобработка для видео
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict, deque

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class VideoPersonDetector:
    """Класс для качественной детекции людей на видео с использованием трекинга."""

    def __init__(
        self,
        model_name: str = "yolo11m.pt",
        conf_threshold: float = 0.20,  # ниже, чем обычно — трекер поможет
        iou_threshold: float = 0.45,
        tracker: str = "botsort.yaml",  # "bytetrack.yaml" или "botsort.yaml"
        smoothing_window: int = 5,
        min_track_length: int = 3,
    ):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker = tracker
        self.smoothing_window = smoothing_window
        self.min_track_length = min_track_length

        self.model = self._load_model()
        self.track_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=smoothing_window)
        )

    def _load_model(self) -> YOLO:
        logger.info(f"Загрузка модели {self.model_name}...")
        model = YOLO(self.model_name)
        if torch.cuda.is_available():
            model.to(0)
            logger.info(f"Используется GPU: {torch.cuda.get_device_name(0)}")
        return model

    def _smooth_bbox(self, track_id: int, bbox: List[float]) -> List[float]:
        """Сглаживает bbox с помощью скользящего среднего."""
        self.track_history[track_id].append(bbox)
        if len(self.track_history[track_id]) < 2:
            return bbox

        smoothed = np.mean(self.track_history[track_id], axis=0)
        return [round(float(x), 2) for x in smoothed]

    def process_video(
        self,
        video_path: Path,
        output_json: Path,
        skip_frames: int = 1,
        max_frames: int | None = None,
    ) -> None:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        logger.info(f"Начало обработки: {video_path.name} ({total_frames} кадров)")

        frame_detections = {}
        track_lengths: Dict[int, int] = defaultdict(int)

        for frame_id in tqdm(range(total_frames), desc="Обработка видео"):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % skip_frames != 0:
                continue

            # Используем track() — это ключевой момент улучшения
            results = self.model.track(
                source=frame,
                persist=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],
                tracker=self.tracker,
                verbose=False,
                device=0 if torch.cuda.is_available() else "cpu",
                half=True,
            )

            current_frame_dets = []

            for result in results:
                if (
                    not result.boxes
                    or not hasattr(result.boxes, "id")
                    or result.boxes.id is None
                ):
                    continue

                boxes = result.boxes
                for i, box in enumerate(boxes):
                    track_id = int(box.id.item())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    conf = float(box.conf.item())

                    bbox = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                    smoothed_bbox = self._smooth_bbox(track_id, bbox)

                    track_lengths[track_id] += 1
                    track_length = track_lengths[track_id]

                    # Boosting уверенности для длинных треков
                    boosted_conf = min(conf * (1.0 + 0.1 * min(track_length, 15)), 0.99)

                    if track_length >= self.min_track_length:
                        current_frame_dets.append(
                            {
                                "track_id": track_id,
                                "bbox": smoothed_bbox,  # используем сглаженный
                                "raw_bbox": bbox,
                                "confidence": round(boosted_conf, 4),
                                "raw_confidence": round(conf, 4),
                                "track_length": track_length,
                            }
                        )

            frame_detections[str(frame_id)] = current_frame_dets

        cap.release()

        # Сохранение
        output_data = {
            "metadata": {
                "video_path": str(video_path),
                "fps": round(fps, 3),
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "model": self.model_name,
                "tracker": self.tracker,
                "conf_threshold": self.conf_threshold,
                "smoothing_window": self.smoothing_window,
                "min_track_length": self.min_track_length,
            },
            "detections_per_frame": frame_detections,
        }

        output_json.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Обработка завершена. Результат сохранён: {output_json}")


# ====================== Запуск ======================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Улучшенная детекция людей на видео (с трекингом)"
    )
    parser.add_argument("video_path", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "-m",
        "--model",
        default="yolo11m.pt",
        help="yolo11m.pt / yolo11x.pt / yolov8x.pt",
    )
    parser.add_argument(
        "--tracker", choices=["botsort.yaml", "bytetrack.yaml"], default="botsort.yaml"
    )
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument(
        "--smoothing", type=int, default=5, help="Окно сглаживания bbox"
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = args.video_path.with_name(f"{args.video_path.stem}_tracked.json")

    detector = VideoPersonDetector(
        model_name=args.model,
        conf_threshold=args.conf,
        tracker=args.tracker,
        smoothing_window=args.smoothing,
        min_track_length=4,
    )

    detector.process_video(args.video_path, args.output, skip_frames=args.skip)
