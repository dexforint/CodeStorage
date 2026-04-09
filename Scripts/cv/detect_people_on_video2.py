#!/usr/bin/env python3
"""
MAXIMUM QUALITY People Detection on Video (Исправленная версия)
Двухпроходная система (Two-Pass) + YOLO11x
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict, deque

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MaximumQualityDetector:
    """
    Максимально качественная двухпроходная детекция людей на видео.
    """

    def __init__(self):
        self.model_name = "yolo12x.pt"
        self.imgsz = 1280
        self.first_pass_conf = 0.15
        self.second_pass_conf = 0.12
        self.iou = 0.45
        self.tracker = "botsort.yaml"
        self.smoothing_window = 7
        self.min_track_length = 6

        self.model = self._load_model()
        self.track_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.smoothing_window)
        )
        self.track_frames: Dict[int, List[Tuple[int, List[float]]]] = defaultdict(list)

    def _load_model(self) -> YOLO:
        """Загрузка модели с подробной диагностикой GPU."""
        logger.info(f"Загрузка модели: {self.model_name} (imgsz={self.imgsz})")

        if not torch.cuda.is_available():
            logger.error("CUDA недоступен! Максимальное качество требует GPU.")
            raise RuntimeError("CUDA is required for maximum quality mode")

        # Диагностика GPU
        device_props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {device_props.name}")
        logger.info(f"VRAM Total: {device_props.total_memory / 1024**3:.2f} GB")
        logger.info(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
        logger.info(f"CUDA Devices: {torch.cuda.device_count()}")

        model = YOLO(self.model_name)
        model.to(0)

        # Проверка выделенной памяти после загрузки модели
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        logger.info(f"VRAM Allocated after model load: {allocated:.2f} GB")

        return model

    def _smooth_and_boost(
        self, track_id: int, bbox: List[float], confidence: float, frame_id: int
    ) -> Dict:
        self.track_history[track_id].append(np.array(bbox))
        self.track_frames[track_id].append((frame_id, bbox))

        if len(self.track_history[track_id]) >= 3:
            smoothed = np.mean(self.track_history[track_id], axis=0).tolist()
        else:
            smoothed = bbox

        track_len = len(self.track_frames[track_id])
        boost_factor = min(1.0 + (track_len * 0.035), 1.35)
        boosted_conf = min(confidence * boost_factor, 0.995)

        return {
            "track_id": track_id,
            "bbox": [round(x, 2) for x in smoothed],
            "confidence": round(float(boosted_conf), 4),
            "raw_confidence": round(float(confidence), 4),
            "track_length": track_len,
        }

    def first_pass(
        self, video_path: Path, max_frames: int | None = None
    ) -> Dict[int, List]:
        logger.info("=== Первый проход: сбор треков (низкий порог) ===")
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        detections_by_frame = {}

        for frame_id in tqdm(range(total_frames), desc="Pass 1"):
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                source=frame,
                persist=True,
                conf=self.first_pass_conf,
                iou=self.iou,
                classes=[0],
                tracker=self.tracker,
                verbose=False,
                device=0,
                half=True,
                imgsz=self.imgsz,
            )

            detections_by_frame[frame_id] = []
            for result in results:
                if not result.boxes or result.boxes.id is None:
                    continue
                for box in result.boxes:
                    track_id = int(box.id.item())
                    bbox = box.xyxy[0].cpu().tolist()
                    conf = float(box.conf.item())
                    detections_by_frame[frame_id].append((track_id, bbox, conf))

        cap.release()
        return detections_by_frame

    def second_pass(
        self, video_path: Path, first_pass_data: Dict[int, List]
    ) -> Dict[str, List[Dict]]:
        logger.info("=== Второй проход: уточнение качества ===")
        cap = cv2.VideoCapture(str(video_path))
        final_detections = {}

        for frame_id in tqdm(sorted(first_pass_data.keys()), desc="Pass 2"):
            ret, frame = cap.read()
            if not ret:
                break

            current_tracks_info = first_pass_data.get(frame_id, [])
            if not current_tracks_info:
                final_detections[str(frame_id)] = []
                continue

            results = self.model.predict(
                source=frame,
                conf=self.second_pass_conf,
                iou=self.iou,
                classes=[0],
                verbose=False,
                device=0,
                half=True,
                imgsz=self.imgsz,
                augment=True,  # TTA — важно для качества
            )

            frame_results = []
            for result in results:
                if not result.boxes:
                    continue
                for box in result.boxes:
                    bbox = box.xyxy[0].cpu().tolist()
                    conf = float(box.conf.item())

                    # Сопоставление с треками из первого прохода
                    for track_id, orig_bbox, _ in current_tracks_info:
                        if self._bbox_iou(bbox, orig_bbox) > 0.5:
                            processed = self._smooth_and_boost(
                                track_id, bbox, conf, frame_id
                            )
                            if processed["track_length"] >= self.min_track_length:
                                frame_results.append(processed)
                            break

            final_detections[str(frame_id)] = frame_results

        cap.release()
        return final_detections

    @staticmethod
    def _bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        return inter / (area1 + area2 - inter + 1e-6) if (area1 + area2) > 0 else 0.0

    def run(self, video_path: Path, output_json: Path, max_frames: int | None = None):
        logger.info("=" * 70)
        logger.info("ЗАПУСК РЕЖИМА МАКСИМАЛЬНОГО КАЧЕСТВА (Two-Pass)")
        logger.info("=" * 70)

        first_pass_data = self.first_pass(video_path, max_frames)
        final_detections = self.second_pass(video_path, first_pass_data)

        output_data = {
            "metadata": {
                "video_path": str(video_path),
                "model": self.model_name,
                "imgsz": self.imgsz,
                "mode": "maximum_quality_two_pass",
                "first_pass_conf": self.first_pass_conf,
                "second_pass_conf": self.second_pass_conf,
                "tracker": self.tracker,
                "smoothing_window": self.smoothing_window,
                "min_track_length": self.min_track_length,
                "torch_version": torch.__version__,
            },
            "detections_per_frame": final_detections,
        }

        output_json.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"\nГотово! Результат сохранён:\n{output_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Максимальное качество детекции людей (Two-Pass)"
    )
    parser.add_argument("video_path", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Для тестирования")

    args = parser.parse_args()

    if args.output is None:
        args.output = args.video_path.with_name(
            f"{args.video_path.stem}_MAX_QUALITY.json"
        )

    detector = MaximumQualityDetector()
    detector.run(args.video_path, args.output, args.max_frames)
