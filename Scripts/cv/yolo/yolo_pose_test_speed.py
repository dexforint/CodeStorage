#!/usr/bin/env python3
"""
Бенчмарк кастомной YOLO Pose на Jetson Orin Nano.

Сравнивает:
  - PyTorch FP32 / FP16  (± fuse)
  - ONNX FP32 / FP16
  - TensorRT FP32 / FP16 / INT8

Визуализация не строится. В конце печатается сравнительная таблица.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# =============================================================================
# Настройки
# =============================================================================
MODEL_PATH = "yolov8n-pose5_v2.pt"
VIDEO_PATH = "test_video.mp4"

IMGSZ = 640  # должен совпадать с размером при обучении/экспорте
MAX_FRAMES = 80  # сколько кадров мерить (одни и те же для всех методов)
WARMUP_FRAMES = 15  # прогрев GPU / TensorRT
TRT_WORKSPACE_GB = 1  # лимит workspace TensorRT, гигабайты (Orin Nano — мало RAM)
ENGINE_DIR = Path("./trt_engines")
CALIB_DIR = Path("./int8_calib")
INT8_CALIB_IMAGES = 64  # кадры из вашего видео для калибровки INT8

# Какие методики гонять. Поставьте False, чтобы пропустить долгий экспорт.
RUN_PYTORCH = True
RUN_ONNX = True
RUN_TENSORRT = True
RUN_TENSORRT_INT8 = True  # самый долгий экспорт, нужна калибровка

# =============================================================================
# Служебные структуры
# =============================================================================
Predictor = Callable[[np.ndarray], None]


@dataclass
class BenchResult:
    name: str
    ok: bool
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    error: str = ""

    @property
    def avg_ms(self) -> float:
        return float(self.times.mean() * 1000) if self.ok else float("nan")

    @property
    def std_ms(self) -> float:
        return float(self.times.std() * 1000) if self.ok else float("nan")

    @property
    def min_ms(self) -> float:
        return float(self.times.min() * 1000) if self.ok else float("nan")

    @property
    def max_ms(self) -> float:
        return float(self.times.max() * 1000) if self.ok else float("nan")

    @property
    def fps(self) -> float:
        return (
            float(1.0 / self.times.mean())
            if self.ok and self.times.mean() > 0
            else float("nan")
        )


def log(msg: str) -> None:
    print(msg, flush=True)


def cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_system_info() -> None:
    log("=" * 72)
    log("Система")
    log("=" * 72)
    log(f"PyTorch:     {torch.__version__}")
    log(f"CUDA:        {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Проверьте сборку PyTorch для Jetson.")
    log(f"GPU:         {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    log(f"VRAM:        {free / 1024**3:.2f} / {total / 1024**3:.2f} ГБ свободно")
    log(f"Модель:      {MODEL_PATH}")
    log(f"Видео:       {VIDEO_PATH}")
    log(f"imgsz:       {IMGSZ}")
    log("Совет: перед замером выполните `sudo jetson_clocks` и выставьте MAXN/25W.")
    log("=" * 72)


def load_frames(video_path: str, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError("В видео нет кадров.")
    log(f"Загружено кадров: {len(frames)} (в файле ~{total})")
    return frames


def prepare_int8_calib(frames: list[np.ndarray], calib_dir: Path, n: int) -> str:
    """Калибровочный мини-датасет из кадров вашего видео (лучше, чем coco8)."""
    img_dir = calib_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    n = min(n, len(frames))
    idxs = np.linspace(0, len(frames) - 1, n, dtype=int)
    for i, idx in enumerate(idxs):
        cv2.imwrite(str(img_dir / f"calib_{i:04d}.jpg"), frames[idx])

    yaml_path = calib_dir / "calib.yaml"
    yaml_path.write_text(
        f"path: {calib_dir.resolve()}\n"
        "train: images\n"
        "val: images\n"
        "names:\n  0: person\n",
        encoding="utf-8",
    )
    log(f"INT8 калибровка: {n} кадров -> {yaml_path}")
    return str(yaml_path)


def export_artifact(
    model_path: str,
    dest: Path,
    fmt: str,
    half: bool = False,
    int8: bool = False,
    data: Optional[str] = None,
) -> Path:
    """Экспорт с кэшированием. dest — итоговый файл (.onnx / .engine)."""
    if dest.exists():
        log(f"  кэш найден: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  экспорт {fmt} -> {dest}  (half={half}, int8={int8})")

    model = YOLO(model_path)
    kwargs = dict(
        format=fmt,
        imgsz=IMGSZ,
        half=half,
        int8=int8,
        dynamic=False,
        simplify=True,
        device=0,
        verbose=True,
        nms=True,
    )
    if fmt == "engine":
        kwargs["workspace"] = TRT_WORKSPACE_GB
    if int8:
        if not data:
            raise ValueError(
                "Для INT8 нужен data=... yaml с калибровочными картинками."
            )
        kwargs["data"] = data

    exported = Path(model.export(**kwargs))
    del model
    cuda_cleanup()

    if exported.resolve() != dest.resolve():
        if dest.exists():
            dest.unlink()
        dest.write_bytes(exported.read_bytes())
        # исходный файл ultralytics оставляем — вдруг понадобится; можно и удалить
    log(f"  готово: {dest} ({dest.stat().st_size / 1024**2:.1f} МБ)")
    return dest


def make_predictor(
    weights: str,
    fuse: bool = False,
    half: bool = False,
) -> tuple[Predictor, object]:
    model = YOLO(weights)
    if fuse:
        model.fuse()

    def _predict(frame: np.ndarray) -> None:
        # half имеет смысл только для PyTorch; TensorRT/ONNX уже в нужной точности
        use_half = half and str(weights).endswith(".pt")
        model.predict(
            source=frame,
            imgsz=IMGSZ,
            verbose=False,
            half=use_half,
            device=0,
        )

    return _predict, model


def benchmark(name: str, predict: Predictor, frames: list[np.ndarray]) -> BenchResult:
    log(f"\n>>> {name}")
    try:
        n_warm = min(WARMUP_FRAMES, len(frames))
        for i in range(n_warm):
            predict(frames[i])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []
        for i, frame in enumerate(frames):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            predict(frame)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            if (i + 1) % 20 == 0:
                log(f"    {i + 1}/{len(frames)}")

        arr = np.asarray(times, dtype=np.float64)
        res = BenchResult(name=name, ok=True, times=arr)
        log(f"    среднее {res.avg_ms:.2f} мс  |  {res.fps:.2f} FPS")
        return res
    except Exception as exc:
        log(f"    ОШИБКА: {exc}")
        traceback.print_exc()
        return BenchResult(name=name, ok=False, error=str(exc))


def fmt(v: float, nd: int = 2) -> str:
    return "—" if v != v else f"{v:.{nd}f}"  # NaN check


def print_table(results: list[BenchResult]) -> None:
    baseline = next((r for r in results if r.ok), None)

    cols = [
        ("Метод", 28),
        ("Среднее, мс", 12),
        ("Std, мс", 9),
        ("Мин, мс", 9),
        ("Макс, мс", 9),
        ("FPS", 8),
        ("vs baseline", 12),
        ("Статус", 24),
    ]
    header = " | ".join(title.ljust(w) for title, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)

    log("\n" + "=" * len(header))
    log("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    log("=" * len(header))
    log(header)
    log(sep)

    fastest = None
    for r in results:
        if r.ok:
            speedup = (
                baseline.avg_ms / r.avg_ms
                if baseline and r.avg_ms > 0
                else float("nan")
            )
            status = "ok"
            if fastest is None or r.avg_ms < fastest.avg_ms:
                fastest = r
        else:
            speedup = float("nan")
            status = (r.error or "ошибка")[:24]

        row = [
            r.name.ljust(28),
            fmt(r.avg_ms).rjust(12),
            fmt(r.std_ms).rjust(9),
            fmt(r.min_ms).rjust(9),
            fmt(r.max_ms).rjust(9),
            fmt(r.fps).rjust(8),
            (fmt(speedup) + "x").rjust(12) if r.ok else "—".rjust(12),
            status.ljust(24),
        ]
        log(" | ".join(row))

    log("=" * len(header))
    if fastest:
        log(
            f"Самый быстрый: {fastest.name}  —  {fastest.avg_ms:.2f} мс/кадр  ({fastest.fps:.2f} FPS)"
        )
    log("baseline = первый успешно прошедший метод (обычно PyTorch FP32).")
    log("Время — чистый инференс одного кадра (I/O и отрисовка не входят).")


def run() -> list[BenchResult]:
    print_system_info()
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")
    if not Path(VIDEO_PATH).exists():
        raise FileNotFoundError(f"Видео не найдено: {VIDEO_PATH}")

    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True

    frames = load_frames(VIDEO_PATH, MAX_FRAMES)
    stem = Path(MODEL_PATH).stem
    results: list[BenchResult] = []

    def run_pt(name: str, fuse: bool, half: bool) -> None:
        predict, model = make_predictor(MODEL_PATH, fuse=fuse, half=half)
        results.append(benchmark(name, predict, frames))
        del predict, model
        cuda_cleanup()

    # ----- PyTorch -----
    if RUN_PYTORCH:
        run_pt("PyTorch FP32", fuse=False, half=False)
        run_pt("PyTorch FP32 + fuse", fuse=True, half=False)
        run_pt("PyTorch FP16", fuse=False, half=True)
        run_pt("PyTorch FP16 + fuse", fuse=True, half=True)

    # ----- ONNX -----
    if RUN_ONNX:
        for half, tag in ((False, "FP32"), (True, "FP16")):
            name = f"ONNX {tag}"
            dest = ENGINE_DIR / f"{stem}_{tag.lower()}.onnx"
            try:
                path = export_artifact(MODEL_PATH, dest, fmt="onnx", half=half)
                predict, model = make_predictor(str(path))
                results.append(benchmark(name, predict, frames))
                del predict, model
            except Exception as exc:
                log(f"    ОШИБКА экспорта/запуска {name}: {exc}")
                traceback.print_exc()
                results.append(BenchResult(name=name, ok=False, error=str(exc)))
            cuda_cleanup()

    # ----- TensorRT -----
    if RUN_TENSORRT:
        for half, tag in ((False, "FP32"), (True, "FP16")):
            name = f"TensorRT {tag}"
            dest = ENGINE_DIR / f"{stem}_{tag.lower()}.engine"
            try:
                path = export_artifact(MODEL_PATH, dest, fmt="engine", half=half)
                predict, model = make_predictor(str(path))
                results.append(benchmark(name, predict, frames))
                del predict, model
            except Exception as exc:
                log(f"    ОШИБКА экспорта/запуска {name}: {exc}")
                traceback.print_exc()
                results.append(BenchResult(name=name, ok=False, error=str(exc)))
            cuda_cleanup()

    if RUN_TENSORRT and RUN_TENSORRT_INT8:
        name = "TensorRT INT8"
        dest = ENGINE_DIR / f"{stem}_int8.engine"
        try:
            calib_yaml = prepare_int8_calib(frames, CALIB_DIR, INT8_CALIB_IMAGES)
            path = export_artifact(
                MODEL_PATH, dest, fmt="engine", int8=True, data=calib_yaml
            )
            predict, model = make_predictor(str(path))
            results.append(benchmark(name, predict, frames))
            del predict, model
        except Exception as exc:
            log(f"    ОШИБКА экспорта/запуска {name}: {exc}")
            traceback.print_exc()
            results.append(BenchResult(name=name, ok=False, error=str(exc)))
        cuda_cleanup()

    print_table(results)
    return results


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        log("\nОстановлено пользователем.")
        sys.exit(130)
