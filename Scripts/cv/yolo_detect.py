#!/usr/bin/env python3
"""
YOLO Detection Script — обработка видео, изображений и веб-камеры.
Работает на Windows и Linux с поддержкой GPU.

Использование:
    python yolo_detect.py --source image.jpg
    python yolo_detect.py --source video.mp4
    python yolo_detect.py --source webcam
    python yolo_detect.py --source webcam --cam-id 1
    python yolo_detect.py --source video.mp4 --model yolov8x.pt --conf 0.5
"""

import argparse
import sys
import os
import time
import platform
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ──────────────────────────── Константы ────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}

# Цвета для боксов (BGR)
COLORS = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 128, 0),
    (0, 255, 128),
    (128, 255, 0),
    (255, 0, 128),
    (64, 224, 208),
    (255, 165, 0),
    (75, 0, 130),
    (240, 128, 128),
]


# ──────────────────────────── Утилиты ──────────────────────────────


def get_device() -> str:
    """Определяет лучшее доступное устройство: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU]  {gpu_name} ({vram:.1f} GB VRAM)")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[GPU]  Apple MPS")
        return "mps"
    else:
        print("[CPU]  GPU не найден, используется CPU")
        return "cpu"


def get_color(class_id: int) -> tuple:
    """Возвращает цвет для класса."""
    return COLORS[class_id % len(COLORS)]


def build_output_path(source_path: str, suffix: str = "_detected") -> Path:
    """Строит путь для выходного файла рядом с исходным."""
    p = Path(source_path)
    return p.parent / f"{p.stem}{suffix}{p.suffix}"


def draw_boxes(
    frame: np.ndarray,
    results,
    model: YOLO,
    conf_threshold: float = 0.25,
    line_width: int = 2,
) -> np.ndarray:
    """
    Рисует bounding-боксы, метки и confidence на кадре.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Адаптивный размер шрифта
    font_scale = max(0.4, min(w, h) / 1200)
    thickness = max(1, line_width)
    text_thickness = max(1, thickness - 1)

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label != "person":
                continue

            color = get_color(cls_id)

            # Координаты бокса
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Бокс
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Текст с фоном
            text = f"{label} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )
            # Фон для текста
            cv2.rectangle(
                annotated, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1
            )
            # Текст
            cv2.putText(
                annotated,
                text,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

    return annotated


def print_detection_summary(results, model: YOLO):
    """Выводит сводку обнаружений."""
    counts = {}
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            counts[name] = counts.get(name, 0) + 1

    if counts:
        summary = ", ".join(f"{name}: {cnt}" for name, cnt in sorted(counts.items()))
        print(f"    Обнаружено: {summary}")
    else:
        print("    Ничего не обнаружено")


# ──────────────────────── Обработка изображения ────────────────────


def process_image(
    model: YOLO,
    source: str,
    conf: float = 0.25,
    iou: float = 0.45,
    line_width: int = 2,
    save_txt: bool = False,
):
    """Детекция на одном изображении."""
    print(f"\n{'='*60}")
    print(f"  Обработка изображения: {source}")
    print(f"{'='*60}")

    img = cv2.imread(source)
    if img is None:
        print(f"[ОШИБКА] Не удалось прочитать: {source}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"  Размер: {w}x{h}")

    # Инференс
    t0 = time.time()
    results = model.predict(
        source=img,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    dt = time.time() - t0
    print(f"  Время инференса: {dt*1000:.1f} мс")

    print_detection_summary(results, model)

    # Рисуем боксы
    annotated = draw_boxes(img, results, model, conf, line_width)

    # Сохраняем
    output_path = build_output_path(source, "_detected")
    cv2.imwrite(str(output_path), annotated)
    print(f"  Сохранено: {output_path}")

    # Сохраняем аннотации в txt
    if save_txt:
        txt_path = output_path.with_suffix(".txt")
        save_annotations_txt(results, model, txt_path)
        print(f"  Аннотации: {txt_path}")

    # Показываем (если есть дисплей)
    try:
        cv2.imshow("YOLO Detection", annotated)
        print("  Нажмите любую клавишу для закрытия...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("  (Дисплей недоступен, показ пропущен)")


# ──────────────────────── Обработка видео ──────────────────────────


def process_video(
    model: YOLO,
    source: str,
    conf: float = 0.25,
    iou: float = 0.45,
    line_width: int = 2,
    show: bool = True,
    save_txt: bool = False,
):
    """Детекция на видеофайле с сохранением результата."""
    print(f"\n{'='*60}")
    print(f"  Обработка видео: {source}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ОШИБКА] Не удалось открыть: {source}")
        sys.exit(1)

    # Параметры видео
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Размер: {w}x{h}, FPS: {fps:.1f}, Кадров: {total_frames}")

    # Выходной файл
    output_path = build_output_path(source, "_detected")

    # Кодек — выбираем совместимый
    ext = output_path.suffix.lower()
    if ext == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    elif ext == ".mkv":
        fourcc = cv2.VideoWriter_fourcc(*"X264")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = output_path.with_suffix(".mp4")

    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        # Fallback кодек
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_path = output_path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    print(f"  Сохранение в: {output_path}")
    print(f"  Нажмите 'q' для остановки\n")

    frame_idx = 0
    fps_counter = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            t0 = time.time()

            # Инференс
            results = model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                verbose=False,
            )

            # Рисуем боксы
            annotated = draw_boxes(frame, results, model, conf, line_width)

            dt = time.time() - t0
            current_fps = 1.0 / dt if dt > 0 else 0
            fps_counter.append(current_fps)

            # Информация на кадре
            info_text = f"FPS: {current_fps:.1f} | Frame: {frame_idx}/{total_frames}"
            cv2.putText(
                annotated,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated)

            # Прогресс в консоли
            progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            sys.stdout.write(
                f"\r  [{progress:5.1f}%] Кадр {frame_idx}/{total_frames} "
                f"| FPS: {current_fps:.1f} "
                f"| Среднее: {np.mean(fps_counter):.1f} FPS"
            )
            sys.stdout.flush()

            # Показ
            if show:
                try:
                    cv2.imshow("YOLO Detection — Video", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n\n  Остановлено пользователем")
                        break
                except cv2.error:
                    show = False

    except KeyboardInterrupt:
        print("\n\n  Прервано (Ctrl+C)")

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    avg_fps = np.mean(fps_counter) if fps_counter else 0
    print(f"\n\n  Готово! Обработано кадров: {frame_idx}")
    print(f"  Средний FPS: {avg_fps:.1f}")
    print(f"  Результат: {output_path}")


# ──────────────────────── Веб-камера ───────────────────────────────


def process_webcam(
    model: YOLO,
    cam_id: int = 0,
    conf: float = 0.25,
    iou: float = 0.45,
    line_width: int = 2,
    save: bool = True,
    save_dir: Optional[str] = None,
):
    """Детекция в реальном времени с веб-камеры."""
    print(f"\n{'='*60}")
    print(f"  Веб-камера (ID: {cam_id})")
    print(f"{'='*60}")

    # Выбор бэкенда захвата
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print(f"[ОШИБКА] Не удалось открыть камеру {cam_id}")
        print("  Попробуйте указать другой --cam-id")
        sys.exit(1)

    # Настройка камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"  Разрешение: {w}x{h}, FPS камеры: {fps_cam:.0f}")

    # Запись
    writer = None
    output_path = None
    if save:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base_dir = Path(save_dir)
        else:
            base_dir = Path(".")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = base_dir / f"webcam_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_cam, (w, h))
        print(f"  Запись в: {output_path}")

    print()
    print("  Управление:")
    print("    q     — выход")
    print("    s     — сохранить скриншот")
    print("    p     — пауза/продолжить")
    print("    +/-   — изменить порог confidence")
    print()

    frame_idx = 0
    fps_counter = []
    paused = False
    screenshot_idx = 0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("  Потеряно соединение с камерой")
                    break

                frame_idx += 1
                t0 = time.time()

                # Инференс
                results = model.predict(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    verbose=False,
                )

                # Рисуем боксы
                annotated = draw_boxes(frame, results, model, conf, line_width)

                dt = time.time() - t0
                current_fps = 1.0 / dt if dt > 0 else 0
                fps_counter.append(current_fps)
                if len(fps_counter) > 60:
                    fps_counter.pop(0)

                avg_fps = np.mean(fps_counter)

                # Информация на кадре
                info_lines = [
                    f"FPS: {current_fps:.0f} (avg: {avg_fps:.0f})",
                    f"Conf: {conf:.2f}",
                ]
                if writer:
                    info_lines.append("REC")

                for i, line in enumerate(info_lines):
                    color = (0, 0, 255) if line == "REC" else (0, 255, 0)
                    cv2.putText(
                        annotated,
                        line,
                        (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                # Запись
                if writer:
                    writer.write(annotated)

                # Прогресс
                sys.stdout.write(
                    f"\r  Кадр {frame_idx} | FPS: {current_fps:.0f} "
                    f"| Avg: {avg_fps:.0f} | Conf: {conf:.2f}"
                )
                sys.stdout.flush()

            # Показ
            try:
                cv2.imshow("YOLO Detection — Webcam", annotated)
            except cv2.error:
                print("\n  [ОШИБКА] Дисплей недоступен")
                break

            # Управление клавишами
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # q или Esc
                print("\n\n  Выход")
                break

            elif key == ord("s"):
                screenshot_idx += 1
                ss_path = Path(".") / f"screenshot_{screenshot_idx}.jpg"
                cv2.imwrite(str(ss_path), annotated)
                print(f"\n  Скриншот сохранён: {ss_path}")

            elif key == ord("p"):
                paused = not paused
                state = "ПАУЗА" if paused else "ПРОДОЛЖЕНИЕ"
                print(f"\n  {state}")

            elif key == ord("+") or key == ord("="):
                conf = min(0.95, conf + 0.05)
                print(f"\n  Conf threshold: {conf:.2f}")

            elif key == ord("-"):
                conf = max(0.05, conf - 0.05)
                print(f"\n  Conf threshold: {conf:.2f}")

    except KeyboardInterrupt:
        print("\n\n  Прервано (Ctrl+C)")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    if output_path and output_path.exists():
        print(f"  Запись сохранена: {output_path}")
    print(f"  Всего кадров: {frame_idx}")


# ──────────────────────── Сохранение аннотаций ─────────────────────


def save_annotations_txt(results, model: YOLO, txt_path: Path):
    """Сохраняет аннотации в формате YOLO txt."""
    lines = []
    for result in results:
        if result.boxes is None:
            continue
        img_h, img_w = result.orig_shape
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Конвертация в YOLO формат (нормализованные xywh)
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h

            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))


# ──────────────────────── Точка входа ──────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Object Detection — видео, изображения, веб-камера",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python yolo_detect.py --source photo.jpg
  python yolo_detect.py --source video.mp4 --model yolov8x.pt
  python yolo_detect.py --source webcam --cam-id 0
  python yolo_detect.py --source video.mp4 --conf 0.5 --no-show
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Путь к изображению/видео или 'webcam'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/yolov8n.pt",
        help="Модель YOLO (default: yolov8n.pt). "
        "Варианты: yolov8n/s/m/l/x.pt, yolov8n-seg.pt, yolo11n.pt и т.д.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Порог confidence (default: 0.25)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="Порог IoU для NMS (default: 0.45)"
    )
    parser.add_argument(
        "--line-width", type=int, default=2, help="Толщина линий боксов (default: 2)"
    )
    parser.add_argument(
        "--cam-id", type=int, default=0, help="ID веб-камеры (default: 0)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Не показывать окно (для серверов без GUI)",
    )
    parser.add_argument(
        "--no-save-webcam", action="store_true", help="Не сохранять видео с веб-камеры"
    )
    parser.add_argument(
        "--save-txt", action="store_true", help="Сохранять аннотации в txt"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./data",
        help="Директория для сохранения (webcam)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  YOLO Object Detection")
    print("=" * 60)
    print(f"  Система:   {platform.system()} {platform.machine()}")
    print(f"  Python:    {platform.python_version()}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  OpenCV:    {cv2.__version__}")

    # Устройство
    device = get_device()

    # Загрузка модели
    print(f"\n  Загрузка модели: {args.model}")
    try:
        model = YOLO(args.model)
        model.to(device)
    except Exception as e:
        print(f"[ОШИБКА] Не удалось загрузить модель: {e}")
        sys.exit(1)

    print(f"  Классов: {len(model.names)}")
    print(f"  Устройство: {device}")

    # Прогрев модели (warmup)
    print("  Прогрев модели...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(source=dummy, verbose=False)
    print("  Готово!\n")

    # Определяем тип источника
    source = args.source.strip()

    if source.lower() in ("webcam", "camera", "cam", "0"):
        # ──── Веб-камера ────
        process_webcam(
            model=model,
            cam_id=args.cam_id,
            conf=args.conf,
            iou=args.iou,
            line_width=args.line_width,
            save=not args.no_save_webcam,
            save_dir=args.save_dir,
        )

    elif os.path.isfile(source):
        ext = Path(source).suffix.lower()

        if ext in IMAGE_EXTENSIONS:
            # ──── Изображение ────
            process_image(
                model=model,
                source=source,
                conf=args.conf,
                iou=args.iou,
                line_width=args.line_width,
                save_txt=args.save_txt,
            )

        elif ext in VIDEO_EXTENSIONS:
            # ──── Видео ────
            process_video(
                model=model,
                source=source,
                conf=args.conf,
                iou=args.iou,
                line_width=args.line_width,
                show=not args.no_show,
                save_txt=args.save_txt,
            )

        else:
            print(f"[ОШИБКА] Неподдерживаемый формат: {ext}")
            print(f"  Изображения: {IMAGE_EXTENSIONS}")
            print(f"  Видео: {VIDEO_EXTENSIONS}")
            sys.exit(1)

    else:
        print(f"[ОШИБКА] Файл не найден: {source}")
        sys.exit(1)

    print("\n  Завершено!\n")


if __name__ == "__main__":
    main()
