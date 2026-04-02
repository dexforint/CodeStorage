#!/usr/bin/env python3
"""
Скрипт для автоматического вырезания фрагментов видео, содержащих людей.

Алгоритм:
  1) Первый проход — YOLO-детекция + трекинг людей по всему видео.
  2) Фильтрация коротких треков (вероятные ложные срабатывания).
  3) Построение временных сегментов, расширение на ±1 сек, объединение.
  4) Второй проход — запись выходного видео только из нужных сегментов
     с отрисованными bounding box'ами, ID треков и траекториями.

Установка зависимостей:
    pip install ultralytics opencv-python tqdm numpy

Примеры запуска:
    python extract_people.py video.mp4
    python extract_people.py video.mp4 -o result.mp4 --confidence 0.5
    python extract_people.py video.mp4 --model yolov8s.pt --min-track 20
"""

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict
import argparse
import sys
import os


# ════════════════════════════════════════════════════════════════════
#  Вспомогательные функции
# ════════════════════════════════════════════════════════════════════


def select_device():
    """
    Определяет доступное устройство для вычислений.
    Возвращает 0 (индекс GPU) или 'cpu'.
    """
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  [✓] GPU найден: {name}")
            return 0  # Индекс первой видеокарты
        else:
            print("  [✗] CUDA GPU не обнаружен, используется CPU")
            return "cpu"
    except ImportError:
        print("  [✗] PyTorch не найден, используется CPU")
        return "cpu"


def get_track_color(track_id: int) -> tuple:
    """
    Генерирует уникальный яркий цвет для каждого трека.
    Использует золотое сечение для равномерного распределения
    оттенков в HSV-пространстве.
    """
    golden_ratio = 0.618033988749895
    # Вычисляем оттенок через золотое сечение (0–179 для OpenCV HSV)
    hue = int((track_id * golden_ratio * 180) % 180)
    # Создаём яркий насыщенный цвет
    hsv_pixel = np.uint8([[[hue, 230, 230]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr_pixel[0][0])


def format_time(seconds: float) -> str:
    """Форматирует секунды в строку вида H:MM:SS.ss или M:SS.ss."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:05.2f}"
    return f"{m}:{s:05.2f}"


# ════════════════════════════════════════════════════════════════════
#  Построение сегментов
# ════════════════════════════════════════════════════════════════════


def build_segments(
    frames_with_people: list,
    total_frames: int,
    fps: float,
    extend_sec: float = 1.0,
    min_duration_sec: float = 0.5,
    gap_merge_sec: float = 2.0,
) -> list:
    """
    Строит сегменты видео из списка кадров, содержащих людей.

    Алгоритм:
      1. Группирует кадры с детекциями в непрерывные блоки,
         объединяя соседние, если промежуток между ними ≤ gap_merge_sec.
      2. Отбрасывает блоки короче min_duration_sec.
      3. Расширяет каждый блок на extend_sec вперёд и назад.
      4. Объединяет пересекающиеся блоки.

    Параметры:
        frames_with_people  — отсортированный список индексов кадров
        total_frames        — общее число кадров в видео
        fps                 — частота кадров
        extend_sec          — расширение сегмента (сек)
        min_duration_sec    — минимальная длительность сегмента (сек)
        gap_merge_sec       — максимальный промежуток для склейки (сек)

    Возвращает:
        Список кортежей (start_frame, end_frame).
    """
    if not frames_with_people:
        return []

    gap_frames = int(gap_merge_sec * fps)
    ext_frames = int(extend_sec * fps)
    min_frames = int(min_duration_sec * fps)

    # --- Шаг 1: группировка кадров в непрерывные блоки ---
    segments = []
    seg_start = frames_with_people[0]
    seg_end = frames_with_people[0]

    for fidx in frames_with_people[1:]:
        # Если промежуток до следующего кадра с детекцией небольшой — расширяем
        if fidx - seg_end <= gap_frames:
            seg_end = fidx
        else:
            # Иначе — закрываем текущий блок и начинаем новый
            segments.append((seg_start, seg_end))
            seg_start = fidx
            seg_end = fidx

    segments.append((seg_start, seg_end))

    # --- Шаг 2: отбрасываем слишком короткие блоки ---
    segments = [(s, e) for s, e in segments if (e - s + 1) >= min_frames]

    if not segments:
        return []

    # --- Шаг 3: расширяем каждый блок на extend_sec ---
    expanded = []
    for s, e in segments:
        new_s = max(0, s - ext_frames)
        new_e = min(total_frames - 1, e + ext_frames)
        expanded.append((new_s, new_e))

    # --- Шаг 4: объединяем пересекающиеся блоки ---
    merged = [expanded[0]]
    for s, e in expanded[1:]:
        # Если начало нового блока ≤ конца предыдущего — сливаем
        if s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    return merged


# ════════════════════════════════════════════════════════════════════
#  Отрисовка аннотаций
# ════════════════════════════════════════════════════════════════════


def draw_annotations(
    frame,
    detections: list,
    frame_idx: int,
    fps: float,
    seg_idx: int,
    total_segments: int,
    track_centers: dict,
    trail_length: int = 30,
):
    """
    Рисует на кадре:
      • bounding box каждого человека (цвет уникален для трека);
      • ID трека и уверенность детекции;
      • траекторию (след) последних trail_length позиций центра;
      • общую информацию о сегменте в левом верхнем углу.

    Параметры:
        frame           — кадр (numpy array, BGR)
        detections      — список кортежей (track_id, x1, y1, x2, y2, conf)
        frame_idx       — номер кадра в исходном видео
        fps             — частота кадров
        seg_idx         — индекс текущего сегмента (с 0)
        total_segments  — общее количество сегментов
        track_centers   — dict: track_id → {frame_idx: (cx, cy)}
        trail_length    — длина отображаемого следа (в кадрах)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        tid, x1, y1, x2, y2, conf = det
        color = get_track_color(tid)

        # ─── Траектория (полилиния последних позиций центра) ───
        if tid in track_centers:
            # Собираем точки центра за последние trail_length кадров
            trail_pts = []
            start_f = max(0, frame_idx - trail_length)
            centers_map = track_centers[tid]
            for f in range(start_f, frame_idx + 1):
                if f in centers_map:
                    trail_pts.append(centers_map[f])

            # Рисуем линии с нарастающей толщиной (чем новее — тем толще)
            for i in range(1, len(trail_pts)):
                progress = i / len(trail_pts)  # 0 → 1
                thickness = max(1, int(progress * 4))
                cv2.line(frame, trail_pts[i - 1], trail_pts[i], color, thickness)

        # ─── Bounding box ───
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(frame, pt1, pt2, color, 2)

        # ─── Точка в центре бокса ───
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)

        # ─── Подпись (ID трека + уверенность) ───
        label = f"ID:{tid} {conf:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, font, 0.6, 2)
        # Фон под текстом
        cv2.rectangle(
            frame,
            (pt1[0], pt1[1] - th - baseline - 6),
            (pt1[0] + tw + 6, pt1[1]),
            color,
            -1,
        )
        # Белый текст поверх фона
        cv2.putText(
            frame,
            label,
            (pt1[0] + 3, pt1[1] - baseline - 3),
            font,
            0.6,
            (255, 255, 255),
            2,
        )

    # ─── Информационный оверлей в левом верхнем углу ───
    info_lines = [
        f"Segment {seg_idx + 1}/{total_segments}",
        f"Source: {format_time(frame_idx / fps)}  (frame {frame_idx})",
    ]
    if detections:
        info_lines.append(f"People: {len(detections)}")

    for i, line in enumerate(info_lines):
        y = 30 + i * 30
        # Тёмная тень для читаемости на любом фоне
        cv2.putText(frame, line, (12, y + 2), font, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, line, (10, y), font, 0.7, (0, 255, 0), 2)

    return frame


# ════════════════════════════════════════════════════════════════════
#  Главная функция
# ════════════════════════════════════════════════════════════════════


def main():
    # ── Парсинг аргументов командной строки ──
    parser = argparse.ArgumentParser(
        description="Вырезание фрагментов видео с людьми (YOLO + трекинг)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Путь к входному видео файлу")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Путь к выходному видео (по умолчанию: <input>_people.mp4)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="yolov8m.pt",
        help="Модель YOLO (скачается автоматически при первом запуске)",
    )
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.40,
        help="Порог уверенности детекции (0–1)",
    )
    parser.add_argument(
        "--extend",
        type=float,
        default=1.0,
        help="Расширение каждого сегмента вперёд и назад (сек)",
    )
    parser.add_argument(
        "--min-segment",
        type=float,
        default=0.5,
        help="Минимальная длительность сегмента (сек); более короткие отбрасываются",
    )
    parser.add_argument(
        "--min-track",
        type=int,
        default=10,
        help="Минимальная длина трека (кадры); более короткие считаются ложными",
    )
    parser.add_argument(
        "--gap-merge",
        type=float,
        default=2.0,
        help="Макс. промежуток между детекциями для склейки в один сегмент (сек)",
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=30,
        help="Длина визуального следа трекинга (кадры; 0 = отключить)",
    )
    args = parser.parse_args()

    input_path = args.input

    # Проверяем существование входного файла
    if not os.path.isfile(input_path):
        print(f"Ошибка: файл '{input_path}' не найден!")
        sys.exit(1)

    # Формируем имя выходного файла, если не указано явно
    if args.output is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_people.mp4"
    else:
        output_path = args.output

    # ── Выбор устройства (GPU / CPU) ──
    print("\n╔══════════════════════════════════════════════╗")
    print("║   Извлечение фрагментов видео с людьми      ║")
    print("╚══════════════════════════════════════════════╝")
    print("\n[1/5] Выбор устройства...")
    device = select_device()

    # ── Загрузка модели YOLO ──
    print(f"\n[2/5] Загрузка модели: {args.model}")
    model = YOLO(args.model)
    print("  [✓] Модель загружена")

    # ── Открытие входного видео ──
    print(f"\n[3/5] Открытие видео: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  Ошибка: не удалось открыть '{input_path}'!")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"  Разрешение : {width}×{height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Кадров     : {total_frames}")
    print(f"  Длительность: {format_time(duration_sec)} ({duration_sec:.0f} сек)")

    # ================================================================
    #  ПЕРВЫЙ ПРОХОД: детекция и трекинг людей кадр за кадром
    # ================================================================
    print(f"\n{'═'*55}")
    print("  ПЕРВЫЙ ПРОХОД: детекция и трекинг людей")
    print(f"{'═'*55}")

    # Хранилище детекций для каждого кадра:
    #   frame_detections[frame_idx] = [(track_id, x1, y1, x2, y2, conf), ...]
    frame_detections: dict[int, list] = {}

    # Счётчик: сколько кадров каждый трек был виден
    track_lengths: dict[int, int] = defaultdict(int)

    # В датасете COCO класс 0 — это "person"
    PERSON_CLASS_ID = 0

    frame_idx = 0
    pbar = tqdm(
        total=total_frames,
        desc="  Анализ",
        unit=" кадр",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}]",
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Запускаем YOLO-трекинг:
        #   persist=True  — сохраняет внутреннее состояние трекера между кадрами
        #   classes=[0]   — ищем только людей
        #   verbose=False — не печатаем лог каждого кадра
        results = model.track(
            frame,
            persist=True,
            conf=args.confidence,
            classes=[PERSON_CLASS_ID],
            verbose=False,
            device=device,
        )

        # Извлекаем результаты трекинга из текущего кадра
        detections = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            # boxes.id может быть None, если трекер ещё не назначил ID
            if boxes.id is not None:
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(boxes.conf[i].cpu())
                    track_id = int(boxes.id[i].cpu())

                    detections.append(
                        (
                            track_id,
                            float(xyxy[0]),
                            float(xyxy[1]),
                            float(xyxy[2]),
                            float(xyxy[3]),
                            conf,
                        )
                    )
                    track_lengths[track_id] += 1

        # Сохраняем детекции только если они есть
        if detections:
            frame_detections[frame_idx] = detections

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # Вывод промежуточной статистики
    det_frames = len(frame_detections)
    det_pct = det_frames / total_frames * 100 if total_frames else 0
    print(f"\n  Результаты первого прохода:")
    print(f"    Всего уникальных треков : {len(track_lengths)}")
    print(f"    Кадров с детекциями     : {det_frames} ({det_pct:.1f}%)")

    if not frame_detections:
        print("\n  Люди не обнаружены! Выходной файл не создан.")
        sys.exit(0)

    # ================================================================
    #  ФИЛЬТРАЦИЯ КОРОТКИХ ТРЕКОВ (ложные срабатывания)
    # ================================================================
    print(f"\n{'═'*55}")
    print("  ФИЛЬТРАЦИЯ ТРЕКОВ")
    print(f"{'═'*55}")

    # Оставляем только треки, присутствующие на ≥ min_track кадрах
    valid_track_ids = {
        tid for tid, length in track_lengths.items() if length >= args.min_track
    }

    removed_count = len(track_lengths) - len(valid_track_ids)
    print(f"  Порог: ≥ {args.min_track} кадров")
    print(f"  Удалено коротких треков : {removed_count}")
    print(f"  Осталось валидных       : {len(valid_track_ids)}")

    # Убираем из детекций все записи с невалидными треками
    filtered_detections: dict[int, list] = {}
    for fidx, dets in frame_detections.items():
        valid_dets = [d for d in dets if d[0] in valid_track_ids]
        if valid_dets:
            filtered_detections[fidx] = valid_dets

    frame_detections = filtered_detections
    print(f"  Кадров после фильтрации : {len(frame_detections)}")

    if not frame_detections:
        print("\n  После фильтрации детекций не осталось!")
        sys.exit(0)

    # ================================================================
    #  ПОСТРОЕНИЕ ВРЕМЕННЫХ СЕГМЕНТОВ
    # ================================================================
    print(f"\n{'═'*55}")
    print("  ПОСТРОЕНИЕ СЕГМЕНТОВ")
    print(f"{'═'*55}")

    # Получаем отсортированный список кадров с валидными детекциями
    frames_with_people = sorted(frame_detections.keys())

    segments = build_segments(
        frames_with_people,
        total_frames,
        fps,
        extend_sec=args.extend,
        min_duration_sec=args.min_segment,
        gap_merge_sec=args.gap_merge,
    )

    if not segments:
        print("  Не найдено сегментов достаточной длительности!")
        sys.exit(0)

    # Подсчёт итоговой длительности
    total_out_frames = sum(e - s + 1 for s, e in segments)
    total_out_sec = total_out_frames / fps
    compression = total_out_sec / duration_sec * 100 if duration_sec else 0

    print(f"\n  Найдено сегментов: {len(segments)}")
    print(f"  {'─'*50}")
    for i, (s, e) in enumerate(segments):
        dur = (e - s + 1) / fps
        print(
            f"  {i + 1:3d}. "
            f"{format_time(s / fps):>10s} — {format_time(e / fps):>10s}  "
            f"[{dur:6.1f} сек, {e - s + 1:6d} кадров]"
        )
    print(f"  {'─'*50}")
    print(f"  Итого на выходе : {format_time(total_out_sec)} ({total_out_sec:.1f} сек)")
    print(f"  Сжатие          : {compression:.1f}% от оригинала")

    # ================================================================
    #  ПРЕДВАРИТЕЛЬНЫЙ РАСЧЁТ ТРАЕКТОРИЙ (для визуализации следов)
    # ================================================================
    # Строим словарь: track_id → {frame_idx: (cx, cy)}
    # Это позволит быстро рисовать «хвосты» трекинга во втором проходе
    track_centers: dict[int, dict[int, tuple]] = defaultdict(dict)
    for fidx, dets in frame_detections.items():
        for det in dets:
            tid = det[0]
            cx = int((det[1] + det[3]) / 2)
            cy = int((det[2] + det[4]) / 2)
            track_centers[tid][fidx] = (cx, cy)

    # ================================================================
    #  ВТОРОЙ ПРОХОД: запись выходного видео с аннотациями
    # ================================================================
    print(f"\n{'═'*55}")
    print("  ВТОРОЙ ПРОХОД: запись выходного видео")
    print(f"{'═'*55}")

    cap = cv2.VideoCapture(input_path)

    # Кодек mp4v (MPEG-4) — работает на всех платформах без доп. библиотек
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"  Ошибка: не удалось создать файл '{output_path}'!")
        cap.release()
        sys.exit(1)

    pbar = tqdm(
        total=total_out_frames,
        desc="  Запись",
        unit=" кадр",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}]",
    )

    written = 0

    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        # Перемещаемся к первому кадру сегмента
        # (намного быстрее, чем читать всё видео последовательно)
        cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)

        for fidx in range(seg_start, seg_end + 1):
            ret, frame = cap.read()
            if not ret:
                # Если видео закончилось раньше ожидаемого
                break

            # Получаем детекции для текущего кадра (может быть пустой список)
            dets = frame_detections.get(fidx, [])

            # Рисуем аннотации: боксы, ID, траектории, информационный оверлей
            frame = draw_annotations(
                frame=frame,
                detections=dets,
                frame_idx=fidx,
                fps=fps,
                seg_idx=seg_idx,
                total_segments=len(segments),
                track_centers=track_centers,
                trail_length=args.trail,
            )

            out.write(frame)
            written += 1
            pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    # ── Итоговая статистика ──
    out_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n{'═'*55}")
    print("  ГОТОВО!")
    print(f"{'═'*55}")
    print(f"  Записано кадров  : {written}")
    print(f"  Длительность     : {format_time(written / fps)}")
    print(f"  Выходной файл    : {output_path}")
    print(f"  Размер файла     : {out_size_mb:.1f} МБ")
    print()


if __name__ == "__main__":
    main()
