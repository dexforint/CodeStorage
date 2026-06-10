#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import trange


def get_props(cap: cv2.VideoCapture):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return w, h, fps, cnt


def hstack_videos(inputs, output):
    caps = []
    try:
        # Open all inputs
        for p in inputs:
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise RuntimeError(f"Не удалось открыть видео: {p}")
            caps.append(cap)

        # Validate одинаковые параметры
        w0, h0, fps0, cnt0 = get_props(caps[0])
        for i, cap in enumerate(caps[1:], start=2):
            w, h, fps, cnt = get_props(cap)
            if (w, h) != (w0, h0):
                raise RuntimeError(
                    f"Размеры не совпадают (видео #{i}): {w}x{h} != {w0}x{h0}"
                )
            # FPS иногда хранится с погрешностью — сравним с допуском
            if abs(fps - fps0) > 1e-3:
                raise RuntimeError(f"FPS не совпадает (видео #{i}): {fps} != {fps0}")
            if cnt != cnt0:
                raise RuntimeError(
                    f"Длительность/число кадров не совпадает (видео #{i}): {cnt} != {cnt0}"
                )

        out_w = w0 * len(caps)
        out_h = h0

        # Выбор кодека по расширению (просто и обычно работает)
        ext = Path(output).suffix.lower()
        if ext in (".mp4", ".m4v", ".mov"):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        elif ext in (".avi",):
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        else:
            # fallback
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(str(output), fourcc, fps0, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError(f"Не удалось создать выходной файл: {output}")

        # Покадровая склейка
        for frame_idx in trange(cnt0):
            frames = []
            for cap in caps:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"Ошибка чтения кадра {frame_idx} из одного из видео"
                    )
                frames.append(frame)

            stitched = np.hstack(frames)
            stitched = np.ascontiguousarray(stitched)  # на всякий случай для writer
            writer.write(stitched)

        writer.release()

    finally:
        for cap in caps:
            try:
                cap.release()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Горизонтальная покадровая склейка видео одинаковых параметров."
    )
    parser.add_argument("inputs", nargs="+", help="Пути к входным видео (2 и более).")
    parser.add_argument("-o", "--output", required=True, help="Путь к выходному видео.")
    args = parser.parse_args()

    if len(args.inputs) < 2:
        print("Нужно минимум 2 входных видео.", file=sys.stderr)
        sys.exit(2)

    hstack_videos([Path(p) for p in args.inputs], Path(args.output))


if __name__ == "__main__":
    # python hstack_videos.py cam1.mp4 cam2.mp4 cam3.mp4 -o out.mp4
    main()
