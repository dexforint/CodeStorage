import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def infer_output_dir(video_path: Path) -> Path:
    # Рядом с input, имя = имя файла без расширения
    return video_path.parent / video_path.stem


def get_frame_count(cap: cv2.VideoCapture) -> int:
    """Пытаемся получить число кадров из метаданных; если не вышло — считаем проходом (grab)."""
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        return frame_count

    # fallback: считаем кадры проходом (без декодирования)
    n = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        n += 1
    return n


def video_to_frames(
    video_path: str, output_dir: str | None = None, ext: str = "png"
) -> int:
    video_path = Path(video_path)
    if output_dir is None:
        output_dir = infer_output_dir(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    frame_count = get_frame_count(cap)

    # Если считали кадры через grab(), курсор в конце — переоткроем видео
    cap.release()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось повторно открыть видео: {video_path}")

    width = max(2, len(str(frame_count)))
    ext = ext.lower().lstrip(".")

    saved = 0
    with tqdm(total=frame_count if frame_count > 0 else None, unit="frame") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            filename = output_dir / f"{saved:0{width}d}.{ext}"

            if ext in ("jpg", "jpeg"):
                cv2.imwrite(str(filename), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            elif ext == "png":
                # PNG без потерь; компрессия влияет только на размер/скорость, не на качество
                cv2.imwrite(str(filename), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            else:
                cv2.imwrite(str(filename), frame)

            saved += 1
            pbar.update(1)

    cap.release()
    return saved


def parse_args():
    p = argparse.ArgumentParser(
        description="Разложить видео на кадры и сохранить как изображения."
    )
    p.add_argument(
        "-i",
        "--input",
        required=True,
        help="Путь к входному видео файлу (например, input.mp4)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Папка для сохранения кадров. Если не задана, будет создана рядом с input (имя = input без расширения).",
    )
    p.add_argument(
        "-e",
        "--ext",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Формат кадров. Для максимального качества рекомендуется png (lossless). (default: png)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    saved = video_to_frames(args.input, args.output, args.ext)
    out_dir = (
        args.output
        if args.output is not None
        else str(infer_output_dir(Path(args.input)))
    )
    print(f"Сохранено кадров: {saved}")
    print(f"Папка: {out_dir}")


if __name__ == "__main__":
    main()
