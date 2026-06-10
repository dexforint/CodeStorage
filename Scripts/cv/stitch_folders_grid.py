import os
import re
import argparse
from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(s: str):
    # "img2.png" < "img10.png"
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_subfolders(root_dir: str):
    subs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    subs.sort(key=natural_key)
    return [os.path.join(root_dir, d) for d in subs]


def list_images(folder: str):
    files = []
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMG_EXTS:
            files.append(f)
    files.sort(key=natural_key)
    return [os.path.join(folder, f) for f in files]


def load_font(font_size: int):
    # На Win11 обычно есть Arial
    try:
        return ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser(
        description="Склейка изображений из подпапок в одну таблицу."
    )
    ap.add_argument(
        "root", help="Папка, внутри которой лежат подпапки со изображениями"
    )
    ap.add_argument(
        "-o", "--out", default="stitched.png", help="Выходной файл (png рекомендуется)"
    )
    ap.add_argument(
        "--padding", type=int, default=10, help="Отступы между ячейками и по краям (px)"
    )
    ap.add_argument(
        "--header", type=int, default=60, help="Высота шапки с нумерацией столбцов (px)"
    )
    ap.add_argument(
        "--bg", default="255,255,255", help="Фон R,G,B например 255,255,255"
    )
    args = ap.parse_args()

    root_dir = args.root
    padding = args.padding
    header_h = args.header
    bg_rgb = tuple(int(x) for x in args.bg.split(","))

    subfolders = list_subfolders(root_dir)
    if not subfolders:
        raise SystemExit("Не найдено ни одной подпапки в указанной папке.")

    cols = len(subfolders)
    img_paths_by_col = [list_images(sf) for sf in subfolders]

    rows = len(img_paths_by_col[0])
    if rows == 0:
        raise SystemExit(
            f"В подпапке '{os.path.basename(subfolders[0])}' нет изображений."
        )

    # Проверка одинакового количества изображений
    for sf, paths in zip(subfolders, img_paths_by_col):
        if len(paths) != rows:
            raise SystemExit(
                f"Во всех подпапках должно быть одинаковое число изображений.\n"
                f"Папка '{os.path.basename(sf)}' содержит {len(paths)}, ожидалось {rows}."
            )

    # Первый проход: размеры + наличие альфы
    sizes = [[None] * rows for _ in range(cols)]
    any_alpha = False

    for c in range(cols):
        for r in range(rows):
            p = img_paths_by_col[c][r]
            with Image.open(p) as im:
                sizes[c][r] = im.size
                if "A" in im.getbands():
                    any_alpha = True

    # Размеры ячеек: ширина по столбцу, высота по строке
    col_widths = [max(sizes[c][r][0] for r in range(rows)) for c in range(cols)]
    row_heights = [max(sizes[c][r][1] for c in range(cols)) for r in range(rows)]

    total_w = sum(col_widths) + padding * (cols + 1)
    total_h = header_h + sum(row_heights) + padding * (rows + 1)

    mode = "RGBA" if any_alpha else "RGB"
    bg = (*bg_rgb, 0) if mode == "RGBA" else bg_rgb
    canvas = Image.new(mode, (total_w, total_h), bg)

    draw = ImageDraw.Draw(canvas)
    font = load_font(max(12, header_h - 2 * padding))

    # Нумерация столбцов (1..N) по центру каждого столбца в шапке
    x = padding
    for c in range(cols):
        cw = col_widths[c]
        label = str(c + 1)

        # Центрируем текст
        # (anchor работает в современных Pillow; если вдруг нет — можно заменить на textbbox)
        draw.text(
            (x + cw / 2, header_h / 2), label, fill=(0, 0, 0), font=font, anchor="mm"
        )
        x += cw + padding

    # Вставка изображений
    y = header_h + padding
    for r in range(rows):
        x = padding
        rh = row_heights[r]
        for c in range(cols):
            cw = col_widths[c]
            p = img_paths_by_col[c][r]

            with Image.open(p) as im:
                im = im.convert("RGBA" if mode == "RGBA" else "RGB")
                px = x + (cw - im.width) // 2
                py = y + (rh - im.height) // 2

                if mode == "RGBA":
                    canvas.paste(im, (px, py), im)  # учитываем альфу
                else:
                    canvas.paste(im, (px, py))

            x += cw + padding
        y += rh + padding

    canvas.save(args.out)
    print(f"Готово: {args.out}")


if __name__ == "__main__":
    # python stitch_folders_grid.py "C:\path\to\root_folder" -o "C:\path\to\result.png"
    main()
