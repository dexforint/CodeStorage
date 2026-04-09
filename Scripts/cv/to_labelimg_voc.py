import argparse
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def prettify_xml(elem: ET.Element) -> str:
    rough = ET.tostring(elem, encoding="utf-8")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def yolo_norm_to_xyxy(xc, yc, bw, bh, W, H):
    xmin = (xc - bw / 2.0) * W
    ymin = (yc - bh / 2.0) * H
    xmax = (xc + bw / 2.0) * W
    ymax = (yc + bh / 2.0) * H
    return xmin, ymin, xmax, ymax


def make_voc_xml(
    folder_name: str,
    filename: str,
    path_str: str,
    W: int,
    H: int,
    depth: int,
    class_name: str,
    boxes_xyxy: list[tuple[int, int, int, int]],
) -> str:
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = folder_name
    ET.SubElement(annotation, "filename").text = filename
    ET.SubElement(annotation, "path").text = path_str

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(W)
    ET.SubElement(size, "height").text = str(H)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    for xmin, ymin, xmax, ymax in boxes_xyxy:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = class_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(xmin)
        ET.SubElement(bnd, "ymin").text = str(ymin)
        ET.SubElement(bnd, "xmax").text = str(xmax)
        ET.SubElement(bnd, "ymax").text = str(ymax)

    return prettify_xml(annotation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_ann_dir", required=True, help="Папка с txt разметкой (conf xn yn wn hn)"
    )
    ap.add_argument(
        "--dst_xml_dir",
        required=True,
        help="Куда сохранять VOC XML (LabelImg). Если нет — будет создана",
    )

    ap.add_argument(
        "--img_width", type=int, required=True, help="Ширина изображений (px)"
    )
    ap.add_argument(
        "--img_height", type=int, required=True, help="Высота изображений (px)"
    )
    ap.add_argument(
        "--depth", type=int, default=3, help="Depth для VOC (по умолчанию 3)"
    )

    ap.add_argument(
        "--class_name", default="0", help="Имя класса для VOC (по умолчанию '0')"
    )

    ap.add_argument(
        "--folder_name", default="target", help="Значение тега <folder> в XML"
    )
    ap.add_argument(
        "--img_ext",
        default=".png",
        help="Расширение изображения для тега <filename> (по умолчанию .png)",
    )
    ap.add_argument(
        "--path_prefix",
        default="",
        help="Префикс для тега <path>. Например: C:/data/images. "
        "Итоговый <path> будет: path_prefix/<stem><img_ext>. "
        "Если пусто — пишется только filename.",
    )

    ap.add_argument(
        "--normalized",
        action="store_true",
        default=True,
        help="Считать xn/yn/wn/hn нормализованными (0..1). По умолчанию: True",
    )
    ap.add_argument(
        "--skip_empty", action="store_true", help="Если txt пустой — не создавать xml"
    )
    args = ap.parse_args()

    src_ann_dir = Path(args.src_ann_dir)
    dst_xml_dir = Path(args.dst_xml_dir)
    dst_xml_dir.mkdir(parents=True, exist_ok=True)

    W, H = args.img_width, args.img_height
    if W <= 0 or H <= 0:
        raise SystemExit("img_width/img_height должны быть > 0")

    txt_files = sorted(src_ann_dir.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"В {src_ann_dir} не найдено *.txt")

    img_ext = args.img_ext if args.img_ext.startswith(".") else ("." + args.img_ext)
    path_prefix = args.path_prefix.strip().rstrip("/\\")
    for txt_path in tqdm(txt_files):
        stem = txt_path.stem

        boxes = []
        lines = txt_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(
                    f"[WARN] {txt_path.name}: строка не из 5 чисел: '{line}' -> пропуск"
                )
                continue

            # формат: conf xn yn wn hn
            conf, xc, yc, bw, bh = map(float, parts)

            if args.normalized:
                xmin, ymin, xmax, ymax = yolo_norm_to_xyxy(xc, yc, bw, bh, W, H)
            else:
                # если вдруг координаты уже в пикселях (центр+размер)
                xmin = xc - bw / 2.0
                ymin = yc - bh / 2.0
                xmax = xc + bw / 2.0
                ymax = yc + bh / 2.0

            xmin = int(round(clamp(xmin, 0, W - 1)))
            ymin = int(round(clamp(ymin, 0, H - 1)))
            xmax = int(round(clamp(xmax, 0, W - 1)))
            ymax = int(round(clamp(ymax, 0, H - 1)))

            if xmax < xmin:
                xmin, xmax = xmax, xmin
            if ymax < ymin:
                ymin, ymax = ymax, ymin

            if xmax == xmin or ymax == ymin:
                continue

            boxes.append((xmin, ymin, xmax, ymax))

        if args.skip_empty and not boxes:
            print(f"[SKIP] {txt_path.name}: пусто -> xml не создаю (skip_empty)")
            continue

        filename = f"{stem}{img_ext}"
        if path_prefix:
            # делаем "псевдо-абсолютный" путь, как строку в XML
            path_str = str(Path(path_prefix) / filename)
        else:
            path_str = filename

        xml_str = make_voc_xml(
            folder_name=args.folder_name,
            filename=filename,
            path_str=path_str,
            W=W,
            H=H,
            depth=args.depth,
            class_name=args.class_name,
            boxes_xyxy=boxes,
        )

        out_xml = dst_xml_dir / f"{stem}.xml"
        out_xml.write_text(xml_str, encoding="utf-8")
        # print(f"[OK] {txt_path.name} -> {out_xml.name} (boxes={len(boxes)})")


if __name__ == "__main__":
    main()

# python ./cv/to_labelimg_voc.py --src_ann_dir data/ensemble --dst_xml_dir data/labelimg --img_width 476 --img_height 360 --folder_name target --img_ext .png --path_prefix "C:/Users/user/Documents/Projects/CodeStorage/Scripts/data/target"
