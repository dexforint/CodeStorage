import json
import math
import os
from dataclasses import dataclass
from html import escape
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

APP_NAME = "YakorDesigner"

# Точное преобразование миллиметров в PDF points.
MM_TO_PT = 72.0 / 25.4

A4_SIZES_MM = {
    "portrait": (210.0, 297.0),
    "landscape": (297.0, 210.0),
}

ORIENTATION_LABELS = {
    "auto": "Авто",
    "portrait": "Портрет",
    "landscape": "Альбомная",
}
LABEL_TO_ORIENTATION = {label: key for key, label in ORIENTATION_LABELS.items()}

# В нижнем поле оставляется место для номера страницы и контрольной линейки.
A4_FOOTER_BAND_MM = 18.0

DEFAULT_PRINT_SETTINGS = {
    "orientation": "auto",
    "overlap": 10,
    "margin": 10,
}


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    minimum: int
    maximum: int
    default: int


PARAMETER_GROUPS = [
    (
        "Габариты и основание",
        [
            ParamSpec("width", "Общая ширина", 300, 1200, 700),
            ParamSpec("height", "Общая высота", 300, 1200, 600),
            ParamSpec("material_thickness", "Толщина фанеры, Z", 6, 60, 24),
            ParamSpec("base_height", "Высота нижней перекладины", 50, 400, 165),
        ],
    ),
    (
        "Стойка и площадка подшипника",
        [
            ParamSpec("pad_diameter", "Диаметр площадки", 60, 300, 160),
            ParamSpec("hole_diameter", "Центральное отверстие", 5, 200, 45),
            ParamSpec("stem_width", "Ширина стойки", 30, 280, 130),
            ParamSpec("stem_offset", "Смещение стойки по X", -250, 250, 0),
        ],
    ),
    (
        "Боковые упоры и скругления",
        [
            ParamSpec("ear_height", "Высота боковых упоров", 0, 220, 65),
            ParamSpec("ear_width", "Ширина боковых упоров", 10, 220, 60),
            ParamSpec("base_corner_radius", "Радиус нижних углов", 0, 100, 18),
            ParamSpec("ear_corner_radius", "Радиус углов упоров", 0, 80, 18),
        ],
    ),
    (
        "Крепёж фланца — необязательно",
        [
            ParamSpec("mount_hole_diameter", "Диаметр 4 отверстий", 3, 30, 8),
            ParamSpec("mount_pcd", "Диаметр окружности PCD", 40, 240, 100),
        ],
    ),
]

ALL_SPECS = [
    spec for _group_name, group_specs in PARAMETER_GROUPS for spec in group_specs
]
SPEC_BY_KEY = {spec.key: spec for spec in ALL_SPECS}


def get_settings_path() -> Path:
    """Путь файла настроек."""
    if os.name == "nt":
        base = Path(
            os.environ.get(
                "APPDATA",
                str(Path.home() / "AppData" / "Roaming"),
            )
        )
    else:
        base = Path(
            os.environ.get(
                "XDG_CONFIG_HOME",
                str(Path.home() / ".config"),
            )
        )

    return base / APP_NAME / "settings.json"


def fmt_number(value: float) -> str:
    """Форматирует число без лишних нулей."""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


class GeometryError(Exception):
    """Параметры не позволяют построить корректный контур."""


class A4LayoutError(Exception):
    """Не удалось подобрать раскладку для печати A4."""


@dataclass
class CutGeometry:
    width: float
    height: float
    thickness: float
    outline: list
    holes: list


def append_unique(points: list, point: tuple) -> None:
    """Добавляет точку, если она не дублирует предыдущую."""
    x, y = point

    if points:
        previous_x, previous_y = points[-1]
        if math.hypot(x - previous_x, y - previous_y) < 1e-8:
            return

    points.append((x, y))


def append_arc(
    points: list,
    cx: float,
    cy: float,
    radius: float,
    start_angle: float,
    end_angle: float,
    max_step_degrees: float = 1.0,
) -> None:
    """Добавляет дугу как последовательность коротких отрезков."""
    if radius <= 1e-12:
        append_unique(points, (cx, cy))
        return

    max_step = math.radians(max_step_degrees)
    step_count = max(
        1,
        math.ceil(abs(end_angle - start_angle) / max_step),
    )

    for index in range(1, step_count + 1):
        angle = start_angle + (end_angle - start_angle) * index / step_count
        point = (
            cx + radius * math.cos(angle),
            cy + radius * math.sin(angle),
        )
        append_unique(points, point)


def build_geometry(params: dict, mount_holes_enabled: bool) -> CutGeometry:
    """
    Строит внешний контур детали и сквозные отверстия.

    Координаты:
      x — от левого края детали;
      y — от верхней точки круглой площадки.
    """
    width = float(params["width"])
    height = float(params["height"])
    thickness = float(params["material_thickness"])
    base_height = float(params["base_height"])
    pad_diameter = float(params["pad_diameter"])
    hole_diameter = float(params["hole_diameter"])
    stem_width = float(params["stem_width"])
    stem_offset = float(params["stem_offset"])
    ear_height = float(params["ear_height"])
    ear_width = float(params["ear_width"])

    if width <= 0 or height <= 0 or thickness <= 0:
        raise GeometryError("Габариты и толщина должны быть больше нуля.")

    radius = pad_diameter / 2.0
    center_x = width / 2.0 + stem_offset
    center_y = radius

    if center_x - radius < -1e-6 or center_x + radius > width + 1e-6:
        raise GeometryError(
            "Круглая площадка выходит за ширину детали. "
            "Уменьшите её диаметр или смещение стойки."
        )

    # Оставляем минимум 3 мм материала вокруг центрального отверстия.
    if hole_diameter + 6.0 > pad_diameter + 1e-9:
        raise GeometryError(
            "Центральное отверстие слишком большое: "
            "оставьте вокруг него не менее 3 мм материала."
        )

    half_stem = stem_width / 2.0

    if stem_width > pad_diameter:
        raise GeometryError(
            "Ширина стойки должна быть заметно меньше диаметра площадки."
        )

    neck_drop = math.sqrt(max(0.0, radius * radius - half_stem * half_stem))
    join_y = center_y + neck_drop

    base_top = height - base_height
    ear_top = base_top - ear_height

    if ear_top < 0:
        raise GeometryError(
            "Высота перекладины и боковых упоров превышает общую высоту."
        )

    if base_top <= join_y + 2.0:
        raise GeometryError(
            "Недостаточно места для стойки между площадкой и основанием. "
            "Уменьшите высоту перекладины или увеличьте общую высоту."
        )

    if 2.0 * ear_width >= width:
        raise GeometryError("Боковые упоры слишком широкие для выбранной общей ширины.")

    stem_left = center_x - half_stem
    stem_right = center_x + half_stem
    edge_clearance = 2.0

    if (
        stem_left <= ear_width + edge_clearance
        or stem_right >= width - ear_width - edge_clearance
    ):
        raise GeometryError(
            "Стойка слишком близко к боковым упорам. "
            "Уменьшите ширину стойки/упоров или измените смещение."
        )

    left_ear_overlaps_pad = ear_width > center_x - radius + 1e-6
    right_ear_overlaps_pad = width - ear_width < center_x + radius - 1e-6

    if ear_top < join_y and (left_ear_overlaps_pad or right_ear_overlaps_pad):
        raise GeometryError(
            "Боковой упор заходит в область круглой площадки. "
            "Уменьшите высоту упоров или их ширину."
        )

    base_radius = min(
        float(params["base_corner_radius"]),
        base_height / 2.0,
        width / 2.0,
    )
    ear_radius = min(
        float(params["ear_corner_radius"]),
        ear_width / 2.0,
        ear_height,
    )

    points = []

    def add(point):
        append_unique(points, point)

    # Внешний периметр — старт снизу слева.
    add((base_radius, height))
    add((width - base_radius, height))

    # Нижний правый угол.
    append_arc(
        points,
        width - base_radius,
        height - base_radius,
        base_radius,
        math.pi / 2.0,
        0.0,
    )

    # Правая сторона и верхний правый угол упора.
    add((width, ear_top + ear_radius))
    append_arc(
        points,
        width - ear_radius,
        ear_top + ear_radius,
        ear_radius,
        0.0,
        -math.pi / 2.0,
    )

    # Верх упора и его внутренняя сторона.
    add((width - ear_width + ear_radius, ear_top))
    append_arc(
        points,
        width - ear_width + ear_radius,
        ear_top + ear_radius,
        ear_radius,
        -math.pi / 2.0,
        -math.pi,
    )

    # Верх перекладины и правая сторона стойки.
    add((width - ear_width, base_top))
    add((stem_right, base_top))
    add((stem_right, join_y))

    # Дуга круглой площадки.
    alpha = math.atan2(neck_drop, half_stem)
    append_arc(
        points,
        center_x,
        center_y,
        radius,
        alpha,
        -math.pi - alpha,
    )

    # Левая сторона стойки и перекладины.
    add((stem_left, base_top))
    add((ear_width, base_top))
    add((ear_width, ear_top + ear_radius))

    # Внутреннее закругление левого упора.
    append_arc(
        points,
        ear_width - ear_radius,
        ear_top + ear_radius,
        ear_radius,
        0.0,
        -math.pi / 2.0,
    )

    # Верх левого упора и его внешний закруглённый угол.
    add((ear_radius, ear_top))
    append_arc(
        points,
        ear_radius,
        ear_top + ear_radius,
        ear_radius,
        -math.pi / 2.0,
        -math.pi,
    )

    # Левая сторона и нижний левый угол.
    add((0.0, height - base_radius))
    append_arc(
        points,
        base_radius,
        height - base_radius,
        base_radius,
        math.pi,
        math.pi / 2.0,
    )

    if len(points) < 3:
        raise GeometryError("Не удалось построить внешний контур.")

    holes = [(center_x, center_y, hole_diameter / 2.0)]

    if mount_holes_enabled:
        mount_diameter = float(params["mount_hole_diameter"])
        pcd = float(params["mount_pcd"])
        mount_radius = mount_diameter / 2.0
        pitch_radius = pcd / 2.0

        if pitch_radius + mount_radius + 3.0 > radius:
            raise GeometryError(
                "Крепёжные отверстия слишком близко к краю площадки. "
                "Уменьшите PCD/диаметр отверстий или увеличьте площадку."
            )

        if pitch_radius - mount_radius < hole_diameter / 2.0 + 3.0:
            raise GeometryError(
                "Крепёжные отверстия пересекаются с центральным. "
                "Увеличьте PCD или уменьшите диаметры отверстий."
            )

        for angle_degrees in (0, 90, 180, 270):
            angle = math.radians(angle_degrees)
            x = center_x + pitch_radius * math.cos(angle)
            y = center_y + pitch_radius * math.sin(angle)
            holes.append((x, y, mount_radius))

    return CutGeometry(
        width=width,
        height=height,
        thickness=thickness,
        outline=points,
        holes=holes,
    )


def write_svg(path: Path, geometry: CutGeometry, params: dict) -> None:
    """Экспортирует контур в SVG, единицы — миллиметры."""
    path.parent.mkdir(parents=True, exist_ok=True)

    path_data = (
        "M "
        + " L ".join(f"{fmt_number(x)},{fmt_number(y)}" for x, y in geometry.outline)
        + " Z"
    )

    description = (
        f"Якорь OCR. Размер контура: "
        f"{fmt_number(geometry.width)} × {fmt_number(geometry.height)} мм. "
        f"Толщина фанеры: {fmt_number(geometry.thickness)} мм. "
        f"Центральное отверстие: "
        f"Ø{fmt_number(params['hole_diameter'])} мм."
    )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{fmt_number(geometry.width)}mm" '
            f'height="{fmt_number(geometry.height)}mm" '
            f'viewBox="0 0 {fmt_number(geometry.width)} '
            f'{fmt_number(geometry.height)}">'
        ),
        "  <title>Якорь OCR — контур раскроя</title>",
        f"  <desc>{escape(description)}</desc>",
        (
            '  <g id="CUT" fill="none" stroke="#000000" '
            'stroke-width="0.2" stroke-linejoin="round" '
            'stroke-linecap="round">'
        ),
        f'    <path id="outline" d="{path_data}"/>',
    ]

    for index, (x, y, radius) in enumerate(geometry.holes, start=1):
        lines.append(
            f'    <circle id="hole-{index}" '
            f'cx="{fmt_number(x)}" cy="{fmt_number(y)}" '
            f'r="{fmt_number(radius)}"/>'
        )

    lines.extend(["  </g>", "</svg>"])

    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines) + "\n")


def write_dxf(path: Path, geometry: CutGeometry) -> None:
    """Экспортирует контур в DXF R2000, единицы — миллиметры."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "0",
        "SECTION",
        "2",
        "HEADER",
        "9",
        "$ACADVER",
        "1",
        "AC1015",
        "9",
        "$INSUNITS",
        "70",
        "4",
        "9",
        "$MEASUREMENT",
        "70",
        "1",
        "999",
        f"PLYWOOD_THICKNESS_MM={fmt_number(geometry.thickness)}",
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "TABLES",
        "0",
        "TABLE",
        "2",
        "LTYPE",
        "70",
        "1",
        "0",
        "LTYPE",
        "100",
        "AcDbSymbolTableRecord",
        "100",
        "AcDbLinetypeTableRecord",
        "2",
        "CONTINUOUS",
        "70",
        "0",
        "3",
        "Solid line",
        "72",
        "65",
        "73",
        "0",
        "40",
        "0.0",
        "0",
        "ENDTAB",
        "0",
        "TABLE",
        "2",
        "LAYER",
        "70",
        "2",
        "0",
        "LAYER",
        "100",
        "AcDbSymbolTableRecord",
        "100",
        "AcDbLayerTableRecord",
        "2",
        "0",
        "70",
        "0",
        "62",
        "7",
        "6",
        "CONTINUOUS",
        "0",
        "LAYER",
        "100",
        "AcDbSymbolTableRecord",
        "100",
        "AcDbLayerTableRecord",
        "2",
        "CUT",
        "70",
        "0",
        "62",
        "7",
        "6",
        "CONTINUOUS",
        "0",
        "ENDTAB",
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "ENTITIES",
        "0",
        "LWPOLYLINE",
        "100",
        "AcDbEntity",
        "8",
        "CUT",
        "100",
        "AcDbPolyline",
        "90",
        str(len(geometry.outline)),
        "70",
        "1",
    ]

    # В DXF ось Y направлена вверх, поэтому переворачиваем её.
    for x, y in geometry.outline:
        dxf_y = geometry.height - y
        lines.extend(["10", fmt_number(x), "20", fmt_number(dxf_y)])

    for x, y, radius in geometry.holes:
        dxf_y = geometry.height - y
        lines.extend(
            [
                "0",
                "CIRCLE",
                "100",
                "AcDbEntity",
                "8",
                "CUT",
                "100",
                "AcDbCircle",
                "10",
                fmt_number(x),
                "20",
                fmt_number(dxf_y),
                "30",
                "0",
                "40",
                fmt_number(radius),
            ]
        )

    lines.extend(["0", "ENDSEC", "0", "EOF"])

    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("\n".join(lines) + "\n")


def _tile_count(length: float, usable_length: float, step: float) -> int:
    """Количество листов, необходимое для покрытия размера детали."""
    if length <= usable_length:
        return 1

    return max(
        1,
        int(math.ceil((length - usable_length) / step - 1e-12)) + 1,
    )


def calculate_a4_layout(
    width: float,
    height: float,
    orientation: str,
    overlap: float,
    margin: float,
) -> dict:
    """Подбирает раскладку детали по страницам A4."""
    if orientation not in ("auto", "portrait", "landscape"):
        raise A4LayoutError("Неизвестная ориентация A4.")

    if not 10 <= margin <= 25:
        raise A4LayoutError("Поле листа должно быть от 10 до 25 мм.")

    if not 5 <= overlap <= 30:
        raise A4LayoutError("Перекрытие должно быть от 5 до 30 мм.")

    def make_candidate(candidate_orientation: str) -> dict:
        page_width, page_height = A4_SIZES_MM[candidate_orientation]

        left_margin = margin
        right_margin = margin
        top_margin = margin
        bottom_margin = max(margin, A4_FOOTER_BAND_MM)

        usable_width = page_width - left_margin - right_margin
        usable_height = page_height - top_margin - bottom_margin

        if usable_width <= overlap or usable_height <= overlap:
            raise A4LayoutError("Слишком большие поля или перекрытие для листа A4.")

        step_x = usable_width - overlap
        step_y = usable_height - overlap

        columns = _tile_count(width, usable_width, step_x)
        rows = _tile_count(height, usable_height, step_y)

        return {
            "orientation": candidate_orientation,
            "page_width": page_width,
            "page_height": page_height,
            "left_margin": left_margin,
            "right_margin": right_margin,
            "top_margin": top_margin,
            "bottom_margin": bottom_margin,
            "usable_width": usable_width,
            "usable_height": usable_height,
            "overlap": overlap,
            "step_x": step_x,
            "step_y": step_y,
            "columns": columns,
            "rows": rows,
            "total_pages": columns * rows,
        }

    if orientation != "auto":
        return make_candidate(orientation)

    candidates = [
        make_candidate("portrait"),
        make_candidate("landscape"),
    ]

    model_aspect = width / height

    def score(layout: dict):
        page_aspect = layout["usable_width"] / layout["usable_height"]
        aspect_difference = abs(math.log(page_aspect / model_aspect))

        # Сначала минимизируем количество листов, затем подбираем
        # ориентацию, лучше соответствующую пропорциям детали.
        return (
            layout["total_pages"],
            aspect_difference,
            0 if layout["orientation"] == "landscape" else 1,
        )

    return min(candidates, key=score)


def _pdf_circle_path(
    commands: list,
    cx: float,
    cy: float,
    radius: float,
    close_and_stroke: bool = True,
) -> None:
    """Добавляет окружность в PDF как четыре кривые Безье."""
    k = 0.5522847498307936 * radius

    commands.append(f"{fmt_number(cx + radius)} {fmt_number(cy)} m")
    commands.append(
        f"{fmt_number(cx + radius)} {fmt_number(cy + k)} "
        f"{fmt_number(cx + k)} {fmt_number(cy + radius)} "
        f"{fmt_number(cx)} {fmt_number(cy + radius)} c"
    )
    commands.append(
        f"{fmt_number(cx - k)} {fmt_number(cy + radius)} "
        f"{fmt_number(cx - radius)} {fmt_number(cy + k)} "
        f"{fmt_number(cx - radius)} {fmt_number(cy)} c"
    )
    commands.append(
        f"{fmt_number(cx - radius)} {fmt_number(cy - k)} "
        f"{fmt_number(cx - k)} {fmt_number(cy - radius)} "
        f"{fmt_number(cx)} {fmt_number(cy - radius)} c"
    )
    commands.append(
        f"{fmt_number(cx + k)} {fmt_number(cy - radius)} "
        f"{fmt_number(cx + radius)} {fmt_number(cy - k)} "
        f"{fmt_number(cx + radius)} {fmt_number(cy)} c"
    )

    if close_and_stroke:
        commands.append("h S")


def _pdf_escape_text(text: str) -> str:
    """Экранирует строку для PDF."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_add_text(
    commands: list,
    text: str,
    x_mm: float,
    y_mm: float,
    font_size: float = 8.0,
) -> None:
    """Добавляет простой текст в координатах страницы, заданных в мм."""
    x_pt = x_mm * MM_TO_PT
    y_pt = y_mm * MM_TO_PT
    escaped = _pdf_escape_text(text)

    commands.append(
        f"BT /F1 {fmt_number(font_size)} Tf "
        f"1 0 0 1 {fmt_number(x_pt)} {fmt_number(y_pt)} Tm "
        f"({escaped}) Tj ET"
    )


def _make_pdf_page_content(
    geometry: CutGeometry,
    layout: dict,
    tile_column: int,
    tile_row: int,
    page_number: int,
) -> bytes:
    """Создаёт PDF-команды для одной страницы A4."""
    page_width = layout["page_width"]
    page_height = layout["page_height"]
    margin = layout["left_margin"]
    bottom_margin = layout["bottom_margin"]
    usable_width = layout["usable_width"]
    usable_height = layout["usable_height"]
    overlap = layout["overlap"]

    tile_x = tile_column * layout["step_x"]
    tile_y = tile_row * layout["step_y"]

    commands = [
        "q",
        (
            f"{fmt_number(margin * MM_TO_PT)} "
            f"{fmt_number(bottom_margin * MM_TO_PT)} "
            f"{fmt_number(usable_width * MM_TO_PT)} "
            f"{fmt_number(usable_height * MM_TO_PT)} re W n"
        ),
    ]

    # Переводим координаты модели в миллиметры страницы.
    # Отражение Y компенсирует направление оси в PDF.
    translate_x = (margin - tile_x) * MM_TO_PT
    translate_y = (page_height - margin + tile_y) * MM_TO_PT

    commands.extend(
        [
            (
                f"{fmt_number(MM_TO_PT)} 0 0 "
                f"{fmt_number(-MM_TO_PT)} "
                f"{fmt_number(translate_x)} "
                f"{fmt_number(translate_y)} cm"
            ),
            "0 0 0 RG",
            "0.2 w",
            "1 J",
            "1 j",
        ]
    )

    # Внешний контур детали.
    first_x, first_y = geometry.outline[0]
    commands.append(f"{fmt_number(first_x)} {fmt_number(first_y)} m")

    for x, y in geometry.outline[1:]:
        commands.append(f"{fmt_number(x)} {fmt_number(y)} l")

    commands.append("h S")

    # Сквозные отверстия.
    for x, y, radius in geometry.holes:
        _pdf_circle_path(commands, x, y, radius)

    commands.append("Q")

    # Метки для совмещения листов в зоне перекрытия.
    registration_marks = []

    if overlap > 0:
        if tile_column > 0:
            mark_x = tile_x + overlap / 2.0
            for fraction in (0.25, 0.75):
                registration_marks.append((mark_x, tile_y + usable_height * fraction))

        if tile_column < layout["columns"] - 1:
            mark_x = tile_x + usable_width - overlap / 2.0
            for fraction in (0.25, 0.75):
                registration_marks.append((mark_x, tile_y + usable_height * fraction))

        if tile_row > 0:
            mark_y = tile_y + overlap / 2.0
            for fraction in (0.25, 0.75):
                registration_marks.append((tile_x + usable_width * fraction, mark_y))

        if tile_row < layout["rows"] - 1:
            mark_y = tile_y + usable_height - overlap / 2.0
            for fraction in (0.25, 0.75):
                registration_marks.append((tile_x + usable_width * fraction, mark_y))

    if registration_marks:
        commands.extend(
            [
                "q",
                "0.45 0.45 0.45 RG",
                "0.15 w",
                "1 J",
                "1 j",
            ]
        )

        half_size = 1.5
        for global_x, global_y in registration_marks:
            page_x = (margin + global_x - tile_x) * MM_TO_PT
            page_y = (page_height - margin - global_y + tile_y) * MM_TO_PT
            half_pt = half_size * MM_TO_PT

            commands.append(
                f"{fmt_number(page_x - half_pt)} {fmt_number(page_y)} m "
                f"{fmt_number(page_x + half_pt)} {fmt_number(page_y)} l"
            )
            commands.append(
                f"{fmt_number(page_x)} {fmt_number(page_y - half_pt)} m "
                f"{fmt_number(page_x)} {fmt_number(page_y + half_pt)} l S"
            )

        commands.append("Q")

    # Заголовок страницы находится в верхнем поле.
    commands.append("0 0 0 rg")
    _pdf_add_text(
        commands,
        (
            f"MODEL {fmt_number(geometry.width)} x "
            f"{fmt_number(geometry.height)} mm | "
            "A4 TEMPLATE | PRINT AT 100%"
        ),
        margin,
        page_height - 8.0,
        8.0,
    )

    # Номер листа и его положение в сетке.
    footer = (
        f"PAGE {page_number}/{layout['total_pages']} | "
        f"ROW {tile_row + 1}/{layout['rows']} | "
        f"COL {tile_column + 1}/{layout['columns']} | "
        f"OVERLAP {fmt_number(overlap)} mm"
    )
    _pdf_add_text(commands, footer, margin, 6.0, 7.0)

    # На первой странице — контрольная линия длиной ровно 100 мм.
    if page_number == 1:
        bar_start = page_width - 110.0
        bar_end = page_width - 10.0
        bar_y = 7.5
        tick_half_height = 1.5

        commands.extend(
            [
                "0 0 0 RG",
                "0.3 w",
                (
                    f"{fmt_number(bar_start * MM_TO_PT)} "
                    f"{fmt_number(bar_y * MM_TO_PT)} m "
                    f"{fmt_number(bar_end * MM_TO_PT)} "
                    f"{fmt_number(bar_y * MM_TO_PT)} l"
                ),
                (
                    f"{fmt_number(bar_start * MM_TO_PT)} "
                    f"{fmt_number((bar_y - tick_half_height) * MM_TO_PT)} m "
                    f"{fmt_number(bar_start * MM_TO_PT)} "
                    f"{fmt_number((bar_y + tick_half_height) * MM_TO_PT)} l"
                ),
                (
                    f"{fmt_number(bar_end * MM_TO_PT)} "
                    f"{fmt_number((bar_y - tick_half_height) * MM_TO_PT)} m "
                    f"{fmt_number(bar_end * MM_TO_PT)} "
                    f"{fmt_number((bar_y + tick_half_height) * MM_TO_PT)} l S"
                ),
            ]
        )

        _pdf_add_text(
            commands,
            "CHECK: THIS LINE MUST MEASURE 100 mm",
            bar_start,
            12.0,
            6.5,
        )

    return ("\n".join(commands) + "\n").encode("ascii")


def write_tiled_pdf(
    path: Path,
    geometry: CutGeometry,
    orientation: str,
    overlap: int,
    margin: int,
) -> dict:
    """Записывает многостраничный A4 PDF без сторонних библиотек."""
    path.parent.mkdir(parents=True, exist_ok=True)

    layout = calculate_a4_layout(
        geometry.width,
        geometry.height,
        orientation,
        overlap,
        margin,
    )

    page_ids = [4 + 2 * index for index in range(layout["total_pages"])]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)

    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            f"<< /Type /Pages /Kids [{kids}] " f"/Count {layout['total_pages']} >>"
        ).encode("ascii"),
        3: (
            b"<< /Type /Font /Subtype /Type1 "
            b"/BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
        ),
    }

    page_number = 0

    for row in range(layout["rows"]):
        for column in range(layout["columns"]):
            page_number += 1
            page_id = 4 + 2 * (page_number - 1)
            stream_id = page_id + 1

            content = _make_pdf_page_content(
                geometry,
                layout,
                column,
                row,
                page_number,
            )

            page_width_pt = layout["page_width"] * MM_TO_PT
            page_height_pt = layout["page_height"] * MM_TO_PT

            objects[page_id] = (
                f"<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {fmt_number(page_width_pt)} "
                f"{fmt_number(page_height_pt)}] "
                f"/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {stream_id} 0 R >>"
            ).encode("ascii")

            objects[stream_id] = (
                f"<< /Length {len(content)} >>\nstream\n".encode("ascii")
                + content
                + b"\nendstream"
            )

    max_object_id = max(objects)
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0] * (max_object_id + 1)

    for object_id in range(1, max_object_id + 1):
        offsets[object_id] = len(pdf)
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(objects[object_id])
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {max_object_id + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")

    for object_id in range(1, max_object_id + 1):
        pdf.extend(f"{offsets[object_id]:010d} 00000 n \n".encode("ascii"))

    pdf.extend(
        (
            f"trailer\n<< /Size {max_object_id + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )

    path.write_bytes(pdf)
    return layout


class ScrollableFrame(ttk.Frame):
    """Прокручиваемая область для панели параметров."""

    def __init__(self, parent):
        super().__init__(parent)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self,
            highlightthickness=0,
            background="#f5f6f8",
        )
        self.scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview,
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window(
            (0, 0),
            window=self.inner,
            anchor="nw",
        )

        self.inner.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_inner)

        self.winfo_toplevel().bind_all(
            "<MouseWheel>",
            self._on_mousewheel,
        )

    def _update_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_inner(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event):
        widget = self.winfo_containing(event.x_root, event.y_root)
        if widget is None:
            return

        widget_path = str(widget)
        inner_path = str(self.inner)

        if widget_path.startswith(inner_path):
            units = int(-event.delta / 120)
            if units == 0:
                units = -1 if event.delta > 0 else 1

            self.canvas.yview_scroll(units, "units")
            return "break"


class YakorDesignerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Якорь OCR — редактор контура")
        self.root.geometry("1320x850")
        self.root.minsize(1080, 720)

        self.settings_path = get_settings_path()
        self._suspend_callbacks = True
        self._save_after_id = None
        self._closed = False

        self.parameter_vars = {
            spec.key: tk.IntVar(master=root, value=spec.default) for spec in ALL_SPECS
        }

        self.mount_holes_var = tk.BooleanVar(master=root, value=False)

        self.print_orientation_var = tk.StringVar(
            master=root,
            value=ORIENTATION_LABELS["auto"],
        )
        self.print_overlap_var = tk.IntVar(
            master=root,
            value=DEFAULT_PRINT_SETTINGS["overlap"],
        )
        self.print_margin_var = tk.IntVar(
            master=root,
            value=DEFAULT_PRINT_SETTINGS["margin"],
        )

        self.last_print_options = DEFAULT_PRINT_SETTINGS.copy()
        self.current_geometry = None
        self.last_geometry = None
        self.current_params = None
        self.last_params = {spec.key: spec.default for spec in ALL_SPECS}

        self.status_var = tk.StringVar(
            master=root,
            value="Подготовка контура…",
        )

        self.mount_controls = []
        self.spinboxes = {}
        self.print_spinboxes = {}

        self._load_settings()
        self._setup_style()
        self._build_ui()

        for variable in self.parameter_vars.values():
            variable.trace_add("write", self._on_parameters_changed)

        self.mount_holes_var.trace_add(
            "write",
            self._on_parameters_changed,
        )

        self.print_orientation_var.trace_add(
            "write",
            self._on_print_settings_changed,
        )
        self.print_overlap_var.trace_add(
            "write",
            self._on_print_settings_changed,
        )
        self.print_margin_var.trace_add(
            "write",
            self._on_print_settings_changed,
        )

        self._suspend_callbacks = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._on_parameters_changed()

    def _setup_style(self):
        style = ttk.Style(self.root)

        try:
            style.theme_use("vista")
        except tk.TclError:
            pass

        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 15, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            foreground="#667085",
        )
        style.configure(
            "Error.TLabel",
            foreground="#b42318",
        )
        style.configure(
            "Success.TLabel",
            foreground="#067647",
        )
        style.configure(
            "Info.TLabel",
            foreground="#344054",
        )

    def _load_settings(self):
        if not self.settings_path.exists():
            return

        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                return

            saved_params = data.get("parameters", {})
            if not isinstance(saved_params, dict):
                saved_params = {}

            for spec in ALL_SPECS:
                try:
                    value = int(round(float(saved_params.get(spec.key, spec.default))))
                except (TypeError, ValueError, OverflowError):
                    value = spec.default

                value = max(spec.minimum, min(spec.maximum, value))
                self.parameter_vars[spec.key].set(value)

            self.mount_holes_var.set(bool(data.get("mount_holes_enabled", False)))

            print_data = data.get("print", {})
            if not isinstance(print_data, dict):
                print_data = {}

            orientation = print_data.get("orientation", "auto")
            if orientation not in ORIENTATION_LABELS:
                orientation = "auto"

            try:
                overlap = int(print_data.get("overlap", 10))
            except (TypeError, ValueError, OverflowError):
                overlap = 10

            try:
                margin = int(print_data.get("margin", 10))
            except (TypeError, ValueError, OverflowError):
                margin = 10

            overlap = max(5, min(30, overlap))
            margin = max(10, min(25, margin))

            self.print_orientation_var.set(ORIENTATION_LABELS[orientation])
            self.print_overlap_var.set(overlap)
            self.print_margin_var.set(margin)

            self.last_print_options = {
                "orientation": orientation,
                "overlap": overlap,
                "margin": margin,
            }

        except (
            OSError,
            json.JSONDecodeError,
            AttributeError,
            TypeError,
        ):
            # При повреждённом файле запускаемся с исходными настройками.
            pass

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=9)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        # Левая панель параметров.
        self.left_panel = ttk.Frame(container, width=400)
        self.left_panel.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 9),
        )
        self.left_panel.grid_propagate(False)
        self.left_panel.rowconfigure(1, weight=1)
        self.left_panel.columnconfigure(0, weight=1)

        ttk.Label(
            self.left_panel,
            text="Параметры модели",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(2, 5))

        self.controls = ScrollableFrame(self.left_panel)
        self.controls.grid(row=1, column=0, sticky="nsew")

        for group_name, specs in PARAMETER_GROUPS[:3]:
            group = ttk.LabelFrame(
                self.controls.inner,
                text=group_name,
                padding=(8, 6),
            )
            group.pack(fill="x", padx=6, pady=5)

            for spec in specs:
                self._add_parameter_control(group, spec)

        # Необязательные отверстия фланца.
        mount_group = ttk.LabelFrame(
            self.controls.inner,
            text=PARAMETER_GROUPS[3][0],
            padding=(8, 6),
        )
        mount_group.pack(fill="x", padx=6, pady=5)

        mount_option_row = ttk.Frame(mount_group)
        mount_option_row.pack(fill="x", pady=(0, 5))

        ttk.Checkbutton(
            mount_option_row,
            text="Добавить 4 отверстия по PCD",
            variable=self.mount_holes_var,
        ).pack(side="left", fill="x", expand=True)

        ttk.Button(
            mount_option_row,
            text="Сброс",
            width=6,
            command=self._reset_mount_holes,
        ).pack(side="right", padx=(5, 0))

        for spec in PARAMETER_GROUPS[3][1]:
            self._add_parameter_control(
                mount_group,
                spec,
                is_mount_control=True,
            )

        # Настройки разбиения на листы A4.
        print_group = ttk.LabelFrame(
            self.controls.inner,
            text="Печать на листах A4",
            padding=(8, 6),
        )
        print_group.pack(fill="x", padx=6, pady=5)

        orientation_row = ttk.Frame(print_group)
        orientation_row.pack(fill="x", pady=3)
        orientation_row.columnconfigure(0, weight=1)

        ttk.Label(
            orientation_row,
            text="Ориентация листов",
        ).grid(row=0, column=0, sticky="w")

        ttk.Combobox(
            orientation_row,
            textvariable=self.print_orientation_var,
            values=tuple(ORIENTATION_LABELS.values()),
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky="e", padx=(5, 0))

        ttk.Button(
            orientation_row,
            text="Сброс",
            width=6,
            command=lambda: self._reset_print_setting("orientation"),
        ).grid(row=0, column=2, sticky="e", padx=(5, 0))

        self._add_print_numeric_control(
            print_group,
            "Перекрытие листов, мм",
            "overlap",
            5,
            30,
            self.print_overlap_var,
        )
        self._add_print_numeric_control(
            print_group,
            "Поле листа, мм",
            "margin",
            10,
            25,
            self.print_margin_var,
        )

        ttk.Label(
            print_group,
            text=(
                "«Авто» выбирает ориентацию с меньшим числом листов. "
                "Перекрытие удобно для склейки страниц."
            ),
            wraplength=355,
            justify="left",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        ttk.Button(
            self.left_panel,
            text="Сбросить все параметры",
            command=self._reset_defaults,
        ).grid(row=2, column=0, sticky="ew", pady=(8, 4))

        ttk.Label(
            self.left_panel,
            text=("Автосохранение настроек:\n" f"{self.settings_path}"),
            wraplength=375,
            justify="left",
            style="Subtitle.TLabel",
        ).grid(row=3, column=0, sticky="w", pady=(3, 2))

        # Правая панель предпросмотра и экспорта.
        self.right_panel = ttk.Frame(container)
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        self.right_panel.columnconfigure(0, weight=1)
        self.right_panel.rowconfigure(1, weight=1)

        header = ttk.Frame(self.right_panel)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 7))

        ttk.Label(
            header,
            text="Предпросмотр контура",
            style="Title.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="Вид спереди · размеры показаны в миллиметрах",
            style="Subtitle.TLabel",
        ).pack(anchor="w")

        self.preview_canvas = tk.Canvas(
            self.right_panel,
            background="#f7f8fa",
            highlightthickness=1,
            highlightbackground="#d0d5dd",
        )
        self.preview_canvas.grid(row=1, column=0, sticky="nsew")
        self.preview_canvas.bind(
            "<Configure>",
            lambda _event: self._draw_preview(),
        )

        self.status_label = ttk.Label(
            self.right_panel,
            textvariable=self.status_var,
            style="Info.TLabel",
            wraplength=850,
            justify="left",
        )
        self.status_label.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 3),
        )

        ttk.Label(
            self.right_panel,
            text=(
                "Для A4 PDF печатайте на бумаге A4 в масштабе "
                "«Фактический размер» / 100%, без «Вписать в страницу». "
                "Совместите серые кресты и склейте листы с обратной стороны. "
                "На первой странице есть контрольная линия 100 мм."
            ),
            wraplength=850,
            justify="left",
            style="Subtitle.TLabel",
        ).grid(row=3, column=0, sticky="ew", pady=(0, 8))

        buttons = ttk.Frame(self.right_panel)
        buttons.grid(row=4, column=0, sticky="e")

        self.svg_button = ttk.Button(
            buttons,
            text="Экспорт SVG",
            command=self._export_svg,
            state=tk.DISABLED,
        )
        self.svg_button.pack(side="left", padx=(0, 7))

        self.dxf_button = ttk.Button(
            buttons,
            text="Экспорт DXF",
            command=self._export_dxf,
            state=tk.DISABLED,
        )
        self.dxf_button.pack(side="left", padx=(0, 7))

        self.a4_button = ttk.Button(
            buttons,
            text="Экспорт A4 PDF",
            command=self._export_a4_pdf,
            state=tk.DISABLED,
        )
        self.a4_button.pack(side="left")

    def _add_parameter_control(
        self,
        parent,
        spec: ParamSpec,
        is_mount_control: bool = False,
    ):
        variable = self.parameter_vars[spec.key]

        block = ttk.Frame(parent)
        block.pack(fill="x", pady=3)
        block.columnconfigure(0, weight=1)

        ttk.Label(
            block,
            text=f"{spec.label}, мм",
        ).grid(row=0, column=0, sticky="w")

        spinbox = ttk.Spinbox(
            block,
            from_=spec.minimum,
            to=spec.maximum,
            increment=1,
            textvariable=variable,
            width=7,
            justify="right",
        )
        spinbox.grid(row=0, column=1, sticky="e", padx=(5, 0))
        spinbox.bind(
            "<FocusOut>",
            lambda _event, key=spec.key: self._commit_spinbox(key),
        )
        spinbox.bind(
            "<Return>",
            lambda _event, key=spec.key: self._commit_spinbox(key),
        )
        self.spinboxes[spec.key] = spinbox

        # Сбрасывает только этот параметр.
        ttk.Button(
            block,
            text="Сброс",
            width=6,
            command=lambda key=spec.key: self._reset_parameter(key),
        ).grid(row=0, column=2, sticky="e", padx=(5, 0))

        scale = tk.Scale(
            block,
            from_=spec.minimum,
            to=spec.maximum,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=variable,
            showvalue=False,
            highlightthickness=0,
            borderwidth=0,
            width=12,
            sliderlength=18,
            background="#f5f6f8",
            troughcolor="#d0d5dd",
            activebackground="#98a2b3",
        )
        scale.grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(0, 1),
        )

        if is_mount_control:
            # Поле и ползунок отключаются вместе с опцией отверстий.
            # Кнопка сброса отдельного параметра остаётся доступной.
            self.mount_controls.extend([spinbox, scale])

    def _add_print_numeric_control(
        self,
        parent,
        label: str,
        key: str,
        minimum: int,
        maximum: int,
        variable: tk.IntVar,
    ):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        row.columnconfigure(0, weight=1)

        ttk.Label(row, text=label).grid(
            row=0,
            column=0,
            sticky="w",
        )

        spinbox = ttk.Spinbox(
            row,
            from_=minimum,
            to=maximum,
            increment=1,
            textvariable=variable,
            width=7,
            justify="right",
        )
        spinbox.grid(row=0, column=1, sticky="e", padx=(5, 0))
        spinbox.bind(
            "<FocusOut>",
            lambda _event, setting_key=key: self._commit_print_spinbox(setting_key),
        )
        spinbox.bind(
            "<Return>",
            lambda _event, setting_key=key: self._commit_print_spinbox(setting_key),
        )
        self.print_spinboxes[key] = spinbox

        ttk.Button(
            row,
            text="Сброс",
            width=6,
            command=lambda setting_key=key: self._reset_print_setting(setting_key),
        ).grid(row=0, column=2, sticky="e", padx=(5, 0))

    def _commit_spinbox(self, key: str):
        try:
            self.parameter_vars[key].get()
        except (tk.TclError, ValueError):
            fallback = self.last_params.get(
                key,
                SPEC_BY_KEY[key].default,
            )

            self._suspend_callbacks = True
            try:
                self.parameter_vars[key].set(fallback)
            finally:
                self._suspend_callbacks = False

            self._on_parameters_changed()

    def _read_parameters(self) -> dict:
        values = {}
        corrections = {}

        for spec in ALL_SPECS:
            value = int(self.parameter_vars[spec.key].get())
            bounded = max(spec.minimum, min(spec.maximum, value))
            values[spec.key] = bounded

            if bounded != value:
                corrections[spec.key] = bounded

        if corrections:
            self._suspend_callbacks = True
            try:
                for key, value in corrections.items():
                    self.parameter_vars[key].set(value)
            finally:
                self._suspend_callbacks = False

        return values

    def _read_print_options(self) -> dict:
        label = self.print_orientation_var.get()
        orientation = LABEL_TO_ORIENTATION.get(label, "auto")

        overlap = int(self.print_overlap_var.get())
        margin = int(self.print_margin_var.get())

        bounded_overlap = max(5, min(30, overlap))
        bounded_margin = max(10, min(25, margin))

        corrections_needed = (
            orientation != LABEL_TO_ORIENTATION.get(label, "auto")
            or bounded_overlap != overlap
            or bounded_margin != margin
            or label not in LABEL_TO_ORIENTATION
        )

        if corrections_needed:
            self._suspend_callbacks = True
            try:
                self.print_orientation_var.set(ORIENTATION_LABELS[orientation])
                self.print_overlap_var.set(bounded_overlap)
                self.print_margin_var.set(bounded_margin)
            finally:
                self._suspend_callbacks = False

        return {
            "orientation": orientation,
            "overlap": bounded_overlap,
            "margin": bounded_margin,
        }

    def _set_mount_controls_state(self):
        state = tk.NORMAL if self.mount_holes_var.get() else tk.DISABLED

        for widget in self.mount_controls:
            widget.configure(state=state)

    def _set_export_state(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.svg_button.configure(state=state)
        self.dxf_button.configure(state=state)
        self.a4_button.configure(state=state)

    def _set_status(self, text: str, kind: str = "info"):
        self.status_var.set(text)

        style = {
            "error": "Error.TLabel",
            "success": "Success.TLabel",
            "info": "Info.TLabel",
        }.get(kind, "Info.TLabel")

        self.status_label.configure(style=style)

    def _ready_status_text(self) -> str:
        params = self.current_params

        base = (
            f"Готово: {params['width']} × {params['height']} мм; "
            f"фанера {params['material_thickness']} мм; "
            f"центральное отверстие Ø{params['hole_diameter']} мм."
        )

        try:
            print_options = self._read_print_options()
            layout = calculate_a4_layout(
                params["width"],
                params["height"],
                print_options["orientation"],
                print_options["overlap"],
                print_options["margin"],
            )
        except (tk.TclError, ValueError, A4LayoutError):
            return base

        orientation_text = (
            "альбомная" if layout["orientation"] == "landscape" else "портретная"
        )

        return (
            f"{base} A4 PDF: {layout['total_pages']} листов, "
            f"{orientation_text}, перекрытие "
            f"{fmt_number(print_options['overlap'])} мм."
        )

    def _on_parameters_changed(self, *_args):
        if self._suspend_callbacks or self._closed:
            return

        self._set_mount_controls_state()

        try:
            params = self._read_parameters()
        except (tk.TclError, ValueError, TypeError, OverflowError):
            self.current_geometry = None
            self.current_params = None
            self._set_export_state(False)
            self._set_status(
                "Введите в поля целые числовые значения.",
                "error",
            )
            self._draw_preview()
            return

        self.current_params = params
        self.last_params = params.copy()

        try:
            self.current_geometry = build_geometry(
                params,
                bool(self.mount_holes_var.get()),
            )
        except GeometryError as error:
            self.current_geometry = None
            self._set_export_state(False)
            self._set_status(
                f"Контур не построен: {error}",
                "error",
            )
        else:
            self.last_geometry = self.current_geometry
            self._set_export_state(True)
            self._set_status(self._ready_status_text(), "info")

        self._schedule_save()
        self._draw_preview()

    def _on_print_settings_changed(self, *_args):
        if self._suspend_callbacks or self._closed:
            return

        try:
            self.last_print_options = self._read_print_options()
        except (tk.TclError, ValueError, TypeError, OverflowError):
            # Например, пользователь временно очистил поле ввода.
            return

        self._schedule_save()

        if self.current_geometry is not None:
            self._set_status(self._ready_status_text(), "info")

    def _draw_preview(self):
        if not hasattr(self, "preview_canvas"):
            return

        canvas = self.preview_canvas
        canvas.delete("all")

        geometry = self.current_geometry or self.last_geometry

        if geometry is None:
            canvas.create_text(
                max(canvas.winfo_width(), 300) / 2,
                max(canvas.winfo_height(), 200) / 2,
                text="Здесь появится контур детали",
                fill="#667085",
                font=("Segoe UI", 13),
            )
            return

        canvas_width = max(canvas.winfo_width(), 300)
        canvas_height = max(canvas.winfo_height(), 200)

        margin_left = 64
        margin_right = 24
        margin_top = 35
        margin_bottom = 62

        available_width = max(
            1,
            canvas_width - margin_left - margin_right,
        )
        available_height = max(
            1,
            canvas_height - margin_top - margin_bottom,
        )

        scale = min(
            available_width / geometry.width,
            available_height / geometry.height,
        )

        draw_width = geometry.width * scale
        draw_height = geometry.height * scale

        origin_x = margin_left + (available_width - draw_width) / 2
        origin_y = margin_top + (available_height - draw_height) / 2

        # Координатная сетка.
        for x in range(0, int(geometry.width) + 1, 100):
            screen_x = origin_x + x * scale
            canvas.create_line(
                screen_x,
                origin_y,
                screen_x,
                origin_y + draw_height,
                fill="#e9edf2",
            )

        for y in range(0, int(geometry.height) + 1, 100):
            screen_y = origin_y + y * scale
            canvas.create_line(
                origin_x,
                screen_y,
                origin_x + draw_width,
                screen_y,
                fill="#e9edf2",
            )

        is_current = self.current_geometry is not None

        if is_current:
            fill_color = "#ffffff"
            outline_color = "#202939"
        else:
            fill_color = "#eaecf0"
            outline_color = "#98a2b3"

        screen_points = []
        for x, y in geometry.outline:
            screen_points.extend(
                [
                    origin_x + x * scale,
                    origin_y + y * scale,
                ]
            )

        canvas.create_polygon(
            *screen_points,
            fill=fill_color,
            outline=outline_color,
            width=2,
        )

        for x, y, radius in geometry.holes:
            center_screen_x = origin_x + x * scale
            center_screen_y = origin_y + y * scale
            screen_radius = radius * scale

            canvas.create_oval(
                center_screen_x - screen_radius,
                center_screen_y - screen_radius,
                center_screen_x + screen_radius,
                center_screen_y + screen_radius,
                fill="#f7f8fa",
                outline=outline_color,
                width=2,
            )

        # Размерная линия ширины.
        bottom_y = origin_y + draw_height
        width_dimension_y = bottom_y + 25

        canvas.create_line(
            origin_x,
            bottom_y + 3,
            origin_x,
            width_dimension_y + 4,
            fill="#667085",
        )
        canvas.create_line(
            origin_x + draw_width,
            bottom_y + 3,
            origin_x + draw_width,
            width_dimension_y + 4,
            fill="#667085",
        )
        canvas.create_line(
            origin_x,
            width_dimension_y,
            origin_x + draw_width,
            width_dimension_y,
            fill="#667085",
            arrow=tk.BOTH,
        )
        canvas.create_text(
            origin_x + draw_width / 2,
            width_dimension_y + 11,
            text=f"{fmt_number(geometry.width)} мм",
            fill="#344054",
            font=("Segoe UI", 9),
        )

        # Размерная линия высоты.
        height_dimension_x = origin_x - 27

        canvas.create_line(
            height_dimension_x,
            origin_y,
            height_dimension_x,
            origin_y + draw_height,
            fill="#667085",
            arrow=tk.BOTH,
        )
        canvas.create_text(
            height_dimension_x - 11,
            origin_y + draw_height / 2,
            text=f"{fmt_number(geometry.height)} мм",
            angle=90,
            fill="#344054",
            font=("Segoe UI", 9),
        )

        if is_current:
            central_hole_diameter = geometry.holes[0][2] * 2
            legend = (
                f"Фанера {fmt_number(geometry.thickness)} мм  ·  "
                f"отверстие Ø{fmt_number(central_hole_diameter)} мм"
            )
            canvas.create_text(
                canvas_width - 12,
                14,
                anchor="ne",
                text=legend,
                fill="#475467",
                font=("Segoe UI", 9),
            )
        else:
            canvas.create_text(
                canvas_width / 2,
                15,
                anchor="n",
                text="Показан последний корректный контур",
                fill="#b42318",
                font=("Segoe UI", 9, "bold"),
            )

    def _schedule_save(self):
        if self._closed:
            return

        if self._save_after_id is not None:
            try:
                self.root.after_cancel(self._save_after_id)
            except tk.TclError:
                pass

        self._save_after_id = self.root.after(
            450,
            self._save_settings_now,
        )

    def _save_settings_now(self):
        self._save_after_id = None

        try:
            params = self._read_parameters()
        except (tk.TclError, ValueError, TypeError, OverflowError):
            return False

        try:
            print_options = self._read_print_options()
        except (tk.TclError, ValueError, TypeError, OverflowError):
            print_options = self.last_print_options.copy()

        data = {
            "version": 2,
            "parameters": params,
            "mount_holes_enabled": bool(self.mount_holes_var.get()),
            "print": print_options,
        }

        try:
            self.settings_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_path = self.settings_path.with_name(
                self.settings_path.name + ".tmp"
            )
            temporary_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary_path, self.settings_path)

        except OSError as error:
            self._set_status(
                f"Не удалось сохранить настройки: {error}",
                "error",
            )
            return False

        return True

    def _reset_parameter(self, key: str):
        """Сбрасывает один параметр к исходному значению."""
        spec = SPEC_BY_KEY[key]

        self._suspend_callbacks = True
        try:
            self.parameter_vars[key].set(spec.default)
        finally:
            self._suspend_callbacks = False

        self._on_parameters_changed()

    def _reset_mount_holes(self):
        """Выключает только опцию дополнительных отверстий."""
        self._suspend_callbacks = True
        try:
            self.mount_holes_var.set(False)
        finally:
            self._suspend_callbacks = False

        self._on_parameters_changed()

    def _commit_print_spinbox(self, key: str):
        variable = self.print_overlap_var if key == "overlap" else self.print_margin_var

        try:
            value = int(variable.get())
        except (tk.TclError, ValueError, TypeError):
            value = int(self.last_print_options.get(key, 10))

        if key == "overlap":
            value = max(5, min(30, value))
        else:
            value = max(10, min(25, value))

        self._suspend_callbacks = True
        try:
            variable.set(value)
        finally:
            self._suspend_callbacks = False

        self._on_print_settings_changed()

    def _reset_print_setting(self, key: str):
        """Сбрасывает одну настройку печати."""
        self._suspend_callbacks = True
        try:
            if key == "orientation":
                self.print_orientation_var.set(
                    ORIENTATION_LABELS[DEFAULT_PRINT_SETTINGS["orientation"]]
                )
            elif key == "overlap":
                self.print_overlap_var.set(DEFAULT_PRINT_SETTINGS["overlap"])
            elif key == "margin":
                self.print_margin_var.set(DEFAULT_PRINT_SETTINGS["margin"])
        finally:
            self._suspend_callbacks = False

        self._on_print_settings_changed()

    def _reset_defaults(self):
        """Сбрасывает параметры детали и печати."""
        confirmed = messagebox.askyesno(
            "Сброс параметров",
            "Вернуть параметры детали и печати к исходным?",
            parent=self.root,
        )

        if not confirmed:
            return

        self._suspend_callbacks = True
        try:
            for spec in ALL_SPECS:
                self.parameter_vars[spec.key].set(spec.default)

            self.mount_holes_var.set(False)

            self.print_orientation_var.set(
                ORIENTATION_LABELS[DEFAULT_PRINT_SETTINGS["orientation"]]
            )
            self.print_overlap_var.set(DEFAULT_PRINT_SETTINGS["overlap"])
            self.print_margin_var.set(DEFAULT_PRINT_SETTINGS["margin"])
        finally:
            self._suspend_callbacks = False

        self._on_parameters_changed()

    def _export_svg(self):
        if self.current_geometry is None or self.current_params is None:
            return

        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Экспорт контура в SVG",
            initialdir=str(Path.home()),
            initialfile="yakor.svg",
            defaultextension=".svg",
            filetypes=[("SVG-файл", "*.svg")],
        )

        if not filename:
            return

        path = Path(filename)
        if path.suffix.lower() != ".svg":
            path = path.with_suffix(".svg")

        try:
            write_svg(path, self.current_geometry, self.current_params)
        except OSError as error:
            messagebox.showerror(
                "Ошибка экспорта",
                f"Не удалось сохранить SVG:\n{error}",
                parent=self.root,
            )
            return

        self._set_status(f"SVG экспортирован: {path}", "success")

    def _export_dxf(self):
        if self.current_geometry is None:
            return

        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Экспорт контура в DXF",
            initialdir=str(Path.home()),
            initialfile="yakor.dxf",
            defaultextension=".dxf",
            filetypes=[("DXF-файл", "*.dxf")],
        )

        if not filename:
            return

        path = Path(filename)
        if path.suffix.lower() != ".dxf":
            path = path.with_suffix(".dxf")

        try:
            write_dxf(path, self.current_geometry)
        except OSError as error:
            messagebox.showerror(
                "Ошибка экспорта",
                f"Не удалось сохранить DXF:\n{error}",
                parent=self.root,
            )
            return

        self._set_status(f"DXF экспортирован: {path}", "success")

    def _export_a4_pdf(self):
        if self.current_geometry is None:
            return

        try:
            print_options = self._read_print_options()
        except (tk.TclError, ValueError, TypeError, OverflowError) as error:
            messagebox.showerror(
                "Ошибка параметров печати",
                f"Проверьте настройки A4:\n{error}",
                parent=self.root,
            )
            return

        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Экспорт выкройки на листах A4",
            initialdir=str(Path.home()),
            initialfile="yakor_A4.pdf",
            defaultextension=".pdf",
            filetypes=[("PDF-файл", "*.pdf")],
        )

        if not filename:
            return

        path = Path(filename)
        if path.suffix.lower() != ".pdf":
            path = path.with_suffix(".pdf")

        try:
            layout = write_tiled_pdf(
                path,
                self.current_geometry,
                print_options["orientation"],
                print_options["overlap"],
                print_options["margin"],
            )
        except (OSError, A4LayoutError, ValueError) as error:
            messagebox.showerror(
                "Ошибка экспорта",
                f"Не удалось создать A4 PDF:\n{error}",
                parent=self.root,
            )
            return

        orientation_text = (
            "альбомная" if layout["orientation"] == "landscape" else "портретная"
        )

        self._set_status(
            f"A4 PDF экспортирован: {path}. "
            f"Листов: {layout['total_pages']}, "
            f"ориентация: {orientation_text}.",
            "success",
        )

    def _on_close(self):
        self._closed = True

        if self._save_after_id is not None:
            try:
                self.root.after_cancel(self._save_after_id)
            except tk.TclError:
                pass

            self._save_after_id = None

        self._save_settings_now()
        self.root.destroy()


def main():
    root = tk.Tk()
    YakorDesignerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
