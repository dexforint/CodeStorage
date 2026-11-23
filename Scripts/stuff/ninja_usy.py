import json
import math
import os
from dataclasses import dataclass
from html import escape
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

APP_NAME = "UsyDesigner"

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
        "Габариты",
        [
            ParamSpec("width", "Общая ширина", 200, 800, 400),
            ParamSpec("height", "Общая высота", 300, 900, 500),
            ParamSpec("material_thickness", "Толщина материала, Z", 6, 60, 24),
        ],
    ),
    (
        "Профиль буквы А",
        [
            ParamSpec("apex_width", "Ширина верхушки", 30, 300, 90),
            ParamSpec("leg_width", "Толщина наклонной стойки", 15, 150, 45),
            ParamSpec("window_top", "Верх окна от верха детали", 20, 400, 130),
            ParamSpec("crossbar_top", "Верх перемычки от верха детали", 80, 700, 320),
            ParamSpec("crossbar_height", "Высота перемычки", 10, 150, 35),
            ParamSpec("outline_radius", "Радиус скругления контура", 0, 60, 10),
            ParamSpec("window_radius", "Радиус углов окна", 0, 40, 8),
        ],
    ),
    (
        "Подшипник",
        [
            ParamSpec(
                "bearing_hole_diameter", "Диаметр центрального отверстия", 5, 120, 45
            ),
            ParamSpec("bearing_center_y", "Центр отверстия от верха", 10, 250, 70),
            ParamSpec("bearing_offset_x", "Смещение отверстия по X", -150, 150, 0),
        ],
    ),
    (
        "Крепёж фланца — необязательно",
        [
            ParamSpec("mount_hole_diameter", "Диаметр 4 крепёжных отверстий", 3, 20, 8),
            ParamSpec(
                "mount_spacing", "Шаг отверстий по сторонам квадрата", 20, 180, 70
            ),
        ],
    ),
]

ALL_SPECS = [
    spec for _group_name, group_specs in PARAMETER_GROUPS for spec in group_specs
]
SPEC_BY_KEY = {spec.key: spec for spec in ALL_SPECS}


@dataclass
class CutGeometry:
    width: float
    height: float
    thickness: float
    outline: list
    cutouts: list
    holes: list


class GeometryError(Exception):
    """Параметры не позволяют построить корректную геометрию."""


class A4LayoutError(Exception):
    """Не удалось подобрать раскладку для листов A4."""


def get_settings_path() -> Path:
    """Путь к настройкам программы."""
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
    """Формат числа без лишних нулей после запятой."""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def append_unique(points: list, point: tuple) -> None:
    """Добавляет точку, если она не совпадает с предыдущей."""
    x, y = point

    if points:
        previous_x, previous_y = points[-1]
        if math.hypot(x - previous_x, y - previous_y) < 1e-8:
            return

    points.append((x, y))


def rounded_polygon(
    vertices: list,
    requested_radius: float,
    max_step_degrees: float = 2.0,
) -> list:
    """
    Скругляет углы замкнутого многоугольника круговыми дугами.

    Радиус автоматически ограничивается, чтобы соседние скругления
    не пересекались на коротких рёбрах.
    """
    if len(vertices) < 3 or requested_radius <= 0:
        return list(vertices)

    corner_data = []
    count = len(vertices)

    for index in range(count):
        previous = vertices[(index - 1) % count]
        point = vertices[index]
        following = vertices[(index + 1) % count]

        to_previous = (
            previous[0] - point[0],
            previous[1] - point[1],
        )
        to_following = (
            following[0] - point[0],
            following[1] - point[1],
        )

        len_previous = math.hypot(*to_previous)
        len_following = math.hypot(*to_following)

        if len_previous < 1e-8 or len_following < 1e-8:
            corner_data.append(
                {
                    "start": point,
                    "end": point,
                    "center": None,
                    "sweep": 0.0,
                    "radius": 0.0,
                }
            )
            continue

        u = (
            to_previous[0] / len_previous,
            to_previous[1] / len_previous,
        )
        v = (
            to_following[0] / len_following,
            to_following[1] / len_following,
        )

        dot = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
        theta = math.acos(dot)

        incoming = (
            point[0] - previous[0],
            point[1] - previous[1],
        )
        outgoing = (
            following[0] - point[0],
            following[1] - point[1],
        )
        turn = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]

        # Не скругляем прямые или почти прямые углы.
        if theta < 1e-5 or math.pi - theta < 1e-5 or abs(turn) < 1e-8:
            corner_data.append(
                {
                    "start": point,
                    "end": point,
                    "center": None,
                    "sweep": 0.0,
                    "radius": 0.0,
                }
            )
            continue

        tangent_factor = math.tan(theta / 2.0)

        if tangent_factor <= 1e-8:
            corner_data.append(
                {
                    "start": point,
                    "end": point,
                    "center": None,
                    "sweep": 0.0,
                    "radius": 0.0,
                }
            )
            continue

        # Ограничиваем скругление с учётом длины соседних рёбер.
        maximum_radius = 0.45 * min(len_previous, len_following) * tangent_factor
        radius = min(float(requested_radius), maximum_radius)

        if radius <= 1e-8:
            corner_data.append(
                {
                    "start": point,
                    "end": point,
                    "center": None,
                    "sweep": 0.0,
                    "radius": 0.0,
                }
            )
            continue

        tangent_distance = radius / tangent_factor

        bisector = (u[0] + v[0], u[1] + v[1])
        bisector_length = math.hypot(*bisector)

        if bisector_length < 1e-8:
            corner_data.append(
                {
                    "start": point,
                    "end": point,
                    "center": None,
                    "sweep": 0.0,
                    "radius": 0.0,
                }
            )
            continue

        bisector_unit = (
            bisector[0] / bisector_length,
            bisector[1] / bisector_length,
        )

        center_distance = radius / math.sin(theta / 2.0)
        center = (
            point[0] + bisector_unit[0] * center_distance,
            point[1] + bisector_unit[1] * center_distance,
        )

        tangent_start = (
            point[0] + u[0] * tangent_distance,
            point[1] + u[1] * tangent_distance,
        )
        tangent_end = (
            point[0] + v[0] * tangent_distance,
            point[1] + v[1] * tangent_distance,
        )

        start_angle = math.atan2(
            tangent_start[1] - center[1],
            tangent_start[0] - center[0],
        )
        end_angle = math.atan2(
            tangent_end[1] - center[1],
            tangent_end[0] - center[0],
        )

        # Направление обхода угла определяется направлением контура.
        if turn > 0:
            sweep = (end_angle - start_angle) % (2.0 * math.pi)
        else:
            sweep = -((start_angle - end_angle) % (2.0 * math.pi))

        # Для кругового скругления дуга не должна быть большой.
        expected_sweep = math.pi - theta
        if abs(sweep) > math.pi + 1e-6:
            sweep = math.copysign(expected_sweep, turn)

        corner_data.append(
            {
                "start": tangent_start,
                "end": tangent_end,
                "center": center,
                "start_angle": start_angle,
                "sweep": sweep,
                "radius": radius,
            }
        )

    result = []

    for corner in corner_data:
        start = corner["start"]
        end = corner["end"]
        center = corner["center"]
        radius = corner["radius"]
        sweep = corner["sweep"]

        append_unique(result, start)

        if center is None or radius <= 0 or abs(sweep) < 1e-8:
            continue

        start_angle = corner["start_angle"]
        max_step = math.radians(max_step_degrees)
        step_count = max(1, math.ceil(abs(sweep) / max_step))

        # Добавляем промежуточные точки дуги, затем точную конечную точку.
        for step in range(1, step_count):
            angle = start_angle + sweep * step / step_count
            arc_point = (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            )
            append_unique(result, arc_point)

        append_unique(result, end)

    return result


def fit_geometry_to_dimensions(
    outline: list,
    cutouts: list,
    holes: list,
    target_width: float,
    target_height: float,
):
    """
    Нормирует округлённый внешний контур на заданные габариты.

    Центры отверстий и внутренних вырезов масштабируются вместе с контуром.
    Диаметры круглых отверстий сохраняются.
    """
    if not outline:
        raise GeometryError("Внешний контур пуст.")

    min_x = min(point[0] for point in outline)
    max_x = max(point[0] for point in outline)
    min_y = min(point[1] for point in outline)
    max_y = max(point[1] for point in outline)

    span_x = max_x - min_x
    span_y = max_y - min_y

    if span_x <= 1e-8 or span_y <= 1e-8:
        raise GeometryError("У контура нулевой размер.")

    scale_x = target_width / span_x
    scale_y = target_height / span_y

    def transform_point(point):
        return (
            (point[0] - min_x) * scale_x,
            (point[1] - min_y) * scale_y,
        )

    fitted_outline = [transform_point(point) for point in outline]
    fitted_cutouts = [
        [transform_point(point) for point in cutout] for cutout in cutouts
    ]

    fitted_holes = []
    for center_x, center_y, radius in holes:
        fitted_x, fitted_y = transform_point((center_x, center_y))
        fitted_holes.append((fitted_x, fitted_y, radius))

    return fitted_outline, fitted_cutouts, fitted_holes


def build_geometry(params: dict, mount_holes_enabled: bool) -> CutGeometry:
    """Строит контур А, окно, отверстия и проверяет их взаимное положение."""
    width = float(params["width"])
    height = float(params["height"])
    thickness = float(params["material_thickness"])

    apex_width = float(params["apex_width"])
    leg_width = float(params["leg_width"])
    window_top = float(params["window_top"])
    crossbar_top = float(params["crossbar_top"])
    crossbar_height = float(params["crossbar_height"])
    outline_radius = float(params["outline_radius"])
    window_radius = float(params["window_radius"])

    bearing_hole_diameter = float(params["bearing_hole_diameter"])
    bearing_center_y = float(params["bearing_center_y"])
    bearing_offset_x = float(params["bearing_offset_x"])

    if width <= 0 or height <= 0 or thickness <= 0:
        raise GeometryError("Габариты и толщина должны быть больше нуля.")

    if apex_width <= 0 or apex_width >= width - 10:
        raise GeometryError(
            "Ширина верхушки должна быть меньше общей ширины " "хотя бы на 10 мм."
        )

    if leg_width <= 0:
        raise GeometryError("Толщина стоек должна быть больше нуля.")

    if window_top < 0:
        raise GeometryError("Верх окна не может находиться выше детали.")

    if crossbar_top <= window_top + 20:
        raise GeometryError(
            "Перемычка должна располагаться минимум на 20 мм ниже верха окна."
        )

    crossbar_bottom = crossbar_top + crossbar_height

    if crossbar_bottom >= height - 5:
        raise GeometryError(
            "Перемычка слишком близко к нижнему краю. "
            "Уменьшите её высоту или опустите/поднимите перемычку."
        )

    if crossbar_height < 5:
        raise GeometryError("Высота перемычки слишком мала.")

    slope = ((width - apex_width) / 2.0) / height
    normal_offset = leg_width * math.sqrt(1.0 + slope * slope)

    def outer_left_x(y):
        return ((width - apex_width) / 2.0) * (1.0 - y / height)

    def inner_left_x(y):
        # Смещение внутрь перпендикулярно наклонной стороне.
        return outer_left_x(y) + normal_offset

    def inner_right_x(y):
        return width - inner_left_x(y)

    def gap_width(y):
        return inner_right_x(y) - inner_left_x(y)

    if gap_width(window_top) <= 10:
        raise GeometryError(
            "Верхнее окно слишком узкое. Уменьшите толщину стоек "
            "или измените положение окна."
        )

    if gap_width(crossbar_top) <= 10:
        raise GeometryError("Недостаточно места для верхнего окна у перемычки.")

    if gap_width(crossbar_bottom) <= 10:
        raise GeometryError("Недостаточно места для нижнего открытого проёма.")

    bearing_center_x = width / 2.0 + bearing_offset_x
    bearing_radius = bearing_hole_diameter / 2.0

    def circle_fits_in_top_panel(cx, cy, radius, clearance=3.0):
        """
        Проверяет, что круглое отверстие с запасом остаётся в верхней
        сплошной части детали, выше верхнего окна.
        """
        check_radius = radius + clearance
        sample_count = 72

        for index in range(sample_count):
            angle = 2.0 * math.pi * index / sample_count
            x = cx + check_radius * math.cos(angle)
            y = cy + check_radius * math.sin(angle)

            if y < 0 or y >= window_top:
                return False

            left_edge = outer_left_x(y)
            right_edge = width - left_edge

            if x < left_edge or x > right_edge:
                return False

        return True

    if not circle_fits_in_top_panel(
        bearing_center_x,
        bearing_center_y,
        bearing_radius,
    ):
        raise GeometryError(
            "Центральное отверстие не помещается в верхней части детали. "
            "Измените его диаметр/положение или опустите начало окна."
        )

    holes = [
        (
            bearing_center_x,
            bearing_center_y,
            bearing_radius,
        )
    ]

    if mount_holes_enabled:
        mount_diameter = float(params["mount_hole_diameter"])
        mount_spacing = float(params["mount_spacing"])
        mount_radius = mount_diameter / 2.0

        if mount_spacing <= mount_diameter + 3:
            raise GeometryError(
                "Крепёжные отверстия расположены слишком близко друг к другу."
            )

        center_to_mount = mount_spacing / math.sqrt(2.0)
        if center_to_mount < bearing_radius + mount_radius + 3:
            raise GeometryError(
                "Крепёжные отверстия пересекаются с центральным. "
                "Увеличьте шаг отверстий или уменьшите их диаметры."
            )

        for dx in (-mount_spacing / 2.0, mount_spacing / 2.0):
            for dy in (-mount_spacing / 2.0, mount_spacing / 2.0):
                mount_x = bearing_center_x + dx
                mount_y = bearing_center_y + dy

                if not circle_fits_in_top_panel(
                    mount_x,
                    mount_y,
                    mount_radius,
                ):
                    raise GeometryError(
                        "Крепёжные отверстия не помещаются в верхней части "
                        "детали. Измените шаг отверстий или положение "
                        "центра подшипника."
                    )

                holes.append((mount_x, mount_y, mount_radius))

    # Внешний контур с нижней открытой выемкой под перемычкой.
    outline_raw = [
        (0.0, height),
        ((width - apex_width) / 2.0, 0.0),
        (width - (width - apex_width) / 2.0, 0.0),
        (width, height),
        (inner_right_x(height), height),
        (inner_right_x(crossbar_bottom), crossbar_bottom),
        (inner_left_x(crossbar_bottom), crossbar_bottom),
        (inner_left_x(height), height),
    ]

    # Верхнее окно между стойками и перемычкой.
    window_raw = [
        (inner_left_x(window_top), window_top),
        (inner_right_x(window_top), window_top),
        (inner_right_x(crossbar_top), crossbar_top),
        (inner_left_x(crossbar_top), crossbar_top),
    ]

    outline = rounded_polygon(outline_raw, outline_radius)
    window = rounded_polygon(window_raw, window_radius)

    outline, cutouts, holes = fit_geometry_to_dimensions(
        outline,
        [window],
        holes,
        width,
        height,
    )

    return CutGeometry(
        width=width,
        height=height,
        thickness=thickness,
        outline=outline,
        cutouts=cutouts,
        holes=holes,
    )


def polygon_svg_path(points: list) -> str:
    if not points:
        return ""

    return (
        "M " + " L ".join(f"{fmt_number(x)},{fmt_number(y)}" for x, y in points) + " Z"
    )


def write_svg(path: Path, geometry: CutGeometry, params: dict) -> None:
    """Экспортирует контуры реза в SVG, единицы измерения — миллиметры."""
    path.parent.mkdir(parents=True, exist_ok=True)

    description = (
        f"Усы OCR, профиль A. Размер: "
        f"{fmt_number(geometry.width)} x {fmt_number(geometry.height)} мм. "
        f"Толщина материала: {fmt_number(geometry.thickness)} мм. "
        f"Центральное отверстие: "
        f"Ø{fmt_number(params['bearing_hole_diameter'])} мм."
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
        "  <title>Усы OCR — профиль A</title>",
        f"  <desc>{escape(description)}</desc>",
        (
            '  <g id="CUT" fill="none" stroke="#000000" '
            'stroke-width="0.2" stroke-linejoin="round" '
            'stroke-linecap="round">'
        ),
        (f'    <path id="outline" ' f'd="{polygon_svg_path(geometry.outline)}"/>'),
    ]

    for index, cutout in enumerate(geometry.cutouts, start=1):
        lines.append(
            f'    <path id="cutout-{index}" ' f'd="{polygon_svg_path(cutout)}"/>'
        )

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
    """Экспортирует контуры в ASCII DXF R2000 с единицами в миллиметрах."""
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
        f"MATERIAL_THICKNESS_MM={fmt_number(geometry.thickness)}",
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
    ]

    def add_polyline(points: list):
        lines.extend(
            [
                "0",
                "LWPOLYLINE",
                "100",
                "AcDbEntity",
                "8",
                "CUT",
                "100",
                "AcDbPolyline",
                "90",
                str(len(points)),
                "70",
                "1",
            ]
        )

        # DXF обычно использует ось Y вверх; отражаем координаты Y.
        for x, y in points:
            dxf_y = geometry.height - y
            lines.extend(
                [
                    "10",
                    fmt_number(x),
                    "20",
                    fmt_number(dxf_y),
                ]
            )

    add_polyline(geometry.outline)

    for cutout in geometry.cutouts:
        add_polyline(cutout)

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
    """Количество листов по одной оси."""
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
    """Подбирает разбиение полноразмерного чертежа на страницы A4."""
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
    commands.append("h S")


def _pdf_escape_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_add_text(
    commands: list,
    text: str,
    x_mm: float,
    y_mm: float,
    font_size: float = 8.0,
) -> None:
    """Добавляет текст на страницу PDF."""
    x_pt = x_mm * MM_TO_PT
    y_pt = y_mm * MM_TO_PT

    commands.append(
        f"BT /F1 {fmt_number(font_size)} Tf "
        f"1 0 0 1 {fmt_number(x_pt)} {fmt_number(y_pt)} Tm "
        f"({_pdf_escape_text(text)}) Tj ET"
    )


def _pdf_add_polygon(commands: list, points: list) -> None:
    if not points:
        return

    first_x, first_y = points[0]
    commands.append(f"{fmt_number(first_x)} {fmt_number(first_y)} m")

    for x, y in points[1:]:
        commands.append(f"{fmt_number(x)} {fmt_number(y)} l")

    commands.append("h S")


def _make_pdf_page_content(
    geometry: CutGeometry,
    layout: dict,
    tile_column: int,
    tile_row: int,
    page_number: int,
) -> bytes:
    """Формирует содержимое одной страницы полноразмерной выкройки."""
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

    # Координаты модели переводятся в PDF points с масштабом 1:1.
    # Отрицательная шкала по Y переводит начало модели в верх страницы.
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

    _pdf_add_polygon(commands, geometry.outline)

    for cutout in geometry.cutouts:
        _pdf_add_polygon(commands, cutout)

    for x, y, radius in geometry.holes:
        _pdf_circle_path(commands, x, y, radius)

    commands.append("Q")

    # Метки совмещения в области перекрытия соседних страниц.
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
                f"{fmt_number(page_x - half_pt)} "
                f"{fmt_number(page_y)} m "
                f"{fmt_number(page_x + half_pt)} "
                f"{fmt_number(page_y)} l"
            )
            commands.append(
                f"{fmt_number(page_x)} "
                f"{fmt_number(page_y - half_pt)} m "
                f"{fmt_number(page_x)} "
                f"{fmt_number(page_y + half_pt)} l S"
            )

        commands.append("Q")

    commands.append("0 0 0 rg")

    header = (
        f"MODEL {fmt_number(geometry.width)} x "
        f"{fmt_number(geometry.height)} mm | "
        "A TEMPLATE | PRINT AT 100%"
    )
    _pdf_add_text(
        commands,
        header,
        margin,
        page_height - 8.0,
        8.0,
    )

    footer = (
        f"PAGE {page_number}/{layout['total_pages']} | "
        f"ROW {tile_row + 1}/{layout['rows']} | "
        f"COL {tile_column + 1}/{layout['columns']} | "
        f"OVERLAP {fmt_number(overlap)} mm"
    )
    _pdf_add_text(commands, footer, margin, 6.0, 7.0)

    # Контрольная линейка 100 мм на первой странице.
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
    """Записывает многостраничную выкройку A4 в PDF без внешних библиотек."""
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
    """Прокручиваемая область для параметров."""

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


class UsyDesignerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Усы OCR — редактор профиля A")
        self.root.geometry("1340x860")
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
            # Если файл настроек повреждён, используем значения по умолчанию.
            pass

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=9)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        # Левая панель параметров.
        self.left_panel = ttk.Frame(container, width=420)
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

        # Настройки фланца и необязательные крепёжные отверстия.
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
            text="Добавить 4 отверстия по квадрату",
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

        # Настройки печати на A4.
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
                "Авто выбирает раскладку с наименьшим числом листов. "
                "Серые кресты помогают совмещать соседние страницы."
            ),
            wraplength=385,
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
            wraplength=395,
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
            text="Предпросмотр профиля А",
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
                "Для выкройки A4 печатайте на бумаге A4 в масштабе "
                "«Фактический размер» / 100%, без «Вписать в страницу». "
                "На первой странице проверьте контрольную линию 100 мм."
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

        self.pdf_button = ttk.Button(
            buttons,
            text="Экспорт A4 PDF",
            command=self._export_a4_pdf,
            state=tk.DISABLED,
        )
        self.pdf_button.pack(side="left")

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

        needs_update = (
            label not in LABEL_TO_ORIENTATION
            or overlap != bounded_overlap
            or margin != bounded_margin
        )

        if needs_update:
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
        self.pdf_button.configure(state=state)

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
            f"материал {params['material_thickness']} мм; "
            f"центральное отверстие "
            f"Ø{params['bearing_hole_diameter']} мм."
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
            f"{print_options['overlap']} мм."
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

        # Сетка предпросмотра.
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
            cutout_color = "#f7f8fa"
        else:
            fill_color = "#eaecf0"
            outline_color = "#98a2b3"
            cutout_color = "#f7f8fa"

        screen_outline = []
        for x, y in geometry.outline:
            screen_outline.extend(
                [
                    origin_x + x * scale,
                    origin_y + y * scale,
                ]
            )

        canvas.create_polygon(
            *screen_outline,
            fill=fill_color,
            outline=outline_color,
            width=2,
        )

        for cutout in geometry.cutouts:
            screen_cutout = []
            for x, y in cutout:
                screen_cutout.extend(
                    [
                        origin_x + x * scale,
                        origin_y + y * scale,
                    ]
                )

            canvas.create_polygon(
                *screen_cutout,
                fill=cutout_color,
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
                fill=cutout_color,
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
            canvas.create_text(
                canvas_width - 12,
                14,
                anchor="ne",
                text=(f"Материал {fmt_number(geometry.thickness)} мм"),
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
            "version": 1,
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
        """Сбрасывает только один параметр."""
        spec = SPEC_BY_KEY[key]

        self._suspend_callbacks = True
        try:
            self.parameter_vars[key].set(spec.default)
        finally:
            self._suspend_callbacks = False

        self._on_parameters_changed()

    def _reset_mount_holes(self):
        """Выключает только опцию крепёжных отверстий."""
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
                orientation = DEFAULT_PRINT_SETTINGS["orientation"]
                self.print_orientation_var.set(ORIENTATION_LABELS[orientation])
            elif key == "overlap":
                self.print_overlap_var.set(DEFAULT_PRINT_SETTINGS["overlap"])
            elif key == "margin":
                self.print_margin_var.set(DEFAULT_PRINT_SETTINGS["margin"])
        finally:
            self._suspend_callbacks = False

        self._on_print_settings_changed()

    def _reset_defaults(self):
        """Сбрасывает все параметры детали и печати."""
        confirmed = messagebox.askyesno(
            "Сброс параметров",
            "Вернуть все параметры к исходным?",
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
            initialfile="usy_A.svg",
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
            initialfile="usy_A.dxf",
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
            initialfile="usy_A_A4.pdf",
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
    UsyDesignerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
