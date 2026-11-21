import json
import math
import os
from dataclasses import dataclass
from html import escape
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

APP_NAME = "YakorDesigner"


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
    """Форматирует число без лишних нулей после запятой."""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


class GeometryError(Exception):
    """Параметры не позволяют построить корректный контур."""


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
    """Добавляет дугу в виде коротких отрезков."""
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

    Координаты профиля:
      x — от левого края;
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

    # Круглая площадка должна целиком помещаться по ширине детали.
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

    if stem_width >= pad_diameter - 1.0:
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

    # Проверяем, не пересекаются ли высокие боковые упоры с круглой площадкой.
    left_ear_overlaps_pad = ear_width > center_x - radius + 1e-6
    right_ear_overlaps_pad = width - ear_width < center_x + radius - 1e-6

    if ear_top < join_y and (left_ear_overlaps_pad or right_ear_overlaps_pad):
        raise GeometryError(
            "Боковой упор заходит в область круглой площадки. "
            "Уменьшите высоту упоров или их ширину."
        )

    # Ограничиваем радиусы размерами соответствующих участков.
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

    # Начинаем снизу слева и идём по внешнему периметру.
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

    # Верх перекладины и правая сторона центральной стойки.
    add((width - ear_width, base_top))
    add((stem_right, base_top))
    add((stem_right, join_y))

    # Верхняя и боковые части круглой площадки.
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

    # Первое отверстие — центральное.
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

        # Четыре отверстия на окружности PCD.
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
    """Экспортирует контур в SVG с размерами в миллиметрах."""
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
    """Экспортирует контур в DXF R2000, единицы измерения — миллиметры."""
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

    # В SVG ось Y направлена вниз, а в CAD — вверх.
    # Преобразуем координаты, чтобы ориентация детали совпадала.
    for x, y in geometry.outline:
        dxf_y = geometry.height - y
        lines.extend(
            [
                "10",
                fmt_number(x),
                "20",
                fmt_number(dxf_y),
            ]
        )

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
        self.root.geometry("1280x820")
        self.root.minsize(1050, 700)

        self.settings_path = get_settings_path()
        self._suspend_callbacks = True
        self._save_after_id = None
        self._closed = False

        self.parameter_vars = {
            spec.key: tk.IntVar(master=root, value=spec.default) for spec in ALL_SPECS
        }
        self.mount_holes_var = tk.BooleanVar(master=root, value=False)

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

        self._load_settings()
        self._setup_style()
        self._build_ui()

        for variable in self.parameter_vars.values():
            variable.trace_add("write", self._on_parameters_changed)

        self.mount_holes_var.trace_add(
            "write",
            self._on_parameters_changed,
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

        except (
            OSError,
            json.JSONDecodeError,
            AttributeError,
            TypeError,
        ):
            # При повреждённом файле настроек используем значения по умолчанию.
            pass

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=9)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        # Левая панель параметров.
        self.left_panel = ttk.Frame(container, width=390)
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

        option_row = ttk.Frame(mount_group)
        option_row.pack(fill="x", pady=(0, 5))

        ttk.Checkbutton(
            option_row,
            text="Добавить 4 отверстия по PCD",
            variable=self.mount_holes_var,
        ).pack(side="left", fill="x", expand=True)

        ttk.Button(
            option_row,
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

        ttk.Button(
            self.left_panel,
            text="Сбросить все параметры",
            command=self._reset_defaults,
        ).grid(row=2, column=0, sticky="ew", pady=(8, 4))

        ttk.Label(
            self.left_panel,
            text=("Автосохранение настроек:\n" f"{self.settings_path}"),
            wraplength=365,
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
            wraplength=820,
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
                "Экспортируется 2D-контур для раскроя. "
                "Толщина фанеры сохраняется как справочная информация "
                "и не меняет форму плоского контура."
            ),
            wraplength=820,
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
        self.dxf_button.pack(side="left")

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

        # Сбрасывает только этот параметр, не затрагивая остальные.
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
            # Ползунки и поля фланца включаются только вместе с опцией.
            # Кнопки индивидуального сброса остаются доступными.
            self.mount_controls.extend([spinbox, scale])

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

    def _set_mount_controls_state(self):
        state = tk.NORMAL if self.mount_holes_var.get() else tk.DISABLED

        for widget in self.mount_controls:
            widget.configure(state=state)

    def _set_export_state(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.svg_button.configure(state=state)
        self.dxf_button.configure(state=state)

    def _set_status(self, text: str, kind: str = "info"):
        self.status_var.set(text)

        style = {
            "error": "Error.TLabel",
            "success": "Success.TLabel",
            "info": "Info.TLabel",
        }.get(kind, "Info.TLabel")

        self.status_label.configure(style=style)

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
            self._set_status(
                f"Готово: {params['width']} × {params['height']} мм; "
                f"фанера {params['material_thickness']} мм; "
                f"центральное отверстие Ø{params['hole_diameter']} мм.",
                "info",
            )

        self._schedule_save()
        self._draw_preview()

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

        data = {
            "version": 1,
            "parameters": params,
            "mount_holes_enabled": bool(self.mount_holes_var.get()),
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
        """Сбрасывает только один параметр к значению по умолчанию."""
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

    def _reset_defaults(self):
        """Сбрасывает все параметры к исходным значениям."""
        confirmed = messagebox.askyesno(
            "Сброс параметров",
            "Вернуть все значения к исходным?",
            parent=self.root,
        )

        if not confirmed:
            return

        self._suspend_callbacks = True
        try:
            for spec in ALL_SPECS:
                self.parameter_vars[spec.key].set(spec.default)

            self.mount_holes_var.set(False)
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

        self._set_status(
            f"SVG экспортирован: {path}",
            "success",
        )

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

        self._set_status(
            f"DXF экспортирован: {path}",
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
