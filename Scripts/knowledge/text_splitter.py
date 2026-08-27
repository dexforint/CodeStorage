#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Разделитель текста из буфера обмена.

Два режима работы с книгой:

1. «Вся книга сразу» — сначала молча скармливаем фрагменты,
   потом просим анализ и конспекты.
2. «По главам» — каждая глава копируется уже с инструкцией
   разобрать именно её.

Режет текст по лимиту символов или по главам, не разрывая слова.
"""

from __future__ import annotations

import re
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk, messagebox

DEFAULT_MAX_CHARS = 100_000
MIN_MAX_CHARS = 500
MAX_MAX_CHARS = 500_000

MODE_WHOLE = "whole"
MODE_CHAPTERS = "chapters"

PROMPT_START = (
    "Тебе будет предоставлен текст книги. Этот текст будет предоставляться "
    "фрагментарно (одно сообщение - один фрагмент). Пока тебе предоставляются "
    "фрагменты - тебе не нужно никак отвечать на них."
)

PROMPT_ANALYZE = (
    "Все фрагменты тебе были выданы. Теперь твоя задача - понять и проанализировать книгу.\n"
    "Какова центральная идея всей книги?\n"
    "Составь подробное оглавление книги с описанием каждой главыв 3–5 предложениях. "
    "Укажи главную идею каждой главы."
)

PROMPT_SUMMARY = (
    "Теперь для каждой главы сделай детальный и понятный конспект. Конспект должен быть "
    "понятен и его должно быть интересно читать. Учти, что я не являюсь не специалистом "
    "по теме книги.\n"
    "Не делай всё и сразу. Для каждого сообщения - отдельная глава.\n"
    "В данном конспекте ты можешь давать свои комментарии как специалист по теме книги. "
    "Твои комментарии должны быть полезны.\n"
    "Если есть упражнения - выпиши их.\n"
    "В конце каждой главы (сообщения) дай проверяющие вопросы с ответами (если они есть).\n"
)

PROMPT_CHAPTER_START = (
    "Я буду присылать книгу по одной главе. Каждое сообщение — текст главы "
    "и сразу задание к ней.\n"
    "\n"
    "Разбирай только присланную главу. Не пересказывай всю книгу, не забегай вперёд "
    "и не начинай следующую главу сам.\n"
    "Пиши понятно, будто объясняешь неспециалисту. Свои комментарии давай только "
    "если они реально полезны.\n"
    "\n"
    "Жди первую главу. На это сообщение ответь одной строкой: Готов разбирать по главам."
)

PROMPT_CHAPTER_FINALE = (
    "Все главы тебе уже присылались по отдельности. Теперь собери книгу целиком.\n"
    "\n"
    "1. Центральная идея всей книги — 5–8 предложений, без общих мотивационных фраз.\n"
    "2. Как главы связаны: какая зачем нужна и как из них складывается метод.\n"
    "3. Главные практики книги списком: что делать, сколько минут, чего не делать.\n"
    "4. Где автор утверждает жёстко, а где это скорее метафора.\n"
    "5. Что можно выкинуть без потери метода.\n"
    "6. Мой план на 7 дней: по одному микрошагу на каждый день."
)

PROMPTS_WHOLE = (
    {
        "key": "start",
        "title": "1. Не отвечай",
        "hint": "Перед фрагментами: модель молчит, пока идут куски книги.",
        "text": PROMPT_START,
    },
    {
        "key": "analyze",
        "title": "2. Анализ книги",
        "hint": "После всех фрагментов: центральная идея и оглавление.",
        "text": PROMPT_ANALYZE,
    },
    {
        "key": "summary",
        "title": "3. Конспект глав",
        "hint": "Потом: по одной главе в сообщении, с вопросами.",
        "text": PROMPT_SUMMARY,
    },
)

PROMPTS_CHAPTERS = (
    {
        "key": "chapter_start",
        "title": "1. Старт по главам",
        "hint": "Перед первой главой: модель ждёт разбор по одной главе.",
        "text": PROMPT_CHAPTER_START,
    },
    {
        "key": "chapter_finale",
        "title": "2. Итог книги",
        "hint": "После всех глав: собрать идею, практики и план на неделю.",
        "text": PROMPT_CHAPTER_FINALE,
    },
)

CONTINUE_KEY = "continue"
CONTINUE_PHRASES = (
    "Двигайся дальше",
    "Работай дальше",
    "Переходи дальше",
    "Продолжай",
    "Жду твоё следующее сообщение",
    "Идём дальше",
    "К следующему шагу",
    "Продолжай дальше",
    "Не останавливайся",
    "Продолжай в том же духе",
)

CHAPTER_RE = re.compile(
    r"(?:^|(?<=\n))[ \t]*(?:"
    r"Глава\s+\d+"
    r"|Введение\b"
    r"|Заключение\b"
    r"|Содержание\b"
    r"|Эпилог\b"
    r"|Пролог\b"
    r"|Chapter\s+\d+"
    r"|Introduction\b"
    r"|Conclusion\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

MAJOR_CHAPTER_RE = re.compile(
    r"(?:^|(?<=\n))[ \t]*("
    r"(?:Глава|Chapter)\s+\d+[^\n]*"
    r"|Введение\b[^\n]*"
    r"|Заключение\b[^\n]*"
    r"|Содержание\b[^\n]*"
    r"|Оглавление\b[^\n]*"
    r"|Эпилог\b[^\n]*"
    r"|Пролог\b[^\n]*"
    r"|Preface\b[^\n]*"
    r"|Introduction\b[^\n]*"
    r"|Conclusion\b[^\n]*"
    r"|Epilogue\b[^\n]*"
    r"|Prologue\b[^\n]*"
    r"|Afterword\b[^\n]*"
    r"|Послесловие\b[^\n]*"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

HEADING_KEY_RE = re.compile(
    r"(?:"
    r"(?:глава|chapter)\s+\d+"
    r"|введение"
    r"|заключение"
    r"|содержание"
    r"|оглавление"
    r"|эпилог"
    r"|пролог"
    r"|послесловие"
    r"|preface"
    r"|introduction"
    r"|conclusion"
    r"|epilogue"
    r"|prologue"
    r"|afterword"
    r")",
    re.IGNORECASE,
)

BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")
NEWLINE_RE = re.compile(r"\n")
SENTENCE_RE = re.compile(r"[.!?…][»\"”)\]]?[ \t]+")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class Chunk:
    text: str
    title: str = ""
    kind: str = "limit"  # limit | preamble | chapter | chapter_part
    part_index: int = 1
    part_total: int = 1
    is_last_section: bool = False


def format_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def make_preview(text: str, limit: int = 90) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def chapter_instruction(chunk: Chunk, index: int, total: int) -> str:
    title = chunk.title or f"фрагмент {index}/{total}"
    lines = [
        f"Это «{title}» — фрагмент {index} из {total}.",
        "",
    ]

    if chunk.kind == "preamble":
        lines += [
            "Это ещё не глава, а начало книги: титул, выходные данные или предисловие.",
            "Коротко скажи, что это за книга и кому она адресована. Полный разбор глав не начинай.",
            "Жди следующую главу.",
        ]
        return "\n".join(lines)

    if chunk.kind == "chapter_part" and chunk.part_index < chunk.part_total:
        lines += [
            f"Это не вся глава, а часть {chunk.part_index} из {chunk.part_total}.",
            "Кратко зафиксируй ход мысли. Полный конспект этой главы сделай только после последней части.",
            "Не переходи к следующей главе и не выдумывай недостающий текст.",
        ]
        return "\n".join(lines)

    if chunk.kind == "chapter_part":
        lines += [
            f"Это последняя часть главы ({chunk.part_index} из {chunk.part_total}).",
            "Опирайся и на предыдущую часть, если помнишь её. Теперь сделай полный конспект всей главы.",
            "",
        ]

    lines += [
        "Проанализируй только эту главу. Другие главы не трогай, не пересказывай всю книгу и не выдумывай продолжение.",
        "",
        "Сделай детальный и понятный конспект. Его должно быть интересно читать. "
        "Учти, что я не специалист по теме книги.",
        "",
        "Формат:",
        "— Зачем эта глава в книге",
        "— Ход мысли автора",
        "— Ключевые идеи словами автора",
        "— Практика: что делать, сколько минут, чего не делать",
        "— Твои комментарии как специалиста — только если они полезны",
        "— 3 проверяющих вопроса с ответами",
        "— Один микрошаг на сегодня, если глава про практику",
        "",
        "Не переходи к следующей главе сам: жди следующее сообщение.",
    ]

    if chunk.is_last_section:
        lines += [
            "",
            "Это последний фрагмент книги. После конспекта этой главы коротко, в 5–8 предложениях, "
            "собери нить всей книги. Развёрнутый итог не пиши — его я попрошу отдельно.",
        ]

    return "\n".join(lines)


def wrap_fragment(index: int, total: int, chunk: Chunk, with_instruction: bool) -> str:
    header = f"Фрагмент {index}/{total}"
    if chunk.title:
        header = f"{header} — {chunk.title}"
    payload = f"{header}\n```\n{chunk.text}\n```"
    if with_instruction:
        payload = f"{payload}\n\n{chapter_instruction(chunk, index, total)}"
    return payload


def _last_good_cut(
    window: str, start: int, pattern: re.Pattern, use_start: bool
) -> int | None:
    last = None
    min_keep = min(80, max(1, len(window) // 6))
    for match in pattern.finditer(window):
        pos = start + (match.start() if use_start else match.end())
        offset = pos - start
        if min_keep <= offset <= len(window):
            last = pos
    return last


def find_cut(text: str, start: int, limit: int) -> int:
    n = len(text)
    max_end = min(start + limit, n)
    if max_end >= n:
        return n

    window = text[start:max_end]
    for pattern, use_start in (
        (CHAPTER_RE, True),
        (BLANK_LINE_RE, False),
        (NEWLINE_RE, False),
        (SENTENCE_RE, False),
        (WHITESPACE_RE, False),
    ):
        cut = _last_good_cut(window, start, pattern, use_start)
        if cut is not None and cut > start:
            return cut
    return max_end


def split_text(text: str, max_chars: int) -> list[str]:
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        if n - start <= max_chars:
            chunks.append(text[start:])
            break
        cut = find_cut(text, start, max_chars)
        if cut <= start:
            cut = min(start + max_chars, n)
        chunks.append(text[start:cut])
        start = cut
    return chunks


def normalize_heading(raw: str) -> str:
    collapsed = re.sub(r"\s+", " ", raw).strip().lower()
    match = HEADING_KEY_RE.search(collapsed)
    if not match:
        return collapsed[:48]
    return re.sub(r"\s+", " ", match.group(0)).strip().lower()


def extract_title(text: str, start: int) -> str:
    snippet = text[start : start + 240]
    line = snippet.split("\n", 1)[0]
    line = re.sub(r"\s+", " ", line).strip(" \t.-—")
    if not line:
        return "Глава"
    if len(line) > 86:
        return line[:85] + "…"
    return line


def find_chapter_starts(text: str) -> list[int]:
    last_by_key: dict[str, int] = {}
    order: list[str] = []
    for match in MAJOR_CHAPTER_RE.finditer(text):
        key = normalize_heading(match.group(1))
        if key not in last_by_key:
            order.append(key)
        last_by_key[key] = match.start()

    starts = sorted({last_by_key[key] for key in order})
    filtered: list[int] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        letters = re.sub(r"\d+", "", text[start:end])
        letters = re.sub(r"\s+", "", letters)
        if i < len(starts) - 1 and len(letters) < 180:
            continue
        filtered.append(start)
    return filtered


def split_by_limit(text: str, max_chars: int) -> list[Chunk]:
    parts = split_text(text, max_chars)
    total = len(parts)
    return [
        Chunk(text=part, kind="limit", is_last_section=(i == total - 1))
        for i, part in enumerate(parts)
    ]


def split_by_chapters(text: str, max_chars: int) -> list[Chunk]:
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return []

    starts = find_chapter_starts(text)
    if not starts:
        return split_by_limit(text, max_chars)

    chunks: list[Chunk] = []
    if starts[0] > 0:
        preamble = text[: starts[0]]
        if preamble.strip():
            chunks.append(Chunk(text=preamble, title="Начало книги", kind="preamble"))

    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        body = text[start:end]
        title = extract_title(text, start)
        last_section = i == len(starts) - 1
        if len(body) <= max_chars:
            chunks.append(
                Chunk(
                    text=body,
                    title=title,
                    kind="chapter",
                    is_last_section=last_section,
                )
            )
            continue
        parts = split_text(body, max_chars)
        total = len(parts)
        for idx, part in enumerate(parts, start=1):
            chunks.append(
                Chunk(
                    text=part,
                    title=f"{title} — часть {idx}/{total}",
                    kind="chapter_part",
                    part_index=idx,
                    part_total=total,
                    is_last_section=last_section and idx == total,
                )
            )
    return chunks


class ScrollFrame(ttk.Frame):
    def __init__(self, master: tk.Misc, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#f3f0e8")
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        for widget in (self.inner, self.canvas):
            widget.bind("<Enter>", self._bind_wheel)
            widget.bind("<Leave>", self._unbind_wheel)

    def _on_inner_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._win, width=event.width)

    def _bind_wheel(self, _event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel)
        self.canvas.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self, _event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_wheel(self, event: tk.Event) -> None:
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            self.canvas.yview_scroll(-3, "units")
        else:
            self.canvas.yview_scroll(3, "units")

    def clear(self) -> None:
        for child in self.inner.winfo_children():
            child.destroy()
        self.canvas.yview_moveto(0)


class App(tk.Tk):
    BG = "#f3f0e8"
    CARD = "#fffdf7"
    ACCENT = "#1b4332"
    MUTED = "#5c574c"
    PROMPT_BG = "#e8f0ea"
    PROMPT_OK = "#d8f3dc"
    MODE_ON = "#95d5b2"
    MODE_OFF = "#fffdf7"
    DONE = "#d8f3dc"

    def __init__(self) -> None:
        super().__init__()
        self.title("Разделитель текста")
        self.minsize(720, 640)
        self.geometry("840x820")
        self.configure(bg=self.BG)
        self._enable_dpi_awareness()

        self.work_mode = MODE_WHOLE
        self.split_by_chapters_mode = False
        self.source_text = ""
        self.chunks: list[Chunk] = []
        self.copied_parts: set[int] = set()
        self.part_buttons: list[tk.Button] = []
        self.prompt_buttons: dict[str, tk.Button] = {}
        self.continue_button: tk.Button | None = None
        self.chapter_button: tk.Button | None = None
        self.mode_buttons: dict[str, tk.Button] = {}
        self.last_copied: int | None = None
        self.last_prompt: str | None = None
        self.continue_index = 0
        self.last_continue_phrase: str | None = None
        self.chapters_found = 0

        self.max_var = tk.StringVar(value=str(DEFAULT_MAX_CHARS))
        self.status_var = tk.StringVar(value=self._idle_status())

        self._build_style()
        self._build_ui()
        self._apply_mode()

    @staticmethod
    def _enable_dpi_awareness() -> None:
        if sys.platform == "win32":
            try:
                from ctypes import windll

                windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass

    def _idle_status(self) -> str:
        if self.work_mode == MODE_CHAPTERS:
            return (
                "Режим «по главам»: скопируйте книгу и нажмите «Взять из буфера». "
                "Каждая глава уйдёт в буфер уже с заданием на разбор."
            )
        return (
            "Режим «вся книга сразу»: скопируйте текст и нажмите «Взять из буфера». "
            "Сначала фрагменты, потом анализ."
        )

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Root.TFrame", background=self.BG)
        style.configure("PromptBox.TFrame", background="#e7efe8")
        style.configure("PromptBox.TLabel", background="#e7efe8", foreground="#1a1814")
        style.configure(
            "PromptMuted.TLabel", background="#e7efe8", foreground=self.MUTED
        )
        style.configure("TLabel", background=self.BG, foreground="#1a1814")
        style.configure("Muted.TLabel", background=self.BG, foreground=self.MUTED)
        style.configure(
            "Title.TLabel",
            background=self.BG,
            foreground=self.ACCENT,
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background=self.BG,
            foreground=self.MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 7)
        )
        style.configure("TSpinbox", padding=4)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, style="Root.TFrame", padding=16)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="Разделитель текста", style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            root,
            text="Два способа скормить книгу модели: целиком молча, потом анализ — "
            "или сразу по главам, с заданием после каждой.",
            style="Muted.TLabel",
            wraplength=800,
        ).pack(anchor="w", pady=(4, 10))

        self._build_mode_bar(root)
        self._build_prompt_bar(root)
        self._build_controls(root)

        self.status = ttk.Label(
            root, textvariable=self.status_var, style="Status.TLabel"
        )
        self.status.pack(anchor="w", pady=(0, 8))

        self.scroll = ScrollFrame(root)
        self.scroll.pack(fill="both", expand=True)
        ttk.Label(
            root,
            text="Нажмите на фрагмент — он скопируется в буфер обмена.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(10, 0))

    def _plain_button(self, parent: tk.Misc, **kwargs) -> tk.Button:
        options = dict(
            padx=12,
            pady=8,
            font=("Segoe UI", 9),
            bg=self.CARD,
            fg="#1a1814",
            activebackground="#d8f3dc",
            activeforeground="#1a1814",
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            cursor="hand2",
        )
        options.update(kwargs)
        return tk.Button(parent, **options)

    def _build_mode_bar(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent, style="Root.TFrame")
        row.pack(fill="x", pady=(0, 10))
        row.columnconfigure((0, 1), weight=1, uniform="modes")

        specs = (
            (
                MODE_WHOLE,
                "1. Вся книга сразу",
                "Сначала все фрагменты без ответа, потом анализ и конспекты.",
            ),
            (
                MODE_CHAPTERS,
                "2. По главам",
                "Каждая глава копируется уже с инструкцией её разобрать.",
            ),
        )
        for col, (key, title, hint) in enumerate(specs):
            btn = self._plain_button(
                row,
                text=f"{title}\n{hint}",
                anchor="w",
                justify="left",
                wraplength=360,
                font=("Segoe UI", 9, "bold"),
                command=lambda mode=key: self.set_mode(mode),
            )
            btn.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 8, 0))
            self.mode_buttons[key] = btn

    def _build_prompt_bar(self, parent: ttk.Frame) -> None:
        self.prompt_box = ttk.Frame(parent, style="PromptBox.TFrame", padding=10)
        self.prompt_box.pack(fill="x", pady=(0, 12))
        self.prompt_title = ttk.Label(
            self.prompt_box,
            text="Промпты для модели",
            style="PromptBox.TLabel",
            font=("Segoe UI", 10, "bold"),
        )
        self.prompt_title.pack(anchor="w")
        self.prompt_hint = ttk.Label(self.prompt_box, style="PromptMuted.TLabel")
        self.prompt_hint.pack(anchor="w", pady=(0, 8))

        self.prompt_row = ttk.Frame(self.prompt_box, style="PromptBox.TFrame")
        self.prompt_row.pack(fill="x")

        self.continue_button = self._plain_button(
            self.prompt_box,
            text="",
            anchor="w",
            justify="left",
            wraplength=760,
            bg=self.PROMPT_BG,
            command=self.copy_continue,
        )
        self.continue_button.bind(
            "<Enter>",
            lambda _e: self._hover_prompt(self.continue_button, CONTINUE_KEY, True),
        )
        self.continue_button.bind(
            "<Leave>",
            lambda _e: self._hover_prompt(self.continue_button, CONTINUE_KEY, False),
        )

    def _build_controls(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent, style="Root.TFrame")
        controls.pack(fill="x", pady=(0, 10))

        ttk.Label(controls, text="Макс. символов:").pack(side="left")
        spin = ttk.Spinbox(
            controls,
            from_=MIN_MAX_CHARS,
            to=MAX_MAX_CHARS,
            increment=1000,
            textvariable=self.max_var,
            width=10,
        )
        spin.pack(side="left", padx=(8, 12))
        spin.bind("<Return>", lambda _e: self.resplit())

        ttk.Button(
            controls,
            text="Взять из буфера",
            style="Accent.TButton",
            command=self.load_from_clipboard,
        ).pack(side="left")
        ttk.Button(controls, text="Переразбить", command=self.resplit).pack(
            side="left", padx=(8, 0)
        )

        self.chapter_button = self._plain_button(
            controls,
            text="Разделение по главам",
            padx=10,
            pady=4,
            font=("Segoe UI", 9, "bold"),
            fg=self.ACCENT,
            command=self.toggle_chapter_mode,
        )
        self.chapter_button.pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Очистить", command=self.reset).pack(
            side="left", padx=(8, 0)
        )

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_WHOLE, MODE_CHAPTERS) or mode == self.work_mode:
            if mode == self.work_mode:
                self._refresh_mode_buttons()
            return
        self.work_mode = mode
        if mode == MODE_CHAPTERS:
            self.split_by_chapters_mode = True
        self.last_prompt = None
        self.last_continue_phrase = None
        self._apply_mode()
        if self.source_text:
            self._do_split()
        else:
            self.status_var.set(self._idle_status())

    def _apply_mode(self) -> None:
        self._refresh_mode_buttons()
        self._rebuild_prompt_buttons()
        self._refresh_chapter_button()
        if self.work_mode == MODE_CHAPTERS:
            self.title("Разделитель текста — по главам")
        else:
            self.title("Разделитель текста — вся книга сразу")

    def _refresh_mode_buttons(self) -> None:
        for key, btn in self.mode_buttons.items():
            active = key == self.work_mode
            btn.configure(
                bg=self.MODE_ON if active else self.MODE_OFF,
                fg=self.ACCENT,
                relief="solid",
            )

    def _current_prompts(self) -> tuple[dict, ...]:
        return PROMPTS_CHAPTERS if self.work_mode == MODE_CHAPTERS else PROMPTS_WHOLE

    def _rebuild_prompt_buttons(self) -> None:
        for child in self.prompt_row.winfo_children():
            child.destroy()
        self.prompt_buttons.clear()

        if self.work_mode == MODE_CHAPTERS:
            self.prompt_hint.configure(
                text="Сначала «Старт по главам» → затем главы по очереди. "
                "Инструкция к разбору уже дописывается к каждой главе. В конце — «Итог книги»."
            )
        else:
            self.prompt_hint.configure(
                text="Сначала 1 → затем фрагменты книги → затем 2 → затем 3. "
                "Кнопка «Дальше» копирует короткие команды по кругу."
            )

        prompts = self._current_prompts()
        cols = tuple(range(len(prompts)))
        self.prompt_row.columnconfigure(cols, weight=1, uniform="prompts")
        wrap = 360 if len(prompts) == 2 else 210
        for col, prompt in enumerate(prompts):
            btn = self._plain_button(
                self.prompt_row,
                text=f"{prompt['title']}\n{prompt['hint']}",
                anchor="w",
                justify="left",
                wraplength=wrap,
                bg=self.PROMPT_BG,
                command=lambda key=prompt["key"]: self.copy_prompt(key),
            )
            btn.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 6, 0))
            btn.bind(
                "<Enter>",
                lambda _e, b=btn, k=prompt["key"]: self._hover_prompt(b, k, True),
            )
            btn.bind(
                "<Leave>",
                lambda _e, b=btn, k=prompt["key"]: self._hover_prompt(b, k, False),
            )
            self.prompt_buttons[prompt["key"]] = btn

        if self.work_mode == MODE_WHOLE:
            self.continue_button.pack(fill="x", pady=(6, 0))
        else:
            self.continue_button.pack_forget()
        self._refresh_prompt_buttons()

    def _continue_label(self) -> str:
        total = len(CONTINUE_PHRASES)
        nxt = CONTINUE_PHRASES[self.continue_index]
        number = self.continue_index + 1
        if self.last_prompt == CONTINUE_KEY and self.last_continue_phrase:
            return (
                f"✓  4. Дальше — скопировано «{self.last_continue_phrase}»  ({number}/{total})\n"
                f"Следующее нажатие: «{nxt}»"
            )
        return f"4. Дальше  ({number}/{total})\nСледующее нажатие: «{nxt}»"

    def _hover_prompt(self, button: tk.Button | None, key: str, entering: bool) -> None:
        if button is None or self.last_prompt == key:
            return
        button.configure(bg="#dcefe1" if entering else self.PROMPT_BG)

    def _write_clipboard(self, text: str) -> bool:
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update()
            return True
        except tk.TclError:
            messagebox.showerror("Ошибка", "Не удалось записать текст в буфер обмена.")
            return False

    def copy_prompt(self, key: str) -> None:
        prompt = next(
            (item for item in self._current_prompts() if item["key"] == key), None
        )
        if prompt is None:
            return
        if not self._write_clipboard(prompt["text"]):
            return
        self.last_prompt = key
        self.last_copied = None
        self.last_continue_phrase = None
        self._refresh_prompt_buttons()
        self._refresh_part_buttons()
        self.status_var.set(
            f"Скопирован промпт «{prompt['title']}». Вставьте его в чат."
        )

    def copy_continue(self) -> None:
        phrase = CONTINUE_PHRASES[self.continue_index]
        if not self._write_clipboard(phrase):
            return
        self.last_continue_phrase = phrase
        self.continue_index = (self.continue_index + 1) % len(CONTINUE_PHRASES)
        self.last_prompt = CONTINUE_KEY
        self.last_copied = None
        self._refresh_prompt_buttons()
        self._refresh_part_buttons()
        nxt = CONTINUE_PHRASES[self.continue_index]
        self.status_var.set(
            f"Скопировано «{phrase}». Следующее нажатие «4. Дальше» даст «{nxt}»."
        )

    def _refresh_prompt_buttons(self) -> None:
        for prompt in self._current_prompts():
            btn = self.prompt_buttons.get(prompt["key"])
            if btn is None:
                continue
            copied = self.last_prompt == prompt["key"]
            mark = "✓  " if copied else ""
            btn.configure(
                text=f"{mark}{prompt['title']}\n{prompt['hint']}",
                bg=self.PROMPT_OK if copied else self.PROMPT_BG,
                fg=self.ACCENT if copied else "#1a1814",
            )
        if self.continue_button is not None and self.work_mode == MODE_WHOLE:
            copied = self.last_prompt == CONTINUE_KEY
            self.continue_button.configure(
                text=self._continue_label(),
                bg=self.PROMPT_OK if copied else self.PROMPT_BG,
                fg=self.ACCENT if copied else "#1a1814",
            )

    def _refresh_chapter_button(self) -> None:
        if self.chapter_button is None:
            return
        if self.work_mode == MODE_CHAPTERS:
            self.chapter_button.configure(
                text="Разделение по главам · всегда",
                bg=self.MODE_ON,
                fg=self.ACCENT,
                state="disabled",
            )
            return
        self.chapter_button.configure(state="normal")
        if self.split_by_chapters_mode:
            self.chapter_button.configure(
                text="Разделение по главам · вкл", bg=self.MODE_ON, fg=self.ACCENT
            )
        else:
            self.chapter_button.configure(
                text="Разделение по главам", bg=self.CARD, fg=self.ACCENT
            )

    def toggle_chapter_mode(self) -> None:
        if self.work_mode == MODE_CHAPTERS:
            return
        self.split_by_chapters_mode = not self.split_by_chapters_mode
        self._refresh_chapter_button()
        if self.source_text:
            self._do_split()
            return
        if self.split_by_chapters_mode:
            self.status_var.set(
                "В режиме «вся книга» включена нарезка по главам. Возьмите текст из буфера."
            )
        else:
            self.status_var.set(
                "Обычная нарезка по лимиту символов. Возьмите текст из буфера."
            )

    def _read_max_chars(self) -> int | None:
        raw = self.max_var.get().strip().replace(" ", "").replace("_", "")
        try:
            value = int(raw)
        except ValueError:
            messagebox.showerror(
                "Ошибка", "Введите целое число в поле «Макс. символов»."
            )
            return None
        if value < MIN_MAX_CHARS:
            messagebox.showwarning(
                "Слишком маленький лимит",
                f"Минимум — {format_int(MIN_MAX_CHARS)} символов.",
            )
            return None
        if value > MAX_MAX_CHARS:
            messagebox.showwarning(
                "Слишком большой лимит",
                f"Максимум — {format_int(MAX_MAX_CHARS)} символов.",
            )
            return None
        return value

    def load_from_clipboard(self) -> None:
        try:
            text = self.clipboard_get()
        except tk.TclError:
            messagebox.showwarning("Буфер пуст", "В буфере обмена нет текста.")
            return
        if not isinstance(text, str) or not text.strip():
            messagebox.showwarning("Буфер пуст", "В буфере обмена нет текста.")
            return
        self.source_text = text
        self._do_split()

    def resplit(self) -> None:
        if not self.source_text:
            self.load_from_clipboard()
            return
        self._do_split()

    def _use_chapter_split(self) -> bool:
        return self.work_mode == MODE_CHAPTERS or self.split_by_chapters_mode

    def _do_split(self) -> None:
        max_chars = self._read_max_chars()
        if max_chars is None:
            return

        normalized = self.source_text.replace("\r\n", "\n").replace("\r", "\n")
        if self._use_chapter_split():
            starts = find_chapter_starts(normalized)
            self.chapters_found = len(starts)
            self.chunks = split_by_chapters(self.source_text, max_chars)
            if not starts:
                messagebox.showinfo(
                    "Главы не найдены",
                    "Не удалось найти заголовки вроде «Глава 1», «Введение», «Заключение». "
                    "Текст разрезан по обычному лимиту символов.",
                )
        else:
            self.chapters_found = 0
            self.chunks = split_by_limit(self.source_text, max_chars)

        self.copied_parts.clear()
        self.last_copied = None
        self._render_parts()
        self._set_split_status(max_chars)

    def _set_split_status(self, max_chars: int) -> None:
        total = len(self.source_text)
        parts = len(self.chunks)
        if self.work_mode == MODE_CHAPTERS and self.chapters_found:
            self.status_var.set(
                f"Режим «по главам»: {self.chapters_found} "
                f"{self._plural_chapters(self.chapters_found)} → {parts} "
                f"{self._plural_parts(parts)}. К каждой главе допишется задание на разбор. "
                f"Лимит внутри главы: {format_int(max_chars)}."
            )
        elif self._use_chapter_split() and self.chapters_found:
            self.status_var.set(
                f"Нарезка по главам без задания в тексте: {self.chapters_found} "
                f"{self._plural_chapters(self.chapters_found)} → {parts} "
                f"{self._plural_parts(parts)}. Лимит: {format_int(max_chars)}."
            )
        elif self._use_chapter_split():
            self.status_var.set(
                f"Заголовки глав не найдены. Обычная нарезка: {format_int(total)} символов → "
                f"{parts} {self._plural_parts(parts)}."
            )
        else:
            self.status_var.set(
                f"Исходный текст: {format_int(total)} символов → {parts} "
                f"{self._plural_parts(parts)}. Лимит: {format_int(max_chars)}."
            )

    @staticmethod
    def _plural_parts(n: int) -> str:
        n10, n100 = n % 10, n % 100
        if n10 == 1 and n100 != 11:
            return "часть"
        if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
            return "части"
        return "частей"

    @staticmethod
    def _plural_chapters(n: int) -> str:
        n10, n100 = n % 10, n % 100
        if n10 == 1 and n100 != 11:
            return "глава"
        if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
            return "главы"
        return "глав"

    def _payload(self, index: int) -> str:
        return wrap_fragment(
            index + 1,
            len(self.chunks),
            self.chunks[index],
            with_instruction=(self.work_mode == MODE_CHAPTERS),
        )

    def _part_label(self, index: int) -> str:
        chunk = self.chunks[index]
        preview = make_preview(chunk.text)
        total = len(self.chunks)
        payload_len = len(self._payload(index))
        copied = index in self.copied_parts
        head = f"Фрагмент {index + 1}/{total}"
        if chunk.title:
            head = f"{head}  ·  {chunk.title}"
        if self.work_mode == MODE_CHAPTERS:
            if chunk.kind == "preamble":
                head += "  ·  без полного разбора"
            elif chunk.kind == "chapter_part" and chunk.part_index < chunk.part_total:
                head += "  ·  ждать продолжение"
            else:
                head += "  ·  с заданием на разбор"
        mark = "✓  " if copied else ""
        state = "скопирован     " if copied else ""
        return f"{mark}{head}     {state}{format_int(payload_len)} символов\n{preview}"

    def _render_parts(self) -> None:
        self.scroll.clear()
        self.part_buttons.clear()
        if not self.chunks:
            ttk.Label(
                self.scroll.inner, text="Нечего показывать.", style="Muted.TLabel"
            ).pack(anchor="w")
            return
        for i in range(len(self.chunks)):
            btn = self._plain_button(
                self.scroll.inner,
                text=self._part_label(i),
                anchor="w",
                justify="left",
                wraplength=740,
                font=("Segoe UI", 10),
                command=lambda idx=i: self.copy_part(idx),
            )
            btn.pack(fill="x", pady=5, padx=2)
            btn.bind("<Enter>", lambda _e, b=btn: self._hover_part(b, True))
            btn.bind("<Leave>", lambda _e, b=btn: self._hover_part(b, False))
            self.part_buttons.append(btn)
        self._refresh_part_buttons()

    def _refresh_part_buttons(self) -> None:
        for i, btn in enumerate(self.part_buttons):
            copied = i in self.copied_parts
            last = i == self.last_copied
            btn.configure(
                text=self._part_label(i),
                bg="#b7e4c7" if last else (self.DONE if copied else self.CARD),
                fg=self.ACCENT if copied else "#1a1814",
            )

    def _hover_part(self, button: tk.Button, entering: bool) -> None:
        idx = self.part_buttons.index(button) if button in self.part_buttons else None
        if idx is None or idx in self.copied_parts:
            return
        button.configure(bg="#eef6f0" if entering else self.CARD)

    def copy_part(self, index: int) -> None:
        if not (0 <= index < len(self.chunks)):
            return
        payload = self._payload(index)
        if not self._write_clipboard(payload):
            return
        self.last_copied = index
        self.copied_parts.add(index)
        self.last_prompt = None
        self.last_continue_phrase = None
        self._refresh_prompt_buttons()
        self._refresh_part_buttons()

        chunk = self.chunks[index]
        extra = f" — {chunk.title}" if chunk.title else ""
        done = len(self.copied_parts)
        total = len(self.chunks)
        if self.work_mode == MODE_CHAPTERS:
            self.status_var.set(
                f"Скопирован фрагмент {index + 1}/{total}{extra} "
                f"вместе с заданием на разбор ({format_int(len(payload))} символов). "
                f"Отправлено: {done}/{total}."
            )
        else:
            self.status_var.set(
                f"Скопирован фрагмент {index + 1}/{total}{extra} "
                f"({format_int(len(payload))} символов). Отправлено: {done}/{total}."
            )

    def reset(self) -> None:
        self.source_text = ""
        self.chunks = []
        self.copied_parts.clear()
        self.last_copied = None
        self.last_prompt = None
        self.last_continue_phrase = None
        self.chapters_found = 0
        self.scroll.clear()
        self.part_buttons.clear()
        self._refresh_prompt_buttons()
        self.status_var.set(self._idle_status())


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
