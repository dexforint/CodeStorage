#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Разделитель текста из буфера обмена.

Берёт текст из clipboard, режет на части не длиннее N символов
(по умолчанию 100 000), стараясь резать по границам глав и абзацев,
никогда — посередине слова. Каждая часть копируется обратно в буфер
по нажатию своей кнопки.

Дополнительно есть кнопки с готовыми промптами
и циклическая кнопка «Дальше».
"""

from __future__ import annotations

import re
import sys
import tkinter as tk
from tkinter import ttk, messagebox

DEFAULT_MAX_CHARS = 100_000
MIN_MAX_CHARS = 500
MAX_MAX_CHARS = 500_000

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
    "В конце каждой главы (сообщения) дай проверяющие вопросы с ответами (если они есть)."
)

PROMPTS = (
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

CONTINUE_KEY = "continue"
CONTINUE_PHRASES = (
    "Хорошо, работай",
    "Двигайся дальше",
    "Работай дальше",
    "Переходи дальше",
    "Переходи к следующей главе",
    "Продолжай",
    "Следующая глава",
    "Идём дальше",
    "К следующей главе",
    "Продолжай конспект",
    "Не останавливайся",
    "Бери следующую главу",
    "Продолжай в том же формате",
)

# Заголовки глав (рус. + запасные англ. варианты)
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

BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")
NEWLINE_RE = re.compile(r"\n")
SENTENCE_RE = re.compile(r"[.!?…][»\"”)\]]?[ \t]+")
WHITESPACE_RE = re.compile(r"\s+")


def format_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def make_preview(text: str, limit: int = 90) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def wrap_fragment(index: int, total: int, chunk: str) -> str:
    """Оборачивает фрагмент в шаблон для вставки в чат. index — с единицы."""
    return f"Фрагмент {index}/{total}\n```\n{chunk}\n```"


def _last_good_cut(
    window: str, start: int, pattern: re.Pattern, use_start: bool
) -> int | None:
    """Последняя подходящая точка разреза внутри окна. Возвращает индекс в исходном тексте."""
    last = None
    min_keep = min(80, max(1, len(window) // 6))
    for match in pattern.finditer(window):
        pos = start + (match.start() if use_start else match.end())
        offset = pos - start
        if min_keep <= offset <= len(window):
            last = pos
    return last


def find_cut(text: str, start: int, limit: int) -> int:
    """
    Ищем лучшую границу разреза в (start; start+limit].
    Приоритет: глава → пустая строка → перевод строки → конец предложения → пробел.
    Посередине слова не режем.
    """
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


class ScrollFrame(ttk.Frame):
    """Вертикально прокручиваемая область."""

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

        self.inner.bind("<Enter>", self._bind_wheel)
        self.inner.bind("<Leave>", self._unbind_wheel)
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

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
    ACCENT_HOVER = "#2d6a4f"
    MUTED = "#5c574c"
    LINE = "#d9d2c5"
    OK = "#40916c"
    PROMPT_BG = "#e8f0ea"
    PROMPT_OK = "#d8f3dc"

    def __init__(self) -> None:
        super().__init__()
        self.title("Разделитель текста")
        self.minsize(620, 600)
        self.geometry("760x760")
        self.configure(bg=self.BG)

        self._enable_dpi_awareness()

        self.source_text = ""
        self.chunks: list[str] = []
        self.part_buttons: list[tk.Button] = []
        self.prompt_buttons: dict[str, tk.Button] = {}
        self.continue_button: tk.Button | None = None
        self.last_copied: int | None = None
        self.last_prompt: str | None = None
        self.continue_index = 0
        self.last_continue_phrase: str | None = None

        self.max_var = tk.StringVar(value=str(DEFAULT_MAX_CHARS))
        self.status_var = tk.StringVar(
            value="Скопируйте текст и нажмите «Взять из буфера»."
        )

        self._build_style()
        self._build_ui()

    @staticmethod
    def _enable_dpi_awareness() -> None:
        if sys.platform == "win32":
            try:
                from ctypes import windll

                windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Root.TFrame", background=self.BG)
        style.configure("Card.TFrame", background=self.BG)
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
        style.configure("TEntry", padding=4)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, style="Root.TFrame", padding=16)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="Разделитель текста", style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            root,
            text="Режет длинный текст на части для вставки в сервисы с лимитом сообщения. "
            "Границы — главы, абзацы, предложения; слово пополам не режется.",
            style="Muted.TLabel",
            wraplength=720,
        ).pack(anchor="w", pady=(4, 12))

        self._build_prompt_bar(root)

        controls = ttk.Frame(root, style="Root.TFrame")
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
        ttk.Button(controls, text="Очистить", command=self.reset).pack(
            side="left", padx=(8, 0)
        )

        self.status = ttk.Label(
            root, textvariable=self.status_var, style="Status.TLabel"
        )
        self.status.pack(anchor="w", pady=(0, 8))

        self.scroll = ScrollFrame(root)
        self.scroll.pack(fill="both", expand=True)

        ttk.Label(
            root,
            text="Нажмите на часть — она скопируется в буфер обмена.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(10, 0))

    def _make_prompt_button(self, parent: tk.Misc, **kwargs) -> tk.Button:
        return tk.Button(
            parent,
            anchor="w",
            justify="left",
            padx=12,
            pady=8,
            font=("Segoe UI", 9),
            bg=self.PROMPT_BG,
            fg="#1a1814",
            activebackground="#d8f3dc",
            activeforeground="#1a1814",
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            cursor="hand2",
            **kwargs,
        )

    def _build_prompt_bar(self, parent: ttk.Frame) -> None:
        box = ttk.Frame(parent, style="PromptBox.TFrame", padding=10)
        box.pack(fill="x", pady=(0, 12))

        ttk.Label(
            box,
            text="Промпты для модели",
            style="PromptBox.TLabel",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            box,
            text="Сначала 1 → затем фрагменты книги → затем 2 → затем 3. "
            "Кнопка «4. Дальше» копирует короткие команды по кругу.",
            style="PromptMuted.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        row = ttk.Frame(box, style="PromptBox.TFrame")
        row.pack(fill="x")
        row.columnconfigure((0, 1, 2), weight=1, uniform="prompts")

        for col, prompt in enumerate(PROMPTS):
            btn = self._make_prompt_button(
                row,
                text=f"{prompt['title']}\n{prompt['hint']}",
                wraplength=210,
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

        self.continue_button = self._make_prompt_button(
            box,
            text=self._continue_label(),
            wraplength=700,
            command=self.copy_continue,
        )
        self.continue_button.pack(fill="x", pady=(6, 0))
        self.continue_button.bind(
            "<Enter>",
            lambda _e: self._hover_prompt(self.continue_button, CONTINUE_KEY, True),
        )
        self.continue_button.bind(
            "<Leave>",
            lambda _e: self._hover_prompt(self.continue_button, CONTINUE_KEY, False),
        )

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
        prompt = next((item for item in PROMPTS if item["key"] == key), None)
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
        for prompt in PROMPTS:
            btn = self.prompt_buttons[prompt["key"]]
            copied = self.last_prompt == prompt["key"]
            mark = "✓  " if copied else ""
            btn.configure(
                text=f"{mark}{prompt['title']}\n{prompt['hint']}",
                bg=self.PROMPT_OK if copied else self.PROMPT_BG,
                fg=self.ACCENT if copied else "#1a1814",
            )
        if self.continue_button is not None:
            copied = self.last_prompt == CONTINUE_KEY
            self.continue_button.configure(
                text=self._continue_label(),
                bg=self.PROMPT_OK if copied else self.PROMPT_BG,
                fg=self.ACCENT if copied else "#1a1814",
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

    def _do_split(self) -> None:
        max_chars = self._read_max_chars()
        if max_chars is None:
            return

        self.chunks = split_text(self.source_text, max_chars)
        self.last_copied = None
        self._render_parts()

        total = len(self.source_text)
        parts = len(self.chunks)
        self.status_var.set(
            f"Исходный текст: {format_int(total)} символов  →  {parts} "
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

    def _payload(self, index: int) -> str:
        return wrap_fragment(index + 1, len(self.chunks), self.chunks[index])

    def _part_label(self, index: int, copied: bool = False) -> str:
        chunk = self.chunks[index]
        preview = make_preview(chunk)
        total = len(self.chunks)
        payload_len = len(self._payload(index))
        if copied:
            return (
                f"✓  Фрагмент {index + 1}/{total} скопирован     "
                f"{format_int(payload_len)} символов\n{preview}"
            )
        return (
            f"Фрагмент {index + 1}/{total}     "
            f"{format_int(payload_len)} символов\n{preview}"
        )

    def _render_parts(self) -> None:
        self.scroll.clear()
        self.part_buttons.clear()

        if not self.chunks:
            ttk.Label(
                self.scroll.inner, text="Нечего показывать.", style="Muted.TLabel"
            ).pack(anchor="w")
            return

        for i in range(len(self.chunks)):
            btn = tk.Button(
                self.scroll.inner,
                text=self._part_label(i),
                anchor="w",
                justify="left",
                padx=14,
                pady=10,
                wraplength=660,
                font=("Segoe UI", 10),
                bg=self.CARD,
                fg="#1a1814",
                activebackground="#e9f5ee",
                activeforeground="#1a1814",
                relief="solid",
                borderwidth=1,
                highlightthickness=0,
                cursor="hand2",
                command=lambda idx=i: self.copy_part(idx),
            )
            btn.pack(fill="x", pady=5, padx=2)
            btn.bind("<Enter>", lambda _e, b=btn: self._hover(b, True))
            btn.bind("<Leave>", lambda _e, b=btn: self._hover(b, False))
            self.part_buttons.append(btn)

    def _refresh_part_buttons(self) -> None:
        for i, btn in enumerate(self.part_buttons):
            copied = i == self.last_copied
            btn.configure(
                text=self._part_label(i, copied=copied),
                bg="#d8f3dc" if copied else self.CARD,
                fg=self.ACCENT if copied else "#1a1814",
            )

    def _hover(self, button: tk.Button, entering: bool) -> None:
        idx = self.part_buttons.index(button) if button in self.part_buttons else None
        if idx is not None and idx == self.last_copied:
            return
        button.configure(bg="#eef6f0" if entering else self.CARD)

    def copy_part(self, index: int) -> None:
        if not (0 <= index < len(self.chunks)):
            return

        payload = self._payload(index)
        if not self._write_clipboard(payload):
            return

        self.last_copied = index
        self.last_prompt = None
        self.last_continue_phrase = None
        self._refresh_prompt_buttons()
        self._refresh_part_buttons()

        total = len(self.chunks)
        self.status_var.set(
            f"Скопирован фрагмент {index + 1}/{total} "
            f"({format_int(len(payload))} символов вместе с обёрткой). "
            "Вставьте его и возьмите следующий."
        )

    def reset(self) -> None:
        self.source_text = ""
        self.chunks = []
        self.last_copied = None
        self.last_prompt = None
        self.last_continue_phrase = None
        self.scroll.clear()
        self.part_buttons.clear()
        self._refresh_prompt_buttons()
        self.status_var.set("Скопируйте текст и нажмите «Взять из буфера».")


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
