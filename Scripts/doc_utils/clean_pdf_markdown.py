#!/usr/bin/env python3
"""
Очистка Markdown после извлечения из PDF
(pymupdf4llm, markitdown, unstructured и т.п.).

Пример:
    python clean_pdf_markdown.py extracted.md -o clean.md
    python clean_pdf_markdown.py extracted.md --in-place
    python clean_pdf_markdown.py extracted.md --dry-run
    python clean_pdf_markdown.py --self-test
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------


@dataclass
class Stats:
    changed: list[str] = field(default_factory=list)

    def add(self, name: str, before: str, after: str) -> str:
        if before != after:
            self.changed.append(name)
        return after

    def report(self) -> str:
        if not self.changed:
            return "Изменений нет."
        return "Сработали шаги:\n" + "\n".join(f"  • {n}" for n in self.changed)


# ---------------------------------------------------------------------------
# Защита фрагментов, которые трогать нельзя
# ---------------------------------------------------------------------------

_PROTECT_RE = re.compile(r"(```.*?```|`[^`]+`)", re.DOTALL)


def protect(text: str) -> tuple[str, list[str]]:
    chunks: list[str] = []

    def repl(m: re.Match) -> str:
        chunks.append(m.group(0))
        return f"\x00PH{len(chunks) - 1}\x00"

    return _PROTECT_RE.sub(repl, text), chunks


def restore(text: str, chunks: list[str]) -> str:
    return re.sub(r"\x00PH(\d+)\x00", lambda m: chunks[int(m.group(1))], text)


# ---------------------------------------------------------------------------
# Словари для дефисов и «отклеивания» предлогов
# ---------------------------------------------------------------------------

_LEFT_PARTICLES = (
    "из|по|кое|во|кто|что|как|где|куда|когда|зачем|почему|"
    "какой|какая|какое|какие|чем|чём|всё|все|"
    "точь|чуть|мало|давным|крест|туда|сюда|тогда|"
    "чей|чья|чьё|чьи|сколько|откуда|оттого|"
    "self|well|so|re|co|pre|non|anti|multi|semi|mini"
)

_RIGHT_PARTICLES = (
    "за|под|над|то|либо|нибудь|таки|ка|"
    "первых|вторых|третьих|"
    "настоящему|моему|твоему|своему|нашему|вашему|"
    "видимому|прежнему|новому|старому|хорошему|разному|"
    "русски|английски|немецки|французски|простому|"
    "старший|младший|старшая|младшая|"
    "майор|лейтенант|полковник|генерал|"
    "сатирик|романист|романтик|самоучка|кузнец|"
    "йоркский|йоркским|йоркских|йорке|йорка"
)

# Предлоги/местоимения, которые PDF (или прошлый прогон) приклеивает к слову.
_FUNC_WORDS = sorted(
    {
        "на",
        "от",
        "из",
        "до",
        "за",
        "по",
        "со",
        "во",
        "ко",
        "об",
        "ее",
        "её",
        "их",
        "он",
        "она",
        "они",
        "мы",
        "вы",
        "ты",
        "ей",
        "им",
        "его",
        "ему",
        "ум",
        "те",
        "то",
        "бы",
        "ли",
        "же",
    },
    key=len,
    reverse=True,
)

# Короткие, но целые слова — можно отклеивать предлог даже при коротком префиксе.
_SHORT_WORDS = {
    "все",
    "всё",
    "они",
    "она",
    "оно",
    "нам",
    "вам",
    "ему",
    "ему",
    "его",
    "ее",
    "её",
    "им",
    "их",
    "мы",
    "вы",
    "ты",
    "он",
    "не",
    "ни",
    "да",
    "одно",
    "одна",
    "один",
    "одни",
    "это",
    "эта",
    "этот",
    "эти",
    "так",
    "там",
    "тут",
    "уже",
    "еще",
    "ещё",
    "как",
    "чем",
    "тем",
    "том",
    "что",
    "кто",
    "где",
    "куда",
    "когда",
    "тогда",
    "если",
    "после",
    "перед",
    "через",
    "между",
    "среди",
    "только",
    "можно",
    "нужно",
    "надо",
    "даже",
    "также",
    "тоже",
    "сам",
    "сама",
    "сами",
    "себе",
    "себя",
    "меня",
    "тебя",
    "нас",
    "вас",
    "них",
    "ней",
    "ним",
    "граф",
    "лорд",
    "сам",
    "нет",
    "да",
    "вот",
    "уже",
}

# Известные цельные слова, которые нельзя рвать.
_DO_NOT_SPLIT = {
    "чтобы",
    "якобы",
    "будтобы",
    "неужели",
    "неужто",
    "изза",
    "както",
    "ктото",
    "чтото",
    "гдето",
    "когдато",
    "потому",
    "поэтому",
    "оттого",
    "зато",
    "притом",
    "причем",
    "причём",
    "также",
    "тоже",
    "даже",
    "уже",
    "ещё",
    "еще",
    "вообще",
    "впрочем",
    "иначе",
    "сейчас",
    "теперь",
    "победе",
    "свободе",
    "природе",
    "народе",
    "городе",
    "проводе",
    "выводе",
    "заводе",
    "доходе",
    "уходе",
    "молоде",
    "методе",
    "эпизоде",
}

# Окончание, после которого префикс выглядит как готовое слово.
_COMPLETE_ENDING_RE = re.compile(
    r"(?:"
    r"нный|нная|нное|нные|нных|нным|нного|нному|"
    r"вший|вшая|вшее|вшие|вшего|вшем|"
    r"ться|тся|лся|лась|лось|лись|"
    r"ами|ями|ого|ему|ими|ыми|ому|"
    r"ить|ать|еть|нуть|"
    r"ал|ил|ел|ол|ул|ла|ли|ло|"
    r"ет|ит|ут|ют|ат|ят|"
    r"ый|ий|ой|ая|ое|ие|ые|ую|юю|ой|"
    r"ом|ем|ах|ях|ов|ев|ёй|ью|"
    r"ть|ся|сь|ши"
    r")$",
    re.IGNORECASE,
)

_FRENCH_PARTICLES = {"де", "ди", "да", "ле", "ла", "дю", "фон", "ван", "эль", "аль"}

_SECTION_TITLES = ("Упражнение", "Итог")


# ---------------------------------------------------------------------------
# Регулярки
# ---------------------------------------------------------------------------

_PICTURE_BLOCK_RE = re.compile(
    r"<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->",
    re.DOTALL | re.IGNORECASE,
)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG_RE = re.compile(
    r"</?(?:mark|u|sup|sub|span|div|section|article|font|br|hr|a|p|em|strong)"
    r"(?:\s[^>]*)?/?>",
    re.IGNORECASE,
)

# Только HTML вокруг сноски — НЕ трогаем _ и *, это акцент слова.
_FOOTNOTE_WRAP_RE = re.compile(
    r"(?:</?(?:sup|sub|u|span)>)*\s*\[(\d+)\]\s*(?:</?(?:sup|sub|u|span)>)*",
    re.IGNORECASE,
)

_STRIKE_RE = re.compile(r"~~([^~]*)~~")

# _Эмерсо_ _н_   /   _Эмерсо_ _н_[11]   /   _Fiat lu_ _x_
_MERGE_EMPH_RE = re.compile(
    r"(?P<open>_{1,3}|\*{1,3})(?P<a>[^\n]+?)(?P=open)"
    r"[ \t]+"
    r"(?P=open)(?P<b>\w{1,4})(?P=open)"
    r"(?P<fn>\[\d+\])?"
)

# Оторванная последняя буква ТОЛЬКО перед сноской: Ксеркс а[24]
_TORN_LETTER_RE = re.compile(r"([^\W\d_]{2,})[ \t]+([а-яёa-z])(\[\d+\])")

# д ’ Арк  /  Д ' Аламбер
_ELISION_RE = re.compile(r"\b([DdLlNnMmДдЛлНнМм])\s*['’`´ʻʹʼ′]\s*")

_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([.,:;!?…])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([«„“\"(\[{])[ \t]+")
_SPACE_BEFORE_CLOSE_RE = re.compile(r"[ \t]+([»”\")\]}])")
_SPACE_AFTER_QUOTE_RE = re.compile(r"([»\"”‘’])(?=[A-Za-zА-ЯЁа-яё])")

_INITIALS_RE = re.compile(r"\b([A-ZА-ЯЁ])\s*\.\s*(?=[A-ZА-ЯЁ]\s*\.)")
_ABBR_RE = re.compile(
    r"\b("
    r"т|д|п|ч|г|в|н|э|др|пр|см|стр|гл|рис|табл|"
    r"англ|нем|фр|лат|итал|исп|рус|греч|"
    r"гг|вв"
    r")\s+\.",
    re.IGNORECASE,
)

_HYPHEN_LEFT_RE = re.compile(rf"\b({_LEFT_PARTICLES})[ \t]+-[ \t]+(?=\w)", re.I)
_HYPHEN_RIGHT_RE = re.compile(rf"(?<=\w)[ \t]+-[ \t]+({_RIGHT_PARTICLES})\b", re.I)
_HYPHEN_NUM_RE = re.compile(r"(\d+)\s*-\s+(?=\w)")
_HYPHEN_CAPS_RE = re.compile(
    r"\b([A-ZА-ЯЁ][A-Za-zА-ЯЁа-яё]+)\s+-\s+([A-ZА-ЯЁ][A-Za-zА-ЯЁа-яё]+)\b"
)
_HYPHEN_SAME_RE = re.compile(r"\b([A-Za-zА-ЯЁа-яё]+)\s+-\s+([A-Za-zА-ЯЁа-яё]+)\b")

_FOOTNOTE_SPACE_RE = re.compile(r"(\[\d+\])(?=[\w«„\"А-ЯЁа-яё#*])")
_FOOTNOTE_BEFORE_HEADING_RE = re.compile(r"(\[\d+\])[ \t]*(?=#{1,6}\s)")

# _Tribune[60]  →  _Tribune_[60]
_UNCLOSED_EMPH_FN_RE = re.compile(
    r"(?<![\w_])(_{1,3}|\*{1,3})([^_\*\n]{1,80}?)(\[\d+\])"
)

# _Имя Фамилия_[10]Следующее предложение
_ATTR_BREAK_RE = re.compile(
    r"(_{1,3}[А-ЯA-Z][^_\n]{1,60}_{1,3}\s*\[\d+\])[ \t]*(?=[А-ЯA-Z«„])"
)

_HEADING_BOLD_RE = re.compile(r"^(#{1,6}[ \t]+)\*\*(.+?)\*\*\s*$", re.MULTILINE)

_CYR_LAT_RE = re.compile(r"([а-яёА-ЯЁ])([A-Z])")
_ROMAN_RE = re.compile(r"([а-яёА-ЯЁ])([IVX]{1,6})\b")

_SEPARATOR_RE = re.compile(r"^(?:\s*[*#•·‒–—-]\s*){3,}$", re.MULTILINE)
_BLANK_LINE_RE = re.compile(r"(?:[ \t]*_{6,}[ \t]*)+")
_MULTI_SPACE_RE = re.compile(r"(?<=\S)[ \t]{2,}")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_DASH_BULLET_RE = re.compile(r"^–\s+", re.MULTILINE)

_TOKEN_RE = re.compile(r"\S+|\s+")


# ---------------------------------------------------------------------------
# Конвейер
# ---------------------------------------------------------------------------


class PdfMarkdownCleaner:
    """Конвейер очистки Markdown, извлечённого из PDF."""

    def __init__(
        self,
        *,
        unwrap_strikethrough: bool = True,
        join_broken_paragraphs: bool = True,
        simplify_headings: bool = True,
        normalize_blanks: bool = True,
        unglue_words: bool = True,
        section_titles: tuple[str, ...] = _SECTION_TITLES,
        footnote_fmt: str = "[{n}]",
    ) -> None:
        self.unwrap_strikethrough = unwrap_strikethrough
        self.join_broken_paragraphs = join_broken_paragraphs
        self.simplify_headings = simplify_headings
        self.normalize_blanks = normalize_blanks
        self.unglue_words = unglue_words
        self.section_titles = section_titles
        self.footnote_fmt = footnote_fmt
        self.stats = Stats()

    def clean(self, text: str) -> str:
        self.stats = Stats()
        text = (
            text.replace("\ufeff", "")
            .replace("\u00a0", " ")
            .replace("\u200b", "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        text, chunks = protect(text)

        pipeline = [
            ("picture_blocks", self._picture_blocks),
            ("html_comments", self._html_comments),
            ("html_tags", self._html_tags),
            ("footnotes", self._footnotes),
            ("strikethrough", self._strikethrough),
            ("merge_emphasis", self._merge_emphasis),
            ("torn_letters", self._torn_letters),
            ("elision", self._elision),
            ("initials_abbr", self._initials_abbr),
            ("hyphens", self._hyphens),
            ("punctuation", self._punctuation),
            ("footnote_spacing", self._footnote_spacing),
            ("close_emphasis", self._close_emphasis),
            ("attribution_breaks", self._attribution_breaks),
            ("section_titles", self._section_titles),
            ("headings", self._headings),
            ("script_gaps", self._script_gaps),
            ("unglue", self._unglue),
            ("broken_paragraphs", self._broken_paragraphs),
            ("separators", self._separators),
            ("fill_blanks", self._fill_blanks),
            ("bullets", self._bullets),
            ("whitespace", self._whitespace),
        ]
        for name, fn in pipeline:
            before = text
            text = fn(text)
            self.stats.add(name, before, text)

        text = restore(text, chunks)
        return text.strip() + "\n"

    def clean_file(self, src: Path, dst: Path | None = None) -> str:
        cleaned = self.clean(src.read_text(encoding="utf-8"))
        if dst is not None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(cleaned, encoding="utf-8")
        return cleaned

    # -- шаги --------------------------------------------------------------

    def _picture_blocks(self, text: str) -> str:
        return _PICTURE_BLOCK_RE.sub("\n\n________________________________\n\n", text)

    def _html_comments(self, text: str) -> str:
        return _HTML_COMMENT_RE.sub("", text)

    def _html_tags(self, text: str) -> str:
        return _HTML_TAG_RE.sub("", text)

    def _footnotes(self, text: str) -> str:
        fmt = self.footnote_fmt

        def repl(m: re.Match) -> str:
            return fmt.format(n=m.group(1))

        prev = None
        while prev != text:
            prev = text
            text = _FOOTNOTE_WRAP_RE.sub(repl, text)
        # «слово [1] ,» → «слово[1],»
        text = re.sub(r"[ \t]+(\[\d+\])[ \t]*([.,:;!?»]?)", r"\1\2", text)
        return text

    def _strikethrough(self, text: str) -> str:
        if not self.unwrap_strikethrough:
            return text
        return _STRIKE_RE.sub(lambda m: m.group(1), text)

    def _merge_emphasis(self, text: str) -> str:
        def repl(m: re.Match) -> str:
            fn = m.group("fn") or ""
            return f"{m.group('open')}{m.group('a')}{m.group('b')}{m.group('open')}{fn}"

        prev = None
        while prev != text:
            prev = text
            text = _MERGE_EMPH_RE.sub(repl, text)
        return text

    def _torn_letters(self, text: str) -> str:
        return _TORN_LETTER_RE.sub(r"\1\2\3", text)

    def _elision(self, text: str) -> str:
        return _ELISION_RE.sub(r"\1'", text)

    def _initials_abbr(self, text: str) -> str:
        text = _INITIALS_RE.sub(r"\1. ", text)
        text = _ABBR_RE.sub(r"\1.", text)
        return text

    def _hyphens(self, text: str) -> str:
        text = _HYPHEN_LEFT_RE.sub(r"\1-", text)
        text = _HYPHEN_RIGHT_RE.sub(r"-\1", text)
        text = _HYPHEN_NUM_RE.sub(r"\1-", text)
        text = _HYPHEN_CAPS_RE.sub(r"\1-\2", text)

        def same(m: re.Match) -> str:
            a, b = m.group(1), m.group(2)
            if a.lower() == b.lower():
                return f"{a}-{b}"
            return m.group(0)

        return _HYPHEN_SAME_RE.sub(same, text)

    def _punctuation(self, text: str) -> str:
        text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text)
        text = _SPACE_BEFORE_CLOSE_RE.sub(r"\1", text)
        text = _SPACE_AFTER_QUOTE_RE.sub(r"\1 ", text)
        text = re.sub(r"\.\s+\.\s+\.", "…", text)
        return text

    def _footnote_spacing(self, text: str) -> str:
        text = _FOOTNOTE_BEFORE_HEADING_RE.sub(r"\1\n\n", text)
        text = _FOOTNOTE_SPACE_RE.sub(r"\1 ", text)
        return text

    def _close_emphasis(self, text: str) -> str:
        def repl(m: re.Match) -> str:
            open_, body, fn = m.group(1), m.group(2), m.group(3)
            if body.endswith(open_):
                return m.group(0)
            return f"{open_}{body}{open_}{fn}"

        return _UNCLOSED_EMPH_FN_RE.sub(repl, text)

    def _attribution_breaks(self, text: str) -> str:
        return _ATTR_BREAK_RE.sub(r"\1\n\n", text)

    def _section_titles(self, text: str) -> str:
        for title in self.section_titles:
            text = re.sub(
                rf"(?<!# )\*\*{re.escape(title)}\*\*",
                rf"\n\n#### {title}\n\n",
                text,
            )
        return text

    def _headings(self, text: str) -> str:
        if not self.simplify_headings:
            return text
        return _HEADING_BOLD_RE.sub(r"\1\2", text)

    def _script_gaps(self, text: str) -> str:
        text = _CYR_LAT_RE.sub(r"\1 \2", text)
        text = _ROMAN_RE.sub(r"\1 \2", text)
        return text

    def _unglue(self, text: str) -> str:
        if not self.unglue_words:
            return text
        return "".join(
            self._unglue_token(tok) if not tok.isspace() else tok
            for tok in _TOKEN_RE.findall(text)
        )

    def _unglue_token(self, token: str) -> str:
        m = re.match(r"^(.*?)([.,:;!?…»”\")\]]*)$", token, re.DOTALL)
        if not m:
            return token
        core, punct = m.group(1), m.group(2)
        if len(core) < 5:
            return token

        low = core.lower()
        if low in _DO_NOT_SPLIT:
            return token

        for func in _FUNC_WORDS:
            if len(low) <= len(func) + 2 or not low.endswith(func):
                continue
            prefix = core[: -len(func)]
            prefix_low = prefix.lower()
            if prefix_low in _DO_NOT_SPLIT or (prefix_low + func) in _DO_NOT_SPLIT:
                continue
            if not self._prefix_looks_complete(prefix, func):
                continue
            return f"{prefix} {core[-len(func):]}{punct}"
        return token

    def _prefix_looks_complete(self, prefix: str, func: str) -> bool:
        low = prefix.lower()
        if low in _SHORT_WORDS:
            return True
        if func in _FRENCH_PARTICLES and prefix[:1].isupper() and len(prefix) >= 3:
            return True
        if len(prefix) >= 4 and _COMPLETE_ENDING_RE.search(prefix):
            return True
        return False

    def _is_md_special(self, s: str) -> bool:
        t = s.lstrip()
        return (
            t.startswith("#")
            or t.startswith(">")
            or t.startswith("|")
            or t.startswith("```")
            or t.startswith("---")
            or t.startswith("***")
            or t.startswith("- ")
            or t.startswith("* ")
            or t.startswith("+ ")
            or bool(re.match(r"\d+[.)]\s", t))
            or (bool(t) and set(t.strip()) <= {"_", " ", "\t"})
        )

    def _broken_paragraphs(self, text: str) -> str:
        if not self.join_broken_paragraphs:
            return text

        lines = text.split("\n")
        out: list[str] = []
        i = 0
        n = len(lines)

        while i < n:
            cur = lines[i]
            consumed = i
            while True:
                j = consumed + 1
                blanks = 0
                while j < n and not lines[j].strip():
                    blanks += 1
                    j += 1
                if j >= n or blanks > 1:
                    break
                nxt = lines[j]
                if self._is_md_special(cur) or self._is_md_special(nxt):
                    break
                stripped = cur.rstrip()
                if not stripped:
                    break
                if stripped[-1] in '.!?:…»"”)]':
                    break
                nxt_s = nxt.lstrip()
                if not nxt_s or nxt_s[0].isupper() or nxt_s[0] in "—–-•*#":
                    break
                cur = stripped + " " + nxt_s
                consumed = j
            out.append(cur)
            i = consumed + 1
        return "\n".join(out)

    def _separators(self, text: str) -> str:
        return _SEPARATOR_RE.sub("\n---\n", text)

    def _fill_blanks(self, text: str) -> str:
        if not self.normalize_blanks:
            return text

        def repl(m: re.Match) -> str:
            return (
                "\n\n________________________________\n\n"
                if "\n" in m.group(0)
                else "________"
            )

        return _BLANK_LINE_RE.sub(repl, text)

    def _bullets(self, text: str) -> str:
        return _DASH_BULLET_RE.sub("- ", text)

    def _whitespace(self, text: str) -> str:
        text = _MULTI_SPACE_RE.sub(" ", text)
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        text = _MULTI_NL_RE.sub("\n\n", text)
        return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Причёсывает Markdown, извлечённый из PDF.")
    p.add_argument("input", nargs="?", type=Path, help="Входной .md файл")
    p.add_argument("-o", "--output", type=Path, help="Куда записать результат")
    p.add_argument(
        "-i", "--in-place", action="store_true", help="Перезаписать входной файл"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Не писать файл, только отчёт"
    )
    p.add_argument("--keep-heading-bold", action="store_true")
    p.add_argument("--keep-broken-lines", action="store_true")
    p.add_argument("--keep-blanks", action="store_true")
    p.add_argument("--no-unglue", action="store_true", help="Не разклеивать предлоги")
    p.add_argument("--md-footnotes", action="store_true", help="Сноски в виде [^1]")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.input is None:
        raw = sys.stdin.read()
        src_label = "<stdin>"
    else:
        if not args.input.exists():
            print(f"Файл не найден: {args.input}", file=sys.stderr)
            return 1
        raw = args.input.read_text(encoding="utf-8")
        src_label = str(args.input)

    cleaner = PdfMarkdownCleaner(
        join_broken_paragraphs=not args.keep_broken_lines,
        simplify_headings=not args.keep_heading_bold,
        normalize_blanks=not args.keep_blanks,
        unglue_words=not args.no_unglue,
        footnote_fmt="[^{n}]" if args.md_footnotes else "[{n}]",
    )
    cleaned = cleaner.clean(raw)

    print(f"Источник: {src_label}", file=sys.stderr)
    print(f"Было: {len(raw)} символов, стало: {len(cleaned)}", file=sys.stderr)
    print(cleaner.stats.report(), file=sys.stderr)

    if args.dry_run:
        return 0

    if args.in_place:
        if args.input is None:
            print("--in-place нельзя использовать со stdin", file=sys.stderr)
            return 1
        args.input.write_text(cleaned, encoding="utf-8")
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(cleaned, encoding="utf-8")
    else:
        sys.stdout.write(cleaned)
    return 0


if __name__ == "__main__":
    sys.exit(main())
