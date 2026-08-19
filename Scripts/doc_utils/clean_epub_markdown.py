#!/usr/bin/env python3
"""
Причёсывает Markdown, извлечённый из EPUB и других HTML-источников.

Примеры:
    python groom_markdown.py book.md
    python groom_markdown.py book.md -o book.clean.md
    python groom_markdown.py ./data --in-place
    python groom_markdown.py ./data --in-place --keep-images
    python groom_markdown.py book.md --stdout
"""

from __future__ import annotations

import argparse
import html
import logging
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

CODE_FENCE_RE = re.compile(r"^(?P<fence>`{3,}|~{3,})")
HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*$")
ATX_ONLY_RE = re.compile(r"^#{1,6}\s*$")
SETEXT_UNDERLINE_RE = re.compile(r"^(?:=+|-+)\s*$")
HR_RE = re.compile(r"^(\*{3,}|-{3,}|_{3,})\s*$")
OL_RE = re.compile(r"^\d+[.)]\s+")
UL_RE = re.compile(r"^[*+\-]\s+")
BQ_RE = re.compile(r"^>\s?")
TABLE_ROW_RE = re.compile(r"^\|")
IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
EMPTY_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
REF_DEF_RE = re.compile(r"^\[([^\]]+)\]:\s+\S+")
HTML_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
HTML_TAG_RE = re.compile(
    r"</?(?:span|div|section|article|p|em|strong|i|b|sup|sub|a|ul|ol|li|h[1-6])"
    r"(?:\s+[^>]*)?>",
    re.IGNORECASE,
)
SOFT_HYPHEN_RE = re.compile(r"[\u00ad\u200b\u200c\u200d\ufeff]")
MULTI_SPACE_RE = re.compile(r"[^\S\n]{2,}")
HTML2TEXT_ESCAPES_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>])")


@dataclass
class GroomConfig:
    unwrap_paragraphs: bool = True
    unwrap_blockquotes: bool = True
    max_blank_lines: int = 2
    strip_empty_alt_images: bool = True
    unescape_html2text: bool = True
    strip_simple_html: bool = True
    collapse_inline_whitespace: bool = True
    heading_spacing: bool = True
    ensure_final_newline: bool = True


@dataclass
class GroomStats:
    unwrapped: int = 0
    images_removed: int = 0
    blank_lines_collapsed: int = 0
    html_replaced: int = 0

    def as_text(self) -> str:
        return (
            f"склеено строк: {self.unwrapped}, "
            f"удалено картинок: {self.images_removed}, "
            f"схлопнуто пустых строк: {self.blank_lines_collapsed}, "
            f"HTML-замен: {self.html_replaced}"
        )


def groom_markdown(text: str, config: GroomConfig | None = None) -> str:
    """Вернуть причёсанный Markdown."""
    config = config or GroomConfig()
    stats = GroomStats()
    return _groom(text, config, stats)


def groom_markdown_with_stats(
    text: str, config: GroomConfig | None = None
) -> tuple[str, GroomStats]:
    config = config or GroomConfig()
    stats = GroomStats()
    return _groom(text, config, stats), stats


def groom_file(
    src: Path,
    dst: Path | None = None,
    config: GroomConfig | None = None,
) -> tuple[Path, GroomStats]:
    src = Path(src)
    dst = Path(dst) if dst else src
    text = src.read_text(encoding="utf-8")
    cleaned, stats = groom_markdown_with_stats(text, config)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(cleaned, encoding="utf-8")
    return dst, stats


def _groom(text: str, config: GroomConfig, stats: GroomStats) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = SOFT_HYPHEN_RE.sub("", text)
    text = html.unescape(text)

    if config.strip_simple_html:
        text, n_br = HTML_BR_RE.subn("\n", text)
        text, n_tags = HTML_TAG_RE.subn("", text)
        stats.html_replaced += n_br + n_tags

    chunks = _split_code_fences(text)
    out: list[str] = []
    for kind, chunk in chunks:
        if kind == "code":
            out.append(chunk)
        else:
            out.append(_groom_prose(chunk, config, stats))
    text = "".join(out)

    text = _normalize_blank_lines(text, config.max_blank_lines, stats)
    text = text.strip("\n")
    if config.ensure_final_newline:
        text += "\n"
    return text


def _split_code_fences(text: str) -> list[tuple[str, str]]:
    """Режет текст на prose/code, не ломая содержимое фернсов."""
    lines = text.split("\n")
    chunks: list[tuple[str, str]] = []
    buf: list[str] = []
    fence: str | None = None
    kind = "prose"

    def flush(next_kind: str) -> None:
        nonlocal buf, kind
        if buf:
            data = "\n".join(buf)
            if kind == "prose":
                data += "\n"
            chunks.append((kind, data if kind == "prose" else "\n".join(buf) + "\n"))
        buf = []
        kind = next_kind

    i = 0
    while i < len(lines):
        line = lines[i]
        m = CODE_FENCE_RE.match(line.strip())
        if fence is None and m:
            flush("code")
            fence = m.group("fence")[0] * len(m.group("fence"))
            buf.append(line)
        elif fence is not None and line.strip().startswith(fence):
            buf.append(line)
            chunks.append(("code", "\n".join(buf) + "\n"))
            buf = []
            fence = None
            kind = "prose"
        else:
            buf.append(line)
        i += 1

    if buf:
        chunks.append((kind if fence is None else "code", "\n".join(buf)))
    return chunks or [("prose", text)]


def _groom_prose(text: str, config: GroomConfig, stats: GroomStats) -> str:
    if config.unescape_html2text:
        text = _unescape_html2text(text)

    if config.strip_empty_alt_images:
        text, removed = _strip_useless_images(text)
        stats.images_removed += removed

    lines = [ln.rstrip() for ln in text.split("\n")]
    if config.unwrap_paragraphs:
        lines, n = _unwrap_lines(lines, config)
        stats.unwrapped += n

    if config.heading_spacing:
        lines = _ensure_heading_spacing(lines)

    text = "\n".join(lines)

    if config.collapse_inline_whitespace:
        text = _collapse_inline_spaces_keep_indent(text)

    return text


def _unescape_html2text(text: str) -> str:
    """
    html2text часто экранирует то, что в MD экранировать не нужно:
    3\\. сноска, \\- в тексте, \\_ внутри слова.
    """

    def repl(m: re.Match[str]) -> str:
        ch = m.group(1)
        # оставляем экранирование только там, где оно реально что-то значит
        start = m.start()
        line_start = text.rfind("\n", 0, start) + 1
        prefix = text[line_start:start]
        if ch in "-+*" and prefix.strip() == "":
            return m.group(0)
        if ch == "#" and prefix.strip() == "":
            return m.group(0)
        if ch == ">":
            return m.group(0)
        return ch

    return HTML2TEXT_ESCAPES_RE.sub(repl, text)


def _strip_useless_images(text: str) -> tuple[str, int]:
    removed = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal removed
        alt = m.group("alt").strip()
        src = m.group("src").strip()
        # оставляем осмысленные картинки с alt или «нормальным» именем
        if alt:
            return m.group(0)
        stem = Path(src.split("?")[0]).stem
        if re.fullmatch(r"[a-z0-9]{8,}", stem, re.IGNORECASE):
            removed += 1
            return ""
        # пустой alt и случайное epub-имя вроде dvqg24vh2sic
        if re.fullmatch(r"[a-z0-9_-]{6,}", stem, re.IGNORECASE) and not re.search(
            r"[aeiouyаеёиоуыэюя]", stem, re.IGNORECASE
        ):
            removed += 1
            return ""
        removed += 1
        return ""

    # по умолчанию вычищаем пустые ![](...): после EPUB они почти всегда мусор
    def repl_empty(m: re.Match[str]) -> str:
        nonlocal removed
        if m.group(0).startswith("![") and m.group(0)[2] == "]":
            removed += 1
            return ""
        alt_end = m.group(0).find("]")
        alt = m.group(0)[2:alt_end]
        if not alt.strip():
            removed += 1
            return ""
        return m.group(0)

    text = IMAGE_RE.sub(repl_empty, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\*\*\s+\*\*", "", text)
    return text, removed


def _is_blank(line: str) -> bool:
    return line.strip() == ""


def _is_heading(line: str) -> bool:
    return bool(HEADING_RE.match(line) or ATX_ONLY_RE.match(line))


def _is_list_item(line: str) -> bool:
    s = line.lstrip()
    return bool(UL_RE.match(s) or OL_RE.match(s))


def _is_blockquote(line: str) -> bool:
    return bool(BQ_RE.match(line.lstrip()))


def _is_block_start(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if _is_heading(s) or HR_RE.match(s) or SETEXT_UNDERLINE_RE.match(s):
        return True
    if _is_list_item(s) or _is_blockquote(s) or TABLE_ROW_RE.match(s):
        return True
    if CODE_FENCE_RE.match(s) or REF_DEF_RE.match(s):
        return True
    if s.startswith("![") or s.startswith("[^"):
        return True
    return False


def _blockquote_body(line: str) -> str:
    return BQ_RE.sub("", line, count=1)


def _unwrap_lines(lines: list[str], config: GroomConfig) -> tuple[list[str], int]:
    if not lines:
        return lines, 0

    out: list[str] = []
    joined = 0
    i = 0

    while i < len(lines):
        line = lines[i]

        if config.unwrap_blockquotes and _is_blockquote(line):
            block, consumed, n = _unwrap_blockquote(lines, i)
            out.append(block)
            joined += n
            i += consumed
            continue

        if out and _can_join(out[-1], line):
            out[-1] = _join_lines(out[-1], line)
            joined += 1
            i += 1
            continue

        out.append(line)
        i += 1

    return out, joined


def _unwrap_blockquote(lines: list[str], start: int) -> tuple[str, int, int]:
    pieces: list[str] = []
    i = start
    joined = 0
    while i < len(lines) and (_is_blockquote(lines[i]) or lines[i].strip() == ">"):
        body = _blockquote_body(lines[i]).strip()
        if body:
            pieces.append(body)
        i += 1

    if not pieces:
        return ">", i - start, 0

    merged: list[str] = [pieces[0]]
    for piece in pieces[1:]:
        if _is_block_start(piece) and not _is_blockquote(piece):
            merged.append(piece)
        else:
            merged[-1] = _join_lines(merged[-1], piece)
            joined += 1
    text = " ".join(part for part in merged if part)
    # длинные цитаты оставляем одним блоком; абзацы внутри >\n>
    return "> " + text.strip(), i - start, joined


def _can_join(prev: str, curr: str) -> bool:
    if _is_blank(prev) or _is_blank(curr):
        return False
    if _is_block_start(curr):
        return False
    if HR_RE.match(prev.strip()) or SETEXT_UNDERLINE_RE.match(curr.strip()):
        return False
    if _is_heading(prev):
        # «## Заголовок.  \nВторая строка» из <br> внутри h2
        return not _is_block_start(curr) and not _is_heading(curr)
    if TABLE_ROW_RE.match(prev.lstrip()) or REF_DEF_RE.match(prev.lstrip()):
        return False
    if _is_blockquote(prev):
        return False
    # продолжение пункта списка без маркера
    if _is_list_item(prev):
        return not _is_block_start(curr)
    return not _is_block_start(prev)


def _join_lines(prev: str, curr: str) -> str:
    left = prev.rstrip()
    right = curr.strip()
    if left.endswith("-") and not left.endswith("--") and right and right[0].islower():
        return left[:-1] + right
    return left + " " + right


def _ensure_heading_spacing(lines: list[str]) -> list[str]:
    out: list[str] = []
    for idx, line in enumerate(lines):
        if _is_heading(line):
            if out and not _is_blank(out[-1]) and not _is_heading(out[-1]):
                out.append("")
            out.append(line)
            nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
            if nxt and not _is_blank(nxt) and not _is_heading(nxt):
                out.append("")
        else:
            out.append(line)
    return out


def _collapse_inline_spaces_keep_indent(text: str) -> str:
    result: list[str] = []
    for line in text.split("\n"):
        if _is_blank(line) or TABLE_ROW_RE.match(line.lstrip()):
            result.append(line.rstrip())
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = MULTI_SPACE_RE.sub(" ", line.strip())
        result.append((" " * indent) + body if body else "")
    return "\n".join(result)


def _normalize_blank_lines(text: str, max_blank: int, stats: GroomStats) -> str:
    max_blank = max(1, max_blank)
    parts = re.split(r"\n{2,}", text.strip("\n"))
    cleaned: list[str] = []
    raw_blocks = text.split("\n")

    out: list[str] = []
    blank = 0
    for line in raw_blocks:
        if line.strip() == "":
            blank += 1
            if blank <= max_blank:
                out.append("")
            else:
                stats.blank_lines_collapsed += 1
        else:
            blank = 0
            out.append(line)
    # глушим предупреждение линтера про неиспользуемую переменную
    _ = parts
    return "\n".join(out)


def _iter_md_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.md") if p.is_file())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Причёсывает Markdown после извлечения из EPUB."
    )
    p.add_argument("path", type=Path, help="Файл .md или каталог с .md")
    p.add_argument(
        "-o", "--output", type=Path, help="Куда писать (только для одного файла)"
    )
    p.add_argument(
        "--in-place", action="store_true", help="Перезаписать исходные файлы"
    )
    p.add_argument("--stdout", action="store_true", help="Печатать результат в stdout")
    p.add_argument("--suffix", default=".clean.md", help="Суффикс, если не --in-place")
    p.add_argument("--keep-images", action="store_true", help="Не удалять ![](...)")
    p.add_argument("--no-unwrap", action="store_true", help="Не склеивать переносы")
    p.add_argument("--max-blank-lines", type=int, default=2)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = GroomConfig(
        unwrap_paragraphs=not args.no_unwrap,
        max_blank_lines=args.max_blank_lines,
        strip_empty_alt_images=not args.keep_images,
    )

    files = _iter_md_files(args.path)
    if not files:
        logging.error("Не найдено Markdown-файлов: %s", args.path)
        return 1

    if args.output and (len(files) > 1 or args.path.is_dir()):
        logging.error("--output можно использовать только с одним файлом")
        return 1

    for src in files:
        text = src.read_text(encoding="utf-8")
        cleaned, stats = groom_markdown_with_stats(text, config)

        if args.stdout and len(files) == 1:
            sys.stdout.write(cleaned)
        elif args.in_place:
            src.write_text(cleaned, encoding="utf-8")
            logging.info("%s — %s", src, stats.as_text())
        elif args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(cleaned, encoding="utf-8")
            logging.info("%s → %s — %s", src, args.output, stats.as_text())
        else:
            if src.suffix.lower() == ".md" and args.suffix.endswith(".md"):
                dst = src.with_name(src.stem + args.suffix)
            else:
                dst = src.with_name(src.name + args.suffix)
            dst.write_text(cleaned, encoding="utf-8")
            logging.info("%s → %s — %s", src, dst, stats.as_text())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
