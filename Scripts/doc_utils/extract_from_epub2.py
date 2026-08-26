#!/usr/bin/env python3
"""
Качественное извлечение текста из EPUB / FB2 в Markdown.

Зависимости:
    pip install ebooklib html2text beautifulsoup4 lxml
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import html2text

# ---------------------------------------------------------------------------
# Общие утилиты
# ---------------------------------------------------------------------------

FB2_NS = "http://www.gribuser.ru/xml/fictionbook/2.0"
XLINK_NS = "http://www.w3.org/1999/xlink"

# теги FB2 → локальное имя без namespace
_NS_RE = re.compile(r"\{[^}]+\}")


def local_tag(el: ET.Element) -> str:
    return _NS_RE.sub("", el.tag)


def text_of(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def yaml_escape(value: str) -> str:
    """Экранирует значение для YAML front-matter."""
    if not value:
        return value
    if any(c in value for c in ":#{}[]&*!|>'\"%@`"):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


# ---------------------------------------------------------------------------
# EPUB
# ---------------------------------------------------------------------------


def clean_html(html: bytes | str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(
        ["script", "style", "nav", "header", "footer", "aside", "iframe", "noscript"]
    ):
        tag.decompose()
    for tag in soup.find_all(
        class_=re.compile(r"(nav|toc|sidebar|footer|header|pagenum|pagebreak)", re.I)
    ):
        tag.decompose()
    for tag in soup.find_all(id=re.compile(r"(nav|toc|sidebar|footer|header)", re.I)):
        tag.decompose()
    body = soup.body or soup
    return str(body)


def epub_to_markdown(epub_path: Path, ignore_images: bool = True) -> str:
    book = epub.read_epub(str(epub_path))

    def meta(name: str) -> str:
        values = book.get_metadata("DC", name)
        return values[0][0] if values else ""

    title = meta("title") or epub_path.stem
    authors = [a[0] for a in book.get_metadata("DC", "creator")]
    language = meta("language")
    publisher = meta("publisher")
    date = meta("date")

    parts: list[str] = [_build_front_matter(title, authors, language, publisher, date)]
    parts.append(f"# {title}\n")
    if authors:
        parts.append(f"**{', '.join(authors)}**\n")

    h2t = html2text.HTML2Text()
    h2t.body_width = 0
    h2t.unicode_snob = True
    h2t.ignore_links = False
    h2t.ignore_images = ignore_images
    h2t.ignore_emphasis = False
    h2t.ignore_tables = False
    h2t.bypass_tables = False
    h2t.protect_links = True
    h2t.wrap_list_items = True
    h2t.single_line_break = False
    h2t.mark_code = True

    seen: set[str] = set()
    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        if item.file_name in seen:
            continue
        seen.add(item.file_name)

        raw = item.get_content()
        if not raw or not raw.strip():
            continue

        md = h2t.handle(clean_html(raw)).strip()
        md = re.sub(r"\n{3,}", "\n\n", md)
        if md:
            parts.append(md)
            parts.append("\n\n---\n")

    if parts and parts[-1].strip() == "---":
        parts.pop()
    return "\n".join(parts).strip() + "\n"


# ---------------------------------------------------------------------------
# FB2 → Markdown
# ---------------------------------------------------------------------------


class Fb2Converter:
    """Рекурсивно обходит FB2 XML и собирает Markdown."""

    def __init__(self, root: ET.Element, ignore_images: bool = True) -> None:
        self.root = root
        self.ignore_images = ignore_images
        self.notes: dict[str, str] = {}
        self._heading_level = 1  # # заголовок книги уже выведен снаружи

    # --- метаданные --------------------------------------------------------

    def _find(self, parent: ET.Element, *path: str) -> ET.Element | None:
        cur: ET.Element | None = parent
        for name in path:
            if cur is None:
                return None
            cur = next((c for c in cur if local_tag(c) == name), None)
        return cur

    def _findall(self, parent: ET.Element, name: str) -> list[ET.Element]:
        return [c for c in parent if local_tag(c) == name]

    def metadata(self) -> dict:
        desc = self._find(self.root, "description")
        ti = self._find(desc, "title-info") if desc is not None else None
        pi = self._find(desc, "publish-info") if desc is not None else None

        title = text_of(self._find(ti, "book-title")) if ti is not None else ""
        lang = text_of(self._find(ti, "lang")) if ti is not None else ""
        date = ""
        if ti is not None:
            date_el = self._find(ti, "date")
            if date_el is not None:
                date = date_el.get("value") or text_of(date_el)
        publisher = text_of(self._find(pi, "publisher")) if pi is not None else ""

        authors: list[str] = []
        if ti is not None:
            for a in self._findall(ti, "author"):
                parts = [
                    text_of(self._find(a, n))
                    for n in ("first-name", "middle-name", "last-name")
                ]
                name = " ".join(p for p in parts if p)
                if not name:
                    name = text_of(self._find(a, "nickname"))
                if name:
                    authors.append(name)

        annotation = ""
        if ti is not None:
            ann = self._find(ti, "annotation")
            if ann is not None:
                annotation = self._block(ann, heading=2).strip()

        return {
            "title": title,
            "authors": authors,
            "lang": lang,
            "date": date,
            "publisher": publisher,
            "annotation": annotation,
        }

    # --- инлайн ------------------------------------------------------------

    def _inline(self, el: ET.Element) -> str:
        chunks: list[str] = []
        if el.text:
            chunks.append(el.text)
        for child in el:
            tag = local_tag(child)
            inner = self._inline(child)
            if tag in ("strong", "b"):
                chunks.append(f"**{inner}**" if inner else "")
            elif tag in ("emphasis", "em", "i"):
                chunks.append(f"*{inner}*" if inner else "")
            elif tag == "strikethrough":
                chunks.append(f"~~{inner}~~" if inner else "")
            elif tag == "code":
                chunks.append(f"`{inner}`" if inner else "")
            elif tag == "a":
                href = (
                    child.get(f"{{{XLINK_NS}}}href")
                    or child.get("href")
                    or child.get("l:href")
                    or ""
                )
                if href.startswith("#"):
                    # сноска / внутренняя ссылка
                    note_id = href.lstrip("#")
                    chunks.append(f"{inner}[^{note_id}]" if inner else f"[^{note_id}]")
                elif href:
                    chunks.append(f"[{inner}]({href})" if inner else href)
                else:
                    chunks.append(inner)
            elif tag == "image":
                if not self.ignore_images:
                    href = (
                        child.get(f"{{{XLINK_NS}}}href")
                        or child.get("href")
                        or child.get("l:href")
                        or ""
                    )
                    alt = child.get("alt") or child.get("title") or "image"
                    chunks.append(f"![{alt}]({href})")
            elif tag == "style":
                chunks.append(inner)
            else:
                chunks.append(inner)
            if child.tail:
                chunks.append(child.tail)
        return "".join(chunks)

    # --- блоки -------------------------------------------------------------

    def _block(self, el: ET.Element, heading: int) -> str:
        out: list[str] = []
        for child in el:
            tag = local_tag(child)
            if tag == "section":
                out.append(self._section(child, heading))
            elif tag == "title":
                # заголовок секции обрабатывается в _section
                continue
            elif tag == "subtitle":
                text = self._inline(child).strip()
                if text:
                    out.append(f"{'#' * min(heading + 1, 6)} {text}\n")
            elif tag == "p":
                text = self._inline(child).strip()
                if text:
                    out.append(text + "\n")
            elif tag == "empty-line":
                out.append("\n")
            elif tag == "epigraph":
                body = self._block(child, heading).strip()
                quoted = "\n".join(
                    f"> {line}" if line else ">" for line in body.splitlines()
                )
                if quoted:
                    out.append(quoted + "\n")
            elif tag == "cite":
                body = self._block(child, heading).strip()
                quoted = "\n".join(
                    f"> {line}" if line else ">" for line in body.splitlines()
                )
                if quoted:
                    out.append(quoted + "\n")
            elif tag == "text-author":
                author = self._inline(child).strip()
                if author:
                    out.append(f"> — *{author}*\n")
            elif tag == "poem":
                out.append(self._poem(child))
            elif tag == "stanza":
                out.append(self._stanza(child))
            elif tag == "v":
                line = self._inline(child).rstrip()
                out.append(line + "  \n")  # два пробела = перенос в MD
            elif tag == "annotation":
                body = self._block(child, heading).strip()
                if body:
                    out.append(body + "\n")
            elif tag == "table":
                out.append(self._table(child))
            elif tag == "image":
                if not self.ignore_images:
                    href = (
                        child.get(f"{{{XLINK_NS}}}href")
                        or child.get("href")
                        or child.get("l:href")
                        or ""
                    )
                    alt = child.get("alt") or child.get("title") or "image"
                    out.append(f"![{alt}]({href})\n")
            else:
                # неизвестный контейнер — спускаемся внутрь
                nested = self._block(child, heading).strip()
                if nested:
                    out.append(nested + "\n")
        text = "\n".join(out)
        return re.sub(r"\n{3,}", "\n\n", text)

    def _section(self, el: ET.Element, heading: int) -> str:
        title_el = self._find(el, "title")
        title = ""
        if title_el is not None:
            # <title> может содержать несколько <p>
            title = " ".join(
                self._inline(p).strip()
                for p in title_el
                if local_tag(p) in ("p", "subtitle") or True
            ).strip()
            if not title:
                title = self._inline(title_el).strip()

        parts: list[str] = []
        if title:
            parts.append(f"{'#' * min(heading, 6)} {title}\n")

        # остальное содержимое секции (кроме title)
        body_parts: list[str] = []
        for child in el:
            if local_tag(child) == "title":
                continue
            if local_tag(child) == "section":
                body_parts.append(self._section(child, heading + 1))
            else:
                # оборачиваем одиночный элемент во временный контейнер-логику
                fake = ET.Element("wrap")
                fake.append(child)
                body_parts.append(self._block(fake, heading + 1))
        body = "\n".join(p for p in body_parts if p.strip())
        if body:
            parts.append(body)
        return "\n".join(parts) + "\n"

    def _poem(self, el: ET.Element) -> str:
        chunks: list[str] = []
        title_el = self._find(el, "title")
        if title_el is not None:
            t = self._inline(title_el).strip()
            if t:
                chunks.append(f"**{t}**\n")
        for child in el:
            tag = local_tag(child)
            if tag == "title":
                continue
            if tag == "stanza":
                chunks.append(self._stanza(child))
            elif tag == "text-author":
                a = self._inline(child).strip()
                if a:
                    chunks.append(f"— *{a}*\n")
            elif tag == "epigraph":
                body = self._block(child, 3).strip()
                quoted = "\n".join(
                    f"> {line}" if line else ">" for line in body.splitlines()
                )
                if quoted:
                    chunks.append(quoted + "\n")
            elif tag == "date":
                d = text_of(child)
                if d:
                    chunks.append(f"*{d}*\n")
        return "\n".join(chunks) + "\n"

    def _stanza(self, el: ET.Element) -> str:
        lines: list[str] = []
        for child in el:
            if local_tag(child) == "v":
                lines.append(self._inline(child).rstrip() + "  ")
        return "\n".join(lines) + "\n"

    def _table(self, el: ET.Element) -> str:
        rows: list[list[str]] = []
        for tr in el:
            if local_tag(tr) != "tr":
                continue
            cells = [
                self._inline(td).strip().replace("|", "\\|")
                for td in tr
                if local_tag(td) in ("td", "th")
            ]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        for r in rows:
            r.extend([""] * (width - len(r)))
        header = rows[0]
        sep = ["---"] * width
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for r in rows[1:]:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines) + "\n"

    # --- сноски (body name="notes") ----------------------------------------

    def collect_notes(self) -> None:
        for body in self.root:
            if local_tag(body) != "body":
                continue
            name = (body.get("name") or "").lower()
            if name not in ("notes", "comments", "footnotes"):
                continue
            for sec in body.iter():
                if local_tag(sec) != "section":
                    continue
                sid = sec.get("id")
                if not sid:
                    continue
                # текст сноски без заголовка-номера
                fake = ET.Element("wrap")
                for child in sec:
                    if local_tag(child) == "title":
                        continue
                    fake.append(child)
                note_md = self._block(fake, 6).strip()
                note_md = re.sub(r"\s+", " ", note_md)
                if note_md:
                    self.notes[sid] = note_md

    def notes_markdown(self) -> str:
        if not self.notes:
            return ""
        lines = [f"[^{k}]: {v}" for k, v in self.notes.items()]
        return "\n\n" + "\n".join(lines) + "\n"

    # --- точка входа -------------------------------------------------------

    def convert(self) -> str:
        meta = self.metadata()
        title = meta["title"] or "Untitled"
        authors = meta["authors"]

        self.collect_notes()

        parts: list[str] = [
            _build_front_matter(
                title, authors, meta["lang"], meta["publisher"], meta["date"]
            )
        ]
        parts.append(f"# {title}\n")
        if authors:
            parts.append(f"**{', '.join(authors)}**\n")
        if meta["annotation"]:
            parts.append("## Аннотация\n")
            parts.append(meta["annotation"] + "\n")
            parts.append("---\n")

        for body in self.root:
            if local_tag(body) != "body":
                continue
            name = (body.get("name") or "").lower()
            if name in ("notes", "comments", "footnotes"):
                continue  # сноски выведем отдельно в конце
            md = self._block(body, heading=2).strip()
            if md:
                parts.append(md)
                parts.append("\n\n---\n")

        if parts and parts[-1].strip() == "---":
            parts.pop()

        parts.append(self.notes_markdown())
        result = "\n".join(parts).strip() + "\n"
        return re.sub(r"\n{3,}", "\n\n", result)


def _build_front_matter(
    title: str,
    authors: list[str],
    language: str,
    publisher: str,
    date: str,
) -> str:
    front = ["---", f"title: {yaml_escape(title)}"]
    if authors:
        front.append(f"author: {yaml_escape(', '.join(authors))}")
    if language:
        front.append(f"lang: {language}")
    if publisher:
        front.append(f"publisher: {yaml_escape(publisher)}")
    if date:
        front.append(f"date: {yaml_escape(date)}")
    front.append("---\n")
    return "\n".join(front)


def _parse_fb2_xml(data: bytes) -> ET.Element:
    # убираем объявление кодировки — ElementTree читает уже декодированные байты плохо,
    # а fromstring с XML-декларацией в utf-8/windows-1251 работает, если скормить bytes
    try:
        return ET.fromstring(data)
    except ET.ParseError:
        # часто FB2 в windows-1251
        for enc in ("utf-8", "windows-1251", "cp1251", "koi8-r", "latin-1"):
            try:
                text = data.decode(enc)
                # ElementTree не любит encoding= в декларации при строке
                text = re.sub(r"<\?xml[^?]*\?>", "", text, count=1)
                return ET.fromstring(text)
            except (UnicodeDecodeError, ET.ParseError):
                continue
        raise ValueError(
            "Не удалось разобрать FB2: неизвестная кодировка или битый XML"
        )


def fb2_to_markdown(path: Path, ignore_images: bool = True) -> str:
    data = path.read_bytes()

    # .fb2.zip / просто zip с fb2 внутри
    if path.suffix.lower() == ".zip" or data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            fb2_name = next(
                (n for n in zf.namelist() if n.lower().endswith(".fb2")), None
            )
            if fb2_name is None:
                raise ValueError(f"В архиве {path.name} нет .fb2 файла")
            data = zf.read(fb2_name)

    root = _parse_fb2_xml(data)
    return Fb2Converter(root, ignore_images=ignore_images).convert()


# ---------------------------------------------------------------------------
# Диспетчер форматов
# ---------------------------------------------------------------------------


def detect_format(path: Path) -> str:
    suffix = "".join(path.suffixes).lower()  # .fb2.zip
    if suffix.endswith(".epub"):
        return "epub"
    if suffix.endswith(".fb2") or suffix.endswith(".fb2.zip"):
        return "fb2"
    # по сигнатуре
    head = path.read_bytes()[:8]
    if head.startswith(b"PK"):
        # zip: epub или fb2.zip
        try:
            with zipfile.ZipFile(path) as zf:
                names = [n.lower() for n in zf.namelist()]
                if any(n.endswith(".opf") or n == "mimetype" for n in names):
                    return "epub"
                if any(n.endswith(".fb2") for n in names):
                    return "fb2"
        except zipfile.BadZipFile:
            pass
        return "epub"  # пусть epub-парсер даст понятную ошибку
    if head.lstrip().startswith(b"<?xml") or head.lstrip().startswith(b"<FictionBook"):
        return "fb2"
    raise ValueError(
        f"Неизвестный формат файла: {path.name}\n"
        "Поддерживаются EPUB (.epub) и FictionBook (.fb2, .fb2.zip)."
    )


def book_to_markdown(path: Path, ignore_images: bool = True) -> str:
    fmt = detect_format(path)
    if fmt == "epub":
        return epub_to_markdown(path, ignore_images=ignore_images)
    return fb2_to_markdown(path, ignore_images=ignore_images)


def main() -> None:
    raw = input("Введите путь до EPUB/FB2 файла: ").strip().strip('"')
    input_path = Path(raw)

    if not input_path.exists():
        raise SystemExit(f"Файл не найден: {input_path}")

    md = book_to_markdown(input_path, ignore_images=True)
    out = input_path.with_suffix(".md")
    # для .fb2.zip получится .fb2.md — поправим
    if "".join(input_path.suffixes).lower().endswith(".fb2.zip"):
        out = input_path.with_name(input_path.name[: -len(".fb2.zip")] + ".md")
    out.write_text(md, encoding="utf-8")
    print(f"Готово: {out}  ({len(md):,} символов)")


if __name__ == "__main__":
    main()
