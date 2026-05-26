#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from pathlib import Path
from typing import List

H2_RE = re.compile(r"^##(?!#)\s+")  # "## " but not "###"


def split_by_h2(markdown: str) -> List[str]:
    """
    Splits markdown into "sections" by level-2 headings (##),
    preserving original text exactly. Headings inside fenced code blocks
    (``` or ~~~) are ignored.
    """
    lines = markdown.splitlines(keepends=True)

    sections: List[str] = []
    current: List[str] = []

    in_fence = False
    fence_delim = None  # ``` or ~~~

    for line in lines:
        stripped = line.lstrip()

        # Toggle fenced code blocks
        if stripped.startswith("```") or stripped.startswith("~~~"):
            delim = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_delim = delim
            elif fence_delim == delim:
                in_fence = False
                fence_delim = None

        # Split on H2 headings, but only outside fences
        if not in_fence and H2_RE.match(line):
            if current:
                sections.append("".join(current))
                current = []
            current.append(line)
        else:
            current.append(line)

    if current:
        sections.append("".join(current))

    return sections


def split_to_max_chars(text: str, max_chars: int) -> List[str]:
    """
    Hard-splits text into chunks <= max_chars, preferring newline boundaries.
    Preserves text as-is (no added headers/markers).
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    chunks: List[str] = []
    buf = ""

    for line in text.splitlines(keepends=True):
        # If a single line is longer than max_chars, split it hard.
        if len(line) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            start = 0
            while start < len(line):
                chunks.append(line[start : start + max_chars])
                start += max_chars
            continue

        if len(buf) + len(line) <= max_chars:
            buf += line
        else:
            if buf:
                chunks.append(buf)
            buf = line

    if buf:
        chunks.append(buf)

    return chunks


def pack_sections_into_chunks(sections: List[str], max_chars: int) -> List[str]:
    """
    Packs consecutive sections into chunks not exceeding max_chars.
    If a single section exceeds max_chars, it is split internally.
    """
    chunks: List[str] = []
    current = ""

    for sec in sections:
        if not current:
            if len(sec) <= max_chars:
                current = sec
            else:
                chunks.extend(split_to_max_chars(sec, max_chars))
                current = ""
            continue

        if len(current) + len(sec) <= max_chars:
            current += sec
        else:
            chunks.append(current)
            if len(sec) <= max_chars:
                current = sec
            else:
                chunks.extend(split_to_max_chars(sec, max_chars))
                current = ""

    if current:
        chunks.append(current)

    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split a Markdown file into parts by H2 (##) headings, preserving Markdown markup."
    )
    ap.add_argument("input_md", help="Path to the input Markdown file")
    ap.add_argument("max_chars", type=int, help="Maximum number of characters per part")
    ap.add_argument(
        "output_dir", help="Directory where parts will be saved (created if missing)"
    )
    args = ap.parse_args()

    input_path = Path(args.input_md)
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    markdown = input_path.read_text(encoding="utf-8")
    sections = split_by_h2(markdown)
    chunks = pack_sections_into_chunks(sections, args.max_chars)

    stem = input_path.stem
    for i, chunk in enumerate(chunks, start=1):
        out_path = out_dir / f"{stem}_part_{i:03d}.md"
        out_path.write_text(chunk, encoding="utf-8")

    print(f"Done. Created {len(chunks)} file(s) in: {out_dir}")


if __name__ == "__main__":
    main()
