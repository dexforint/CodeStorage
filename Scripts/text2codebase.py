from __future__ import annotations

import argparse
import re
from pathlib import Path, PurePosixPath

FENCE_RE = re.compile(r"^(`{3,})([A-Za-z0-9_-]*)\s*$")
SEP_LINE = "---"

# Чтобы не ловить "обычный текст" как имя файла, разрешаем имена без расширения
# только из этого списка (можете расширить при необходимости).
NO_EXT_BASENAMES = {
    "Makefile",
    "Dockerfile",
    "LICENSE",
    "COPYING",
    "NOTICE",
    "README",
}


def _strip_wrapping_backticks(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "`" and s[-1] == "`":
        return s[1:-1].strip()
    return s


def _normalize_relpath_line(raw: str) -> str:
    """
    Нормализация "строки пути", пришедшей из Markdown/LLM:
    - убираем обрамляющие backticks (`path/to/file.py`)
    - фиксим кейс: **init**.py -> __init__.py (частый результат markdown-рендера __init__.py)
    - убираем ведущий маркер списка "- " (на всякий случай)
    """
    s = raw.strip()

    # на всякий случай: буллет-листы
    if s.startswith("- "):
        s = s[2:].strip()

    s = _strip_wrapping_backticks(s)

    # Нормализация слэшей
    s = s.replace("\\", "/")

    # Фикс markdown-искажения __init__.py -> **init**.py / *init*.py
    # Применяем в любом месте строки (и в подпапках).
    s = re.sub(r"\*\*init\*\*(?=\.py\b)", "__init__", s)
    s = re.sub(r"\*init\*(?=\.py\b)", "__init__", s)

    return s


def _looks_like_relpath(s: str) -> bool:
    """
    Эвристика "похоже ли это на строку пути файла".
    Нужна, чтобы не спутать внутренние markdown-фрагменты с началом следующей секции.
    """
    if not s:
        return False
    if any(ch.isspace() for ch in s):
        return False
    if "`" in s:
        return False
    if s.startswith("<CONTENT"):
        return False
    if s.startswith(SEP_LINE):
        return False
    if s.endswith("/"):
        return False

    # запрещённые на Windows символы + ':' (чтобы не ловить "Example: ..." и т.п.)
    if any(ch in s for ch in '<>:"|?*'):
        return False

    # должен быть относительный posix-путь без ..
    pp = PurePosixPath(s)
    if pp.is_absolute():
        return False
    if ".." in pp.parts:
        return False

    base = pp.name
    if "." in base:
        return True
    if base in NO_EXT_BASENAMES:
        return True
    return False


def _safe_target_path(root: Path, rel_line: str) -> Path:
    """
    Гарантирует, что результирующий путь остаётся внутри root.
    """
    rel_line = rel_line.replace("\\", "/")
    pp = PurePosixPath(rel_line)

    if not rel_line or rel_line.strip() == "":
        raise ValueError("Empty relative path")
    if pp.is_absolute():
        raise ValueError(f"Absolute path is not allowed: {rel_line}")
    if ".." in pp.parts:
        raise ValueError(f"Path traversal is not allowed: {rel_line}")

    target = root.joinpath(*pp.parts)

    root_resolved = root.resolve()
    target_resolved = target.resolve()
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as e:
        raise ValueError(f"Target path escapes root: {rel_line}") from e

    return target


def restore_project_from_text(
    text_file: Path, project_root: Path, *, overwrite: bool = True
) -> list[Path]:
    text_file = text_file.resolve()
    project_root = project_root.resolve()

    if not text_file.is_file():
        raise FileNotFoundError(f"Input file not found: {text_file}")

    project_root.mkdir(parents=True, exist_ok=True)

    # universal newlines: удобно парсить независимо от платформы
    text = text_file.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)

    def next_meaningful(idx: int) -> int:
        """Пропустить пустые строки и строки-разделители (---) между секциями."""
        while idx < len(lines):
            s = lines[idx].rstrip("\n")
            if s == "" or s == SEP_LINE:
                idx += 1
                continue
            break
        return idx

    def is_section_start_at(idx: int) -> bool:
        """Проверить, начинается ли с idx секция файла: PATH + (FENCE|OMITTED)."""
        idx = next_meaningful(idx)
        if idx >= len(lines):
            return False

        raw_path = lines[idx].rstrip("\n")
        rel = _normalize_relpath_line(raw_path)
        if not _looks_like_relpath(rel):
            return False

        j = next_meaningful(idx + 1)
        if j >= len(lines):
            return False

        nxt = lines[j].rstrip("\n")
        if nxt.startswith("<CONTENT OMITTED:"):
            return True
        return FENCE_RE.match(nxt) is not None

    # Если в начале есть "дерево проекта", оно обычно отделено строкой '---'.
    # Но '---' может встречаться и внутри файлов. Поэтому ищем такой '---',
    # после которого действительно начинается секция файла.
    start = 0
    for i, ln in enumerate(lines):
        if ln.rstrip("\n") == SEP_LINE and is_section_start_at(i + 1):
            start = i + 1
            break

    created: list[Path] = []
    i = start

    while True:
        i = next_meaningful(i)
        if i >= len(lines):
            break

        # 1) путь
        raw_rel = lines[i].rstrip("\n")
        rel = _normalize_relpath_line(raw_rel)
        i += 1

        i = next_meaningful(i)
        if i >= len(lines):
            break

        # 2) omitted или opening fence
        head = lines[i].rstrip("\n")

        target_path = _safe_target_path(project_root, rel)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if head.startswith("<CONTENT OMITTED:"):
            i += 1
            if target_path.exists() and not overwrite:
                continue
            with target_path.open("w", encoding="utf-8", newline="\n") as f:
                f.write(head + "\n")
            created.append(target_path)
            continue

        m = FENCE_RE.match(head)
        if not m:
            raise ValueError(
                f"Parse error near line {i+1}: expected opening fence or <CONTENT OMITTED>, got: {head!r}\n"
                f"Current path line: {raw_rel!r} (normalized to {rel!r})"
            )

        fence = m.group(1)
        i += 1  # after opening fence

        # 3) content until REAL closing fence (с lookahead-защитой)
        content_chunks: list[str] = []
        while i < len(lines):
            cur_no_nl = lines[i].rstrip("\n")

            if cur_no_nl == fence:
                # кандидат на closing fence.
                # Примем его как закрытие только если дальше реально начинается следующая секция (или EOF).
                j = next_meaningful(i + 1)
                if j >= len(lines) or is_section_start_at(j):
                    i += 1  # consume closing fence
                    break

                # иначе это "фальшивое закрытие" внутри контента (например, в строке/README)
                content_chunks.append(lines[i])
                i += 1
                continue

            content_chunks.append(lines[i])
            i += 1
        else:
            raise ValueError(
                f"Parse error: missing closing fence {fence!r} for file {rel!r}"
            )

        content = "".join(content_chunks)

        if target_path.exists() and not overwrite:
            continue

        with target_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(content)

        created.append(target_path)

    return created


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Restore a project from LLM text dump: PATH + fenced code blocks."
    )
    ap.add_argument(
        "input_text", type=Path, help="Path to the text file with project contents"
    )
    ap.add_argument(
        "project_root", type=Path, help="Path to the target project root directory"
    )
    ap.add_argument(
        "--no-overwrite", action="store_true", help="Do not overwrite existing files"
    )
    args = ap.parse_args()

    created = restore_project_from_text(
        args.input_text,
        args.project_root,
        overwrite=not args.no_overwrite,
    )
    print(f"Restored {len(created)} files into: {args.project_root}")


if __name__ == "__main__":
    main()
