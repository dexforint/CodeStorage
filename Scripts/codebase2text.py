from __future__ import annotations

import io
import os
from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Tuple, List


# ------------------------- defaults -------------------------

DEFAULT_HIDE_IN_TREE = [
    # VCS / IDE
    ".git/",
    ".hg/",
    ".svn/",
    ".idea/",
    ".vscode/",
    # Python env / caches
    ".venv/",
    "venv/",
    "__pycache__/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".tox/",
    ".nox/",
    # JS / web
    "node_modules/",
    ".next/",
    ".nuxt/",
    # Build artifacts
    "dist/",
    "build/",
    "out/",
    "coverage/",
    # OS junk
    ".DS_Store",
    "Thumbs.db",
]


# ------------------------- gitignore-like matcher -------------------------


@dataclass(frozen=True)
class _GlobRule:
    raw: str
    anchored: bool  # startswith '/'
    dir_only: bool  # endswith '/'
    has_slash: bool  # contains '/'
    parts: tuple[str, ...]  # pattern parts without leading/trailing '/'


def _compile_glob_rules(patterns: Optional[Iterable[str]]) -> list[_GlobRule]:
    rules: list[_GlobRule] = []
    if not patterns:
        return rules

    for p in patterns:
        if p is None:
            continue
        p = str(p).strip()
        if not p or p.startswith("#"):
            continue

        dir_only = p.endswith("/")
        anchored = p.startswith("/")

        p2 = p.strip("/")
        if not p2:
            continue

        parts = tuple(p2.split("/"))
        has_slash = "/" in p2
        rules.append(
            _GlobRule(
                raw=p,
                anchored=anchored,
                dir_only=dir_only,
                has_slash=has_slash,
                parts=parts,
            )
        )
    return rules


@lru_cache(maxsize=200_000)
def _match_parts(path_parts: tuple[str, ...], pat_parts: tuple[str, ...]) -> bool:
    """Компонентный матчинг с поддержкой '**'."""
    if not pat_parts:
        return not path_parts

    head, *tail = pat_parts

    if head == "**":
        tail_t = tuple(tail)
        for k in range(len(path_parts) + 1):
            if _match_parts(path_parts[k:], tail_t):
                return True
        return False

    if not path_parts:
        return False

    if fnmatchcase(path_parts[0], head):
        return _match_parts(path_parts[1:], tuple(tail))
    return False


def _rule_matches(rule: _GlobRule, rel_posix: str, is_dir: bool) -> bool:
    rel_posix = rel_posix.strip("/")
    parts = tuple(rel_posix.split("/")) if rel_posix else tuple()
    dir_parts = parts if is_dir else parts[:-1]

    if rule.dir_only:
        # Директория (и всё внутри неё)
        if rule.has_slash:
            pat = rule.parts
            if rule.anchored:
                if len(dir_parts) < len(pat):
                    return False
                return _match_parts(dir_parts[: len(pat)], pat)
            else:
                if len(dir_parts) < len(pat):
                    return False
                for i in range(0, len(dir_parts) - len(pat) + 1):
                    if _match_parts(dir_parts[i : i + len(pat)], pat):
                        return True
                return False
        else:
            # ".venv/" — имя директории где угодно
            name_pat = rule.parts[0]
            return any(fnmatchcase(d, name_pat) for d in dir_parts)

    # Не dir_only
    if rule.has_slash:
        pat = rule.parts
        if rule.anchored:
            return _match_parts(parts, pat)
        else:
            if len(parts) < len(pat):
                return False
            for i in range(0, len(parts) - len(pat) + 1):
                if _match_parts(parts[i:], pat):
                    return True
            return False
    else:
        if not parts:
            return False
        return fnmatchcase(parts[-1], rule.parts[0])


def _matches_any(rel_posix: str, is_dir: bool, rules: list[_GlobRule]) -> bool:
    return any(_rule_matches(r, rel_posix, is_dir) for r in rules)


# ------------------------- binary detection -------------------------

_TEXTCHARS = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})


def _looks_binary(path: Path, sniff_bytes: int = 8192) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(sniff_bytes)
    except OSError:
        return True

    if not chunk:
        return False
    if b"\x00" in chunk:
        return True

    nontext = sum(b not in _TEXTCHARS for b in chunk)
    return (nontext / len(chunk)) > 0.30


def _guess_language(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".md": "markdown",
        ".sh": "bash",
        ".ps1": "powershell",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
        ".xml": "xml",
    }.get(ext, "")


def _choose_fence(text: str, min_len: int = 3) -> str:
    max_run = 0
    run = 0
    for ch in text:
        if ch == "`":
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return "`" * max(min_len, max_run + 1)


# ------------------------- tree -------------------------


class _Node:
    def __init__(self) -> None:
        self.children: dict[str, _Node | None] = {}


def _add_dir(tree: _Node, parts: list[str]) -> None:
    cur = tree
    for part in parts:
        nxt = cur.children.get(part)
        if not isinstance(nxt, _Node):
            nxt = _Node()
            cur.children[part] = nxt
        cur = nxt


def _add_file(tree: _Node, parts: list[str]) -> None:
    if not parts:
        return
    *dirs, fname = parts
    _add_dir(tree, dirs)
    cur = tree
    for d in dirs:
        cur = cur.children[d]  # type: ignore[assignment]
    cur.children.setdefault(fname, None)


def _render_tree(tree: _Node, root_name: str) -> str:
    lines: list[str] = [f"{root_name}/"]

    def rec(node: _Node, prefix: str) -> None:
        items = list(node.children.items())

        def sort_key(kv: tuple[str, _Node | None]) -> tuple[int, str]:
            name, child = kv
            is_file = child is None
            return (1 if is_file else 0, name.lower())

        items.sort(key=sort_key)

        for i, (name, child) in enumerate(items):
            last = i == len(items) - 1
            connector = "└── " if last else "├── "
            if child is None:
                lines.append(f"{prefix}{connector}{name}")
            else:
                lines.append(f"{prefix}{connector}{name}/")
                rec(child, prefix + ("    " if last else "│   "))

    rec(tree, "")
    return "\n".join(lines)


# ------------------------- main -------------------------


def codebase_folder_to_llm_text(
    folder_path: str | Path,
    allowed: Optional[list[str]] = None,
    ignore_content: Optional[list[str]] = None,
    hide_in_tree: Optional[list[str]] = None,
    *,
    use_default_hide_in_tree: bool = True,
    out_dir: str | Path | None = None,
    max_file_chars: int | None = None,
    max_output_chars: int | None = None,
) -> Tuple[List[Path], Path]:
    """
    Создаёт в out_dir (по умолчанию текущая директория):
      - один или несколько файлов с кодовой базой:
          * без разбиения: <root>.txt
          * с разбиением:  <root>__part001.txt, <root>__part002.txt, ...
      - файл со списком реально включённых по содержимому файлов:
          <root>__included_files.txt

    Разбиение по max_output_chars:
      - секция одного файла (заголовок + fenced code + текст) НИКОГДА не режется между частями
      - если секция > max_output_chars, она пишется отдельным part-файлом целиком

    Возвращает:
      (list_of_codebase_part_paths, included_files_list_path)
    """
    root = Path(folder_path).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"folder_path is not a directory: {root}")

    if max_output_chars is not None and max_output_chars <= 0:
        raise ValueError("max_output_chars must be a positive integer or None")

    out_base_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    out_base_dir.mkdir(parents=True, exist_ok=True)

    included_list_path = out_base_dir / f"{root.name}__included_files.txt"

    hide_patterns: list[str] = []
    if use_default_hide_in_tree:
        hide_patterns.extend(DEFAULT_HIDE_IN_TREE)
    if hide_in_tree:
        hide_patterns.extend(hide_in_tree)

    allowed_rules = _compile_glob_rules(allowed)
    ignore_rules = _compile_glob_rules(ignore_content)
    hide_rules = _compile_glob_rules(hide_patterns)

    def rel_posix(p: Path) -> str:
        return p.relative_to(root).as_posix()

    # 1) Сканируем, не заходим в hide_in_tree директории
    all_dirs: set[str] = set()
    all_files: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        cur = Path(dirpath)
        cur_rel = "" if cur == root else rel_posix(cur)

        kept: list[str] = []
        for d in dirnames:
            rp = rel_posix(cur / d)
            if _matches_any(rp, is_dir=True, rules=hide_rules):
                continue
            kept.append(d)
        dirnames[:] = kept

        if cur_rel:
            all_dirs.add(cur_rel)

        for fn in filenames:
            rp = rel_posix(cur / fn)
            if _matches_any(rp, is_dir=False, rules=hide_rules):
                continue
            all_files.add(rp)

    # 2) Allowed (если задан)
    def is_allowed(rel: str, is_dir: bool) -> bool:
        if allowed_rules == []:
            return True
        return _matches_any(rel, is_dir=is_dir, rules=allowed_rules)

    if allowed is None:
        included_files = set(all_files)
        included_dirs = set(all_dirs)
    else:
        included_files = {f for f in all_files if is_allowed(f, is_dir=False)}
        included_dirs = {d for d in all_dirs if is_allowed(d, is_dir=True)}

        def add_ancestors(rel_path: str) -> None:
            parts = rel_path.split("/")
            for i in range(1, len(parts)):
                included_dirs.add("/".join(parts[:i]))

        for f in included_files:
            add_ancestors(f)
        for d in list(included_dirs):
            add_ancestors(d)

    # 3) Tree
    tree = _Node()
    for d in sorted(included_dirs):
        _add_dir(tree, d.split("/"))
    for f in sorted(included_files):
        _add_file(tree, f.split("/"))

    tree_text = _render_tree(tree, root.name)

    # 4) Prepare per-file sections (so we can split safely)
    included_content_files: list[str] = []

    def should_show_content(rel: str, abs_path: Path) -> tuple[bool, str]:
        if _matches_any(rel, is_dir=False, rules=ignore_rules):
            return False, "ignored_by_pattern"
        if _looks_binary(abs_path):
            return False, "binary"
        return True, ""

    def make_section(rel: str) -> tuple[str, bool]:
        abs_path = root / rel
        show, reason = should_show_content(rel, abs_path)

        if not show:
            sec = f"{rel}\n<CONTENT OMITTED: {reason}>\n\n"
            return sec, False

        try:
            raw = abs_path.read_bytes()
        except OSError as e:
            sec = f"{rel}\n<CONTENT OMITTED: read_error: {e}>\n\n"
            return sec, False

        text = raw.decode("utf-8", errors="replace")
        if max_file_chars is not None and len(text) > max_file_chars:
            text = text[:max_file_chars] + "\n<TRUNCATED>\n"

        lang = _guess_language(abs_path)
        fence = _choose_fence(text)

        # Формируем секцию
        buf = io.StringIO()
        buf.write(rel)
        buf.write("\n")
        buf.write(f"{fence}{lang}\n")
        buf.write(text)
        if not text.endswith("\n"):
            buf.write("\n")
        buf.write(f"{fence}\n\n")
        return buf.getvalue(), True

    file_sections: list[tuple[str, str, bool]] = (
        []
    )  # (rel, section_text, included_content)
    for rel in sorted(included_files):
        sec, included = make_section(rel)
        file_sections.append((rel, sec, included))
        if included:
            included_content_files.append(rel)

    # 5) Write codebase text files (single or split)
    part_paths: list[Path] = []

    def write_part(idx: int, content: str) -> Path:
        if max_output_chars is None:
            path = out_base_dir / f"{root.name}.txt"
        else:
            path = out_base_dir / f"{root.name}__part{idx:03d}.txt"
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return path

    if max_output_chars is None:
        # один файл
        out = io.StringIO()
        out.write(tree_text)
        out.write("\n\n---\n\n")
        for _, sec, _ in file_sections:
            out.write(sec)
        part_paths.append(write_part(1, out.getvalue()))
    else:
        # несколько файлов
        part_idx = 1
        cur = io.StringIO()
        cur_len = 0

        def cur_write(s: str) -> None:
            nonlocal cur_len
            cur.write(s)
            cur_len += len(s)

        # Заголовок для part1: дерево + разделитель
        part1_prefix = tree_text + "\n\n---\n\n"
        cur_write(part1_prefix)

        # Если префикс уже больше лимита — записываем part001 с одним деревом, а контент пойдёт дальше
        if cur_len > max_output_chars:
            part_paths.append(write_part(part_idx, cur.getvalue()))
            part_idx += 1
            cur = io.StringIO()
            cur_len = 0
            # Для продолжений — просто разделитель (без дерева)
            cur_write("---\n\n")

        continuation_prefix = "---\n\n"

        for rel, sec, _included in file_sections:
            sec_len = len(sec)

            # Если секция сама больше лимита: пишем её отдельным part-файлом целиком
            if sec_len > max_output_chars:
                # сначала закрываем текущий part, если там есть что-то кроме continuation_prefix/пустоты
                cur_text = cur.getvalue()
                if cur_text.strip():
                    part_paths.append(write_part(part_idx, cur_text))
                    part_idx += 1
                    cur = io.StringIO()
                    cur_len = 0

                # отдельный part для большой секции
                oversized = io.StringIO()
                oversized.write(continuation_prefix)
                oversized.write(sec)
                part_paths.append(write_part(part_idx, oversized.getvalue()))
                part_idx += 1

                # новый текущий part для последующих секций
                cur = io.StringIO()
                cur_len = 0
                cur_write(continuation_prefix)
                continue

            # Обычный случай: проверяем, влезает ли секция в текущий part
            if cur_len + sec_len > max_output_chars and cur_len > 0:
                # закрыть текущий
                part_paths.append(write_part(part_idx, cur.getvalue()))
                part_idx += 1
                # начать новый
                cur = io.StringIO()
                cur_len = 0
                cur_write(continuation_prefix)

            cur_write(sec)

        # финализировать последний part
        if cur.getvalue().strip():
            part_paths.append(write_part(part_idx, cur.getvalue()))

    # 6) Included files list
    included_list_path.write_text(
        "\n".join(included_content_files).rstrip() + "\n",
        encoding="utf-8",
    )

    return part_paths, included_list_path
