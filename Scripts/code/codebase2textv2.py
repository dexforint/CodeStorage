#!/usr/bin/env python3
from __future__ import annotations

import codecs
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    source: Path
    output: Path

    max_chars: int = 100_000
    max_file_bytes: int = 0

    # Уровень 1: полное исключение.
    gitignore: bool = False
    ignore_file: list[Path] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)

    # Уровень 2: исключение только содержимого.
    content_ignore_file: list[Path] = field(default_factory=list)
    content_exclude: list[str] = field(default_factory=list)

    encoding: list[str] = field(default_factory=list)


# ===================== НАСТРОЙКИ =====================

BASE_DIR = Path(__file__).resolve().parent

source = input("Введите путь до папки с кодом: ").replace('"', "")
source = Path(source)

assert source.is_dir(), "Это должна быть папка!"

source_folder_name = source.name

CONFIG = Config(
    source=source,
    output=Path(f"data/code/{source_folder_name}"),
    # 0 — без ограничения.
    max_chars=100_000,
    max_file_bytes=0,
    # ---------- Уровень 1: полное исключение ----------
    # Использовать корневой .gitignore для полного исключения.
    gitignore=False,
    ignore_file=[
        # BASE_DIR / "my-project" / ".llmignore",
    ],
    exclude=[".git/", ".venv/", "__pycache__/", "node_modules/", "last_bugs.md"],
    # ------ Уровень 2: исключение содержимого ------
    content_ignore_file=[
        # BASE_DIR / "my-project" / ".llmcontentignore",
    ],
    content_exclude=[
        # Эти файлы останутся в дереве, но читаться не будут.
        ".env",
        ".env.*",
        "!.env.example",
        "*.pem",
        "*.key",
        "package-lock.json",
        "poetry.lock",
        "uv.lock",
        # Сохраняем структуру каталога, но не содержимое файлов.
        "docs/generated/",
        "*.dae",
        ####
        # "*.py",
        # "*.sh",
        "*.ps1",
        ".gitignore",
        "*.wbt",
        "*.json",
        "*.proto",
    ],
    encoding=[
        # "cp1251",
    ],
)

# =====================================================


@dataclass
class Node:
    path: Path
    rel: str
    kind: str
    children: list[Node] = field(default_factory=list)
    error: str | None = None


def quoted(value: str) -> str:
    """В том числе экранирует переводы строк в именах файлов."""
    return json.dumps(value, ensure_ascii=False)


def load_ignore(root: Path, config: Config):
    def read_patterns(paths: list[Path]) -> list[str]:
        patterns = []

        for path in paths:
            patterns.extend(
                Path(path).expanduser().read_text(encoding="utf-8-sig").splitlines()
            )

        return patterns

    # Уровень 1: полное исключение.
    full_patterns = []

    if config.gitignore:
        gitignore_path = root / ".gitignore"

        if gitignore_path.is_file():
            full_patterns.extend(
                gitignore_path.read_text(encoding="utf-8-sig").splitlines()
            )

    full_patterns.extend(read_patterns(config.ignore_file))
    full_patterns.extend(config.exclude)

    # Уровень 2: исключение содержимого.
    content_patterns = read_patterns(config.content_ignore_file)
    content_patterns.extend(config.content_exclude)

    if not full_patterns and not content_patterns:
        return None, None

    try:
        import pathspec
    except ImportError:
        raise ValueError(
            "Для правил исключения установите зависимость: "
            "python -m pip install pathspec"
        ) from None

    full_ignore = (
        pathspec.GitIgnoreSpec.from_lines(full_patterns) if full_patterns else None
    )

    content_ignore = (
        pathspec.GitIgnoreSpec.from_lines(content_patterns)
        if content_patterns
        else None
    )

    return full_ignore, content_ignore


def classify(path: Path) -> str:
    # lstat не переходит по символическим ссылкам.
    mode = path.lstat().st_mode

    if stat.S_ISLNK(mode):
        return "symlink"

    # Junction в Windows также не обходим, если Python умеет его определять.
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        return "symlink"

    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"

    # FIFO, сокеты, устройства и т. п. читать нельзя.
    return "special"


def scan_directory(
    path: Path,
    root: Path,
    output: Path,
    ignore,
    rel: str = "",
) -> Node:
    node = Node(path=path, rel=rel, kind="directory")

    try:
        entries = list(path.iterdir())
    except OSError as exc:
        node.error = str(exc)
        return node

    for entry in entries:
        # Если output находится внутри root, исключаем весь output.
        if entry == output:
            continue

        relative = entry.relative_to(root).as_posix()

        try:
            kind = classify(entry)
            error = None
        except OSError as exc:
            kind = "unknown"
            error = str(exc)

        match_path = relative + ("/" if kind == "directory" else "")
        if ignore is not None and ignore.match_file(match_path):
            continue

        if kind == "directory":
            child = scan_directory(entry, root, output, ignore, relative)
        else:
            child = Node(
                path=entry,
                rel=relative,
                kind=kind,
                error=error,
            )

        node.children.append(child)

    # Детерминированный порядок: сначала каталоги, потом остальные элементы.
    node.children.sort(
        key=lambda item: (
            item.kind != "directory",
            item.path.name.casefold(),
            item.path.name,
        )
    )

    return node


def node_label(node: Node) -> str:
    label = quoted(node.path.name)

    if node.kind == "directory":
        label += "/"
    elif node.kind == "symlink":
        label += " [symbolic link / junction; not followed]"
    elif node.kind == "special":
        label += " [special file; not read]"
    elif node.kind == "unknown":
        label += " [unknown type]"

    if node.error:
        label += " [ERROR: " + quoted(node.error) + "]"

    return label


def tree_lines(node: Node, prefix: str = ""):
    for index, child in enumerate(node.children):
        last = index == len(node.children) - 1
        connector = "└── " if last else "├── "

        yield prefix + connector + node_label(child) + "\n"

        if child.kind == "directory":
            continuation = "    " if last else "│   "
            yield from tree_lines(child, prefix + continuation)


def walk_nodes(node: Node):
    for child in node.children:
        yield child
        if child.kind == "directory":
            yield from walk_nodes(child)


def looks_like_text(text: str) -> bool:
    """
    Эвристика, а не математически точное определение текстового файла.
    Проверяем уже декодированный текст, чтобы не отвергать UTF-16 из-за NUL
    в исходных байтах.
    """
    if "\x00" in text:
        return False

    if not text:
        return True

    allowed_controls = "\t\n\r\f"
    suspicious = sum(
        1
        for char in text
        if (ord(char) < 32 and char not in allowed_controls) or ord(char) == 127
    )

    return suspicious / len(text) <= 0.01


def decode_text(data: bytes, extra_encodings: list[str]):
    # UTF-32 проверяется раньше UTF-16: их BOM могут иметь общий префикс.
    bom_encodings = (
        (codecs.BOM_UTF32_LE, "utf-32"),
        (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    )

    for bom, encoding in bom_encodings:
        if data.startswith(bom):
            candidates = [encoding]
            break
    else:
        candidates = list(dict.fromkeys(["utf-8", *extra_encodings]))

    for encoding in candidates:
        try:
            text = data.decode(encoding, errors="strict")
        except UnicodeError:
            continue

        if looks_like_text(text):
            return text, encoding

    return None, None


def read_text_file(
    path: Path,
    max_bytes: int,
    extra_encodings: list[str],
):
    try:
        # Повторно проверяем тип: после сканирования файл мог измениться.
        if classify(path) != "file":
            return None, None, "file type changed; not read"

        if max_bytes and path.stat().st_size > max_bytes:
            return None, None, f"larger than {max_bytes} bytes"

        with path.open("rb") as stream:
            data = stream.read(max_bytes + 1 if max_bytes else -1)

        # Файл мог вырасти между stat() и read().
        if max_bytes and len(data) > max_bytes:
            return None, None, f"larger than {max_bytes} bytes"

        text, encoding = decode_text(data, extra_encodings)

        if text is None:
            return None, None, "binary file or unsupported text encoding"

        return text, encoding, None

    except OSError as exc:
        return None, None, "read error: " + str(exc)


class ChunkWriter:
    """
    Разбивает единый текстовый поток на UTF-8 файлы.

    Лимит измеряется через len(str), то есть в Unicode code points,
    а не в байтах или токенах. Служебный текст тоже входит в лимит.
    """

    def __init__(self, directory: Path, max_chars: int):
        self.directory = directory
        self.max_chars = max_chars
        self.paths: list[Path] = []
        self.stream = None
        self.current_chars = 0
        self.total_chars = 0

    def _open_next(self):
        if self.stream is not None:
            self.stream.close()

        path = self.directory / f"context-{len(self.paths) + 1:04d}.txt"

        # newline="" предотвращает замену \n на \r\n в Windows.
        self.stream = path.open("w", encoding="utf-8", newline="")
        self.paths.append(path)
        self.current_chars = 0

    def write(self, text: str):
        position = 0

        while position < len(text):
            if self.stream is None or (
                self.max_chars and self.current_chars == self.max_chars
            ):
                self._open_next()

            available = (
                self.max_chars - self.current_chars
                if self.max_chars
                else len(text) - position
            )

            end = position + min(available, len(text) - position)
            fragment = text[position:end]

            self.stream.write(fragment)
            self.current_chars += len(fragment)
            self.total_chars += len(fragment)
            position = end

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def export(root: Path, output: Path, args, ignore, content_ignore):
    output.mkdir(parents=True, exist_ok=True)
    tree = scan_directory(root, root, output, ignore)

    text_count = 0
    skipped_count = 0

    # Временный каталог находится внутри output, который не входит в снимок.
    with tempfile.TemporaryDirectory(
        prefix=".context-build-",
        dir=output,
    ) as temporary:
        with ChunkWriter(Path(temporary), args.max_chars) as writer:
            writer.write(
                "SOURCE CODE SNAPSHOT\n"
                "Paths are relative to the source directory.\n"
                "Concatenate numbered parts in numeric order.\n\n"
                "=== DIRECTORY TREE ===\n"
            )

            writer.write(node_label(tree) + "\n")

            for line in tree_lines(tree):
                writer.write(line)

            writer.write("\n=== FILE CONTENTS ===\n")

            for node in walk_nodes(tree):
                if node.kind != "file":
                    continue

                writer.write("\n--- BEGIN FILE " + quoted(node.rel) + " ---\n")

                # Проверка выполняется до чтения файла.
                # Исключённое содержимое даже не загружается в память.
                if content_ignore is not None and content_ignore.match_file(node.rel):
                    skipped_count += 1

                    writer.write("[SKIPPED: content excluded by pattern]\n")

                else:
                    text, encoding, reason = read_text_file(
                        node.path,
                        args.max_file_bytes,
                        args.encoding,
                    )

                    if text is None:
                        skipped_count += 1

                        writer.write("[SKIPPED: " + quoted(reason) + "]\n")

                    else:
                        text_count += 1

                        writer.write(
                            f"[encoding={encoding}; "
                            f"characters={len(text)}]\n"
                            "--- CONTENT ---\n"
                        )

                        writer.write(text)

                        if text and not text.endswith("\n"):
                            writer.write("\n")

                writer.write("--- END FILE " + quoted(node.rel) + " ---\n")

        generated_names = {path.name for path in writer.paths}

        # Каждый отдельный файл заменяется атомарно.
        # Весь набор файлов при этом не является одной атомарной операцией.
        for path in writer.paths:
            os.replace(path, output / path.name)

        # Удаляем лишние части от предыдущего запуска.
        # Остальные файлы в output не трогаем.
        for old in output.iterdir():
            if (
                re.fullmatch(r"context-\d{4,}\.txt", old.name)
                and old.name not in generated_names
                and (old.is_file() or old.is_symlink())
            ):
                old.unlink()

        print(f"Output: {output}", file=sys.stderr)
        print(
            f"Parts: {len(writer.paths)}; "
            f"characters: {writer.total_chars}; "
            f"text files: {text_count}; "
            f"skipped regular files: {skipped_count}",
            file=sys.stderr,
        )


def main() -> int:
    config = CONFIG

    try:
        if config.max_chars < 0 or config.max_file_bytes < 0:
            raise ValueError("Ограничения не могут быть отрицательными")

        for encoding in config.encoding:
            try:
                codecs.lookup(encoding)
            except LookupError:
                raise ValueError(f"Неизвестная кодировка: {encoding}") from None

        root = config.source.expanduser().resolve(strict=True)
        output = config.output.expanduser().resolve()

        if not root.is_dir():
            raise ValueError(f"Исходный путь должен быть каталогом: {root}")

        # Каталог результата можно расположить внутри исходной папки:
        # он автоматически исключается из снимка.
        # Но он не должен совпадать с исходной папкой или содержать её.
        if output == root or output in root.parents:
            raise ValueError(
                "Каталог результата не должен совпадать с исходной "
                "папкой или быть её родительским каталогом"
            )

        full_ignore, content_ignore = load_ignore(root, config)

        export(
            root,
            output,
            config,
            full_ignore,
            content_ignore,
        )

    except (OSError, ValueError, RecursionError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
