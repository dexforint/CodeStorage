#!/usr/bin/env python3
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

MIN_AFFIX_LEN = 3
MIN_SPAM_FILE_COUNT = 3
PREVIEW_LIMIT = 15


def configure_stdio() -> None:
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stdin, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KB"
    if size < 1024**3:
        return f"{size / 1024 ** 2:.1f} MB"
    return f"{size / 1024 ** 3:.1f} GB"


def get_all_paths(root_path: Path) -> List[Path]:
    """
    Рекурсивно собирает пути ко всем файлам и подпапкам, включая корень.
    topdown=False — обход снизу вверх (сначала дети, потом родители).
    """
    paths: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(
        root_path, topdown=False, onerror=lambda e: print(f"[!] {e}")
    ):
        current_dir = Path(dirpath)
        for filename in filenames:
            paths.append(current_dir / filename)
        for dirname in dirnames:
            paths.append(current_dir / dirname)

    paths.append(root_path)
    return paths


def visible_names(paths: Sequence[Path]) -> List[str]:
    names: List[str] = []
    for path in paths:
        try:
            if path.name.startswith("."):
                continue
            if path.exists():
                names.append(path.name)
        except OSError:
            continue
    return names


def visible_stems(paths: Sequence[Path]) -> List[str]:
    """Имена папок и stems файлов (без расширения) — для поиска суффикса."""
    stems: List[str] = []
    for path in paths:
        try:
            if path.name.startswith(".") or not path.exists():
                continue
            if path.is_file():
                stems.append(path.stem)
            else:
                stems.append(path.name)
        except OSError:
            continue
    return stems


def detect_common_affix(
    strings: Sequence[str], *, as_suffix: bool = False, min_len: int = MIN_AFFIX_LEN
) -> str:
    """
    Ищет самый «полезный» общий префикс (или суффикс) среди строк.
    Не требует, чтобы аффикс был у всех — достаточно частого вхождения.
    """
    cleaned = [s for s in strings if s]
    if len(cleaned) < 2:
        return ""

    items = [s[::-1] if as_suffix else s for s in cleaned]
    items.sort()

    candidates = set()
    for i in range(len(items) - 1):
        common = os.path.commonprefix([items[i], items[i + 1]])
        if len(common) >= min_len:
            candidates.add(common)

    if not candidates:
        return ""

    best = ""
    max_score = 0
    for candidate in candidates:
        valid_count = sum(
            1
            for item in items
            if item.startswith(candidate) and len(item) > len(candidate)
        )
        if valid_count <= 1:
            continue
        score = len(candidate) * valid_count
        if score > max_score or (score == max_score and len(candidate) > len(best)):
            max_score = score
            best = candidate

    return best[::-1] if as_suffix else best


def affix_match_count(
    strings: Sequence[str], affix: str, *, as_suffix: bool = False
) -> int:
    if not affix:
        return 0
    if as_suffix:
        return sum(1 for s in strings if s.endswith(affix) and len(s) > len(affix))
    return sum(1 for s in strings if s.startswith(affix) and len(s) > len(affix))


def detect_repeated_files(
    paths: Sequence[Path], min_count: int = MIN_SPAM_FILE_COUNT
) -> List[Dict]:
    """
    Файлы с одинаковым именем и размером, повторённые в нескольких папках.
    Это типичный след «рекламных» файлов, которые кладут в каждую директорию.
    """
    groups: Dict[Tuple[str, int], List[Path]] = defaultdict(list)
    for path in paths:
        try:
            if not path.is_file() or path.name.startswith("."):
                continue
            groups[(path.name, path.stat().st_size)].append(path)
        except OSError:
            continue

    result = []
    for (name, size), group_paths in groups.items():
        unique_dirs = {p.parent for p in group_paths}
        if len(unique_dirs) >= min_count:
            result.append(
                {
                    "name": name,
                    "size": size,
                    "count": len(group_paths),
                    "paths": group_paths,
                }
            )

    result.sort(key=lambda item: (-item["count"], item["name"].lower()))
    return result


def split_filename(name: str) -> Tuple[str, str]:
    return os.path.splitext(name)


def clean_name(name: str, prefix: str, suffix: str, is_file: bool) -> Optional[str]:
    """Возвращает новое имя или None, если менять нечего / имя стало бы пустым."""
    new_name = name

    if prefix and new_name.startswith(prefix):
        new_name = new_name[len(prefix) :]

    if suffix:
        if is_file:
            stem, ext = split_filename(new_name)
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
            new_name = stem + ext
        elif new_name.endswith(suffix):
            new_name = new_name[: -len(suffix)]

    if is_file:
        stem, ext = split_filename(new_name)
        stem = re.sub(r"\s+", " ", stem).strip()
        new_name = stem + ext
    else:
        new_name = re.sub(r"\s+", " ", new_name).strip()

    new_name = new_name.rstrip(" .")

    if not new_name or new_name in {".", ".."}:
        return None
    if any(sep in new_name for sep in ("/", "\\")):
        return None
    if new_name == name:
        return None
    return new_name


def rel_display(path: Path, root_path: Path) -> str:
    try:
        relative = path.relative_to(
            root_path.parent if path == root_path else root_path
        )
        text = str(relative)
        return text if text != "." else path.name
    except ValueError:
        return str(path)


def parse_index_list(raw: str, count: int) -> Optional[List[int]]:
    indexes: Set[int] = set()
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            edges = part.split("-", 1)
            if not edges[0].isdigit() or not edges[1].isdigit():
                return None
            start, end = int(edges[0]), int(edges[1])
            if start > end:
                start, end = end, start
            for value in range(start, end + 1):
                if not 1 <= value <= count:
                    return None
                indexes.add(value)
        else:
            if not part.isdigit():
                return None
            value = int(part)
            if not 1 <= value <= count:
                return None
            indexes.add(value)
    return sorted(indexes)


def ask_yes_no(prompt: str, default_yes: bool = True) -> bool:
    suffix = " [Y/n]: " if default_yes else " [y/N]: "
    raw = input(prompt + suffix).strip().lower()
    if not raw:
        return default_yes
    return raw in {"y", "yes", "д", "да"}


def print_examples(title: str, rows: Sequence[str]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (нет примеров)")
        return
    for row in rows[:PREVIEW_LIMIT]:
        print(f"  {row}")
    extra = len(rows) - PREVIEW_LIMIT
    if extra > 0:
        print(f"  ... и ещё {extra}")


def configure_affix(kind: str, detected: str, sample_count: int, total: int) -> str:
    label = "Префикс" if kind == "prefix" else "Суффикс"
    if detected:
        print(
            f"\n{label} автоматически: '{detected}' "
            f"(совпадений: {sample_count} из {total})"
        )
        print("  Enter — использовать найденный")
        print("  0 — не удалять")
        print("  или введите своё значение (пробелы по краям учитываются)")
        raw = input("Ваш выбор: ")
        stripped = raw.strip()
        if not stripped:
            return detected
        if stripped == "0":
            return ""
        return raw
    print(f"\n{label} автоматически не найден.")
    raw = input(f"Введите {label.lower()} вручную (Enter — пропустить): ")
    return raw if raw.strip() else ""


def configure_spam(candidates: List[Dict]) -> List[Dict]:
    if not candidates:
        print("\nПовторяющиеся файлы-спам не найдены.")
        raw = input(
            "Можно указать точные имена файлов для удаления через ; (Enter — пропустить): "
        ).strip()
        if not raw:
            return []
        names = [part.strip() for part in raw.split(";") if part.strip()]
        return [{"name": name, "size": None, "count": 0, "paths": []} for name in names]

    print("\nВероятные спам-файлы (одинаковые имя и размер в нескольких папках):")
    for i, item in enumerate(candidates, 1):
        print(
            f"  [{i}] {item['count']} копий, {format_size(item['size'])} — {item['name']}"
        )

    print("\nУдалить эти файлы?")
    print("  Enter / Y — удалить все")
    print("  n — не удалять")
    print("  номера, например 1,2 или 1-3")
    raw = input("Ваш выбор: ").strip().lower()

    if raw in {"", "y", "yes", "д", "да"}:
        return candidates
    if raw in {"n", "no", "н", "нет"}:
        return []

    indexes = parse_index_list(raw, len(candidates))
    if not indexes:
        print("[!] Неверный ввод, повторяющиеся файлы пропускаются.")
        return []
    return [candidates[i - 1] for i in indexes]


def collect_delete_targets(
    all_paths: Sequence[Path], selected_spam: Sequence[Dict]
) -> List[Path]:
    by_name_size: Set[Tuple[str, Optional[int]]] = set()
    by_name: Set[str] = set()
    for item in selected_spam:
        if item.get("paths"):
            by_name_size.add((item["name"], item["size"]))
        else:
            by_name.add(item["name"])

    targets: List[Path] = []
    for path in all_paths:
        try:
            if not path.is_file():
                continue
            key = (path.name, path.stat().st_size)
            if key in by_name_size or path.name in by_name:
                targets.append(path)
        except OSError:
            if path.name in by_name:
                targets.append(path)
    return targets


def collect_rename_targets(
    all_paths: Sequence[Path],
    delete_set: Set[Path],
    prefix: str,
    suffix: str,
) -> List[Tuple[Path, str, bool]]:
    planned: List[Tuple[Path, str, bool]] = []
    for path in all_paths:
        if path in delete_set:
            continue
        try:
            if not path.exists():
                continue
            is_file = path.is_file()
        except OSError:
            continue

        new_name = clean_name(path.name, prefix, suffix, is_file)
        if new_name:
            planned.append((path, new_name, is_file))
    return planned


def execute_deletes(targets: Sequence[Path], root_path: Path) -> int:
    deleted = 0
    for path in targets:
        try:
            path.unlink()
            print(f"[DEL] {rel_display(path, root_path)}")
            deleted += 1
        except PermissionError:
            print(f"[!] Нет прав на удаление: '{path.name}'")
        except FileNotFoundError:
            print(f"[!] Уже отсутствует: '{path.name}'")
        except Exception as exc:
            print(f"[!] Ошибка удаления '{path.name}': {exc}")
    return deleted


def execute_renames(planned: Sequence[Tuple[Path, str, bool]], root_path: Path) -> int:
    renamed = 0
    for path, new_name, _is_file in planned:
        if not path.exists():
            print(f"[!] Пропущен (не найден): '{path.name}'")
            continue

        new_path = path.with_name(new_name)
        if new_path.exists():
            print(
                f"[!] Конфликт: нельзя переименовать '{path.name}' в '{new_name}' "
                f"— цель уже существует."
            )
            continue

        try:
            path.rename(new_path)
            print(f"[REN] {rel_display(path, root_path)} -> '{new_name}'")
            renamed += 1
        except PermissionError:
            print(f"[!] Нет прав на переименование: '{path.name}'")
        except Exception as exc:
            print(f"[!] Ошибка переименования '{path.name}': {exc}")
    return renamed


def main() -> None:
    print("=== Утилита очистки спама в именах файлов и папок ===")
    print("Удаляет частый префикс/суффикс и повторяющиеся рекламные файлы.\n")

    input_path = input("Введите полный путь к папке: ").strip()
    if input_path.startswith(("'", '"')) and input_path.endswith(("'", '"')):
        input_path = input_path[1:-1]

    root_path = Path(input_path).resolve()
    if not root_path.exists() or not root_path.is_dir():
        print(f"[!] Ошибка: путь '{root_path}' не существует или это не папка.")
        sys.exit(1)

    print(f"\nСканирование директории: {root_path} ...")
    all_paths = get_all_paths(root_path)
    if len(all_paths) <= 1:
        print("[!] В папке нет содержимого для обработки.")
        sys.exit(0)

    names = visible_names(all_paths)
    stems = visible_stems(all_paths)

    detected_prefix = detect_common_affix(names, as_suffix=False)
    detected_suffix = detect_common_affix(stems, as_suffix=True)
    spam_candidates = detect_repeated_files(all_paths)

    prefix_hits = affix_match_count(names, detected_prefix, as_suffix=False)
    suffix_hits = affix_match_count(stems, detected_suffix, as_suffix=True)

    print("\n" + "=" * 50)
    print("Результаты анализа")
    print("=" * 50)
    if detected_prefix:
        print(f"Префикс: '{detected_prefix}' ({prefix_hits} из {len(names)})")
    else:
        print("Префикс: не найден")
    if detected_suffix:
        print(f"Суффикс: '{detected_suffix}' ({suffix_hits} из {len(stems)})")
    else:
        print("Суффикс: не найден")
    if spam_candidates:
        print(f"Повторяющиеся файлы: {len(spam_candidates)} групп")
        for item in spam_candidates[:PREVIEW_LIMIT]:
            print(
                f"  • {item['count']} копий, {format_size(item['size'])} — {item['name']}"
            )
        extra = len(spam_candidates) - PREVIEW_LIMIT
        if extra > 0:
            print(f"  ... и ещё {extra}")
    else:
        print("Повторяющиеся файлы: не найдены")
    print("=" * 50)

    print("\nВыберите действие:")
    print("  [1] Использовать всё найденное")
    print("  [2] Настроить вручную")
    print("  [3] Отмена и выход")

    while True:
        choice = input("Ваш выбор: ").strip()
        if choice in {"1", "2", "3"}:
            break
        print("Неверный ввод. Выберите 1, 2 или 3.")

    if choice == "3":
        print("Операция отменена.")
        sys.exit(0)

    if choice == "1":
        prefix_to_remove = detected_prefix
        suffix_to_remove = detected_suffix
        selected_spam = spam_candidates
    else:
        prefix_to_remove = configure_affix(
            "prefix", detected_prefix, prefix_hits, len(names)
        )
        suffix_to_remove = configure_affix(
            "suffix", detected_suffix, suffix_hits, len(stems)
        )
        selected_spam = configure_spam(spam_candidates)

    if not prefix_to_remove and not suffix_to_remove and not selected_spam:
        print("\nНечего делать: префикс, суффикс и спам-файлы не выбраны.")
        sys.exit(0)

    delete_targets = collect_delete_targets(all_paths, selected_spam)
    delete_set = set(delete_targets)
    rename_targets = collect_rename_targets(
        all_paths, delete_set, prefix_to_remove, suffix_to_remove
    )

    delete_examples = [f"- {rel_display(p, root_path)}" for p in delete_targets]
    rename_examples = [
        f"- {rel_display(path, root_path)} -> '{new_name}'"
        for path, new_name, _ in rename_targets
    ]

    print("\n" + "=" * 50)
    print("Предпросмотр")
    print("=" * 50)
    if prefix_to_remove:
        print(f"Префикс к удалению: '{prefix_to_remove}'")
    if suffix_to_remove:
        print(f"Суффикс к удалению: '{suffix_to_remove}'")
    print(f"Файлов к удалению: {len(delete_targets)}")
    print(f"Элементов к переименованию: {len(rename_targets)}")
    print_examples("Примеры удаления:", delete_examples)
    print_examples("Примеры переименования:", rename_examples)

    if not delete_targets and not rename_targets:
        print("\nПосле применения правил изменений нет.")
        sys.exit(0)

    print("\nВнимание: удаление файлов безвозвратное (не в корзину).")
    if not ask_yes_no("Продолжить выполнение?", default_yes=True):
        print("Операция отменена.")
        sys.exit(0)

    print("\nНачало обработки...")
    deleted_count = execute_deletes(delete_targets, root_path)
    renamed_count = execute_renames(rename_targets, root_path)

    print("\n=== Готово! ===")
    print(f"Удалено файлов: {deleted_count}")
    print(f"Переименовано элементов: {renamed_count}")


if __name__ == "__main__":
    configure_stdio()
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Программа прервана пользователем.")
        sys.exit(1)
