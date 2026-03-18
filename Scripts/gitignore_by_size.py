#!/usr/bin/env python3
"""
Автоматически добавляет/удаляет файлы в .gitignore по размеру.
  - Файлы > лимита добавляются
  - Файлы <= лимита или удалённые — убираются
  - Дубликатов не будет
  - Ручные записи не затрагиваются

Использование:
  python gitignore_by_size.py          # лимит по умолчанию 5 МБ
  python gitignore_by_size.py 10M
  python gitignore_by_size.py 500K
"""

import os
import sys
import re

# ── Маркеры автоблока ──────────────────────────────────────────────
MARKER_START = "# >>> AUTO-SIZE-IGNORE START >>>"
MARKER_END = "# <<< AUTO-SIZE-IGNORE END <<<"
GITIGNORE = ".gitignore"


def parse_size(value: str) -> int:
    """Разбирает строку вида '5M', '500K', '1G' → байты."""
    value = value.strip().upper()
    units = {"K": 1024, "M": 1024**2, "G": 1024**3}
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMG])?B?", value)
    if not match:
        raise ValueError(f"Неверный формат размера: '{value}'. Примеры: 5M, 500K, 1G")
    number = float(match.group(1))
    unit = match.group(2)
    return int(number * units.get(unit, 1))


def find_large_files(root: str, max_size: int) -> dict[str, int]:
    """Возвращает {относительный_путь: размер} для файлов > max_size."""
    large = {}
    for dirpath, dirnames, filenames in os.walk(root):
        # Пропускаем .git и саму папку скрипта не трогаем
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for fname in filenames:
            filepath = os.path.join(dirpath, fname)
            rel = os.path.relpath(filepath, root).replace(os.sep, "/")

            if rel == GITIGNORE:
                continue
            try:
                size = os.path.getsize(filepath)
                if size > max_size:
                    large[rel] = size
            except OSError:
                pass
    return large


def read_gitignore() -> tuple[list[str], set[str]]:
    """
    Читает .gitignore и разделяет на:
      - manual_lines: все строки ВНЕ автоблока
      - auto_entries: множество путей внутри автоблока
    """
    if not os.path.exists(GITIGNORE):
        return [], set()

    with open(GITIGNORE, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    manual_lines = []
    auto_entries = set()
    inside_block = False

    for line in lines:
        if line.strip() == MARKER_START:
            inside_block = True
            continue
        if line.strip() == MARKER_END:
            inside_block = False
            continue
        if inside_block:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                auto_entries.add(stripped)
        else:
            manual_lines.append(line)

    return manual_lines, auto_entries


def manual_already_ignores(manual_lines: list[str]) -> set[str]:
    """Собирает множество записей из ручной части (простое сравнение строк)."""
    entries = set()
    for line in manual_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.add(stripped)
    return entries


def write_gitignore(manual_lines: list[str], auto_entries: set[str], max_size: int):
    """Записывает .gitignore: ручная часть + автоблок."""
    # Убираем лишние пустые строки в конце ручной части
    while manual_lines and manual_lines[-1].strip() == "":
        manual_lines.pop()

    output = []

    # Ручная часть
    if manual_lines:
        output.extend(manual_lines)
        output.append("")

    # Автоблок
    if auto_entries:
        size_str = format_size(max_size)
        output.append(MARKER_START)
        output.append(f"# Файлы больше {size_str} (автоматически, не редактировать)")
        for entry in sorted(auto_entries):
            output.append(entry)
        output.append(MARKER_END)

    output.append("")  # финальный перевод строки

    with open(GITIGNORE, "w", encoding="utf-8") as f:
        f.write("\n".join(output))


def format_size(size_bytes: int) -> str:
    """Форматирует байты в человекочитаемую строку."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.1f} ГБ"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.1f} МБ"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} КБ"
    return f"{size_bytes} Б"


def main():
    # ── Парсинг аргументов ──
    if len(sys.argv) > 1:
        try:
            max_size = parse_size(sys.argv[1])
        except ValueError as e:
            print(f"Ошибка: {e}")
            sys.exit(1)
    else:
        max_size = 5 * 1024 * 1024  # 5 МБ по умолчанию

    print(f"Лимит: {format_size(max_size)}\n")

    # ── Сканирование ──
    large_files = find_large_files(".", max_size)
    manual_lines, old_auto = read_gitignore()
    manual_entries = manual_already_ignores(manual_lines)

    # Не добавляем то, что уже вручную прописано
    new_auto = set(large_files.keys()) - manual_entries

    # ── Вычисляем разницу ──
    added = new_auto - old_auto
    removed = old_auto - new_auto
    unchanged = new_auto & old_auto

    # ── Отчёт ──
    if added:
        print("Добавлены:")
        for f in sorted(added):
            print(f"  + {f}  ({format_size(large_files[f])})")

    if removed:
        print("Удалены из .gitignore:")
        for f in sorted(removed):
            if os.path.exists(f):
                print(f"  − {f}  (теперь {format_size(os.path.getsize(f))})")
            else:
                print(f"  − {f}  (файл удалён)")

    if unchanged:
        print(f"Без изменений: {len(unchanged)} файл(ов)")

    if not added and not removed:
        print("Изменений нет.")
        return

    # ── Запись ──
    write_gitignore(manual_lines, new_auto, max_size)
    print(f"\n✅ .gitignore обновлён: +{len(added)} / −{len(removed)}")


if __name__ == "__main__":
    main()
