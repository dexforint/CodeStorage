import os
import sys
from pathlib import Path
from typing import List, Optional


def get_all_paths(root_path: Path) -> List[Path]:
    """
    Рекурсивно собирает пути ко всем файлам и подпапкам, включая саму корневую папку.
    """
    paths = []
    # topdown=False гарантирует обход снизу-вверх (сначала дети, потом родители)
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        current_dir = Path(dirpath)
        for filename in filenames:
            paths.append(current_dir / filename)
        for dirname in dirnames:
            paths.append(current_dir / dirname)

    paths.append(root_path)
    return paths


def auto_detect_prefix(paths: List[Path]) -> str:
    """
    Автоматически определяет наиболее частый общий префикс среди файлов.
    Находит префикс, который встречается у МНОГИХ файлов, игнорируя те,
    которые называются иначе.
    """
    # Собираем только имена файлов/папок, игнорируя скрытые
    names = [p.name for p in paths if not p.name.startswith(".") and p.is_dir()]

    if len(names) < 2:
        return ""

    names.sort()
    candidates = set()

    # Шаг 1: Находим все возможные общие префиксы между соседними отсортированными именами
    for i in range(len(names) - 1):
        cp = os.path.commonprefix([names[i], names[i + 1]])
        if len(cp) > 0:
            candidates.add(cp)

    if not candidates:
        return ""

    best_prefix = ""
    max_score = 0

    # Шаг 2: Оцениваем каждого кандидата
    for candidate in candidates:
        # Считаем, у скольких файлов есть этот префикс.
        # Условие len(name) > len(candidate) гарантирует, что после удаления
        # префикса останется хотя бы 1 символ (имя не станет пустым).
        valid_count = sum(
            1
            for name in names
            if name.startswith(candidate) and len(name) > len(candidate)
        )

        # Нас интересуют префиксы, которые есть хотя бы у двух подходящих файлов
        if valid_count > 1:
            # Вес = длина префикса умноженная на количество файлов (максимизируем "пользу")
            score = len(candidate) * valid_count

            # Если счет больше, или при равном счете префикс длиннее — берем его
            if score > max_score or (
                score == max_score and len(candidate) > len(best_prefix)
            ):
                max_score = score
                best_prefix = candidate

    return best_prefix


def rename_item(path: Path, prefix: str) -> Optional[Path]:
    """
    Переименовывает один файл или папку, удаляя префикс.
    Возвращает новый путь, если переименование прошло успешно, или None.
    """
    if not path.name.startswith(prefix):
        return None  # Префикса нет, пропускаем

    if path.name == prefix:
        print(
            f"[-] Пропущен: '{path.name}' (имя не может быть пустым после удаления префикса)"
        )
        return None

    new_name = path.name[len(prefix) :]
    new_path = path.with_name(new_name)

    if new_path.exists():
        print(
            f"[!] Ошибка: Невозможно переименовать '{path.name}' в '{new_name}', так как файл уже существует."
        )
        return None

    try:
        path.rename(new_path)
        # print(f"[+] Переименовано: '{path.name}' -> '{new_name}'")
        return new_path
    except PermissionError:
        print(f"[!] Ошибка доступа: Нет прав для переименования '{path.name}'")
        return None
    except Exception as e:
        print(f"[!] Неизвестная ошибка при переименовании '{path.name}': {e}")
        return None


def main():
    print("=== Утилита массового удаления префикса ===")

    # Получаем путь от пользователя
    input_path = input("Введите полный путь к папке: ").strip()

    # Убираем кавычки, если путь был скопирован из терминала
    if input_path.startswith(('"', "'")) and input_path.endswith(('"', "'")):
        input_path = input_path[1:-1]

    root_path = Path(input_path).resolve()

    if not root_path.exists() or not root_path.is_dir():
        print(f"[!] Ошибка: Путь '{root_path}' не существует или не является папкой.")
        sys.exit(1)

    print(f"\nСканирование директории: {root_path} ...")
    all_paths = get_all_paths(root_path)

    if len(all_paths) <= 1:
        print("[!] В папке нет содержимого для обработки.")
        sys.exit(0)

    # Автоопределение
    detected_prefix = auto_detect_prefix(all_paths)
    prefix_to_remove = ""

    print("\n" + "=" * 40)
    if detected_prefix:
        print(f"Автоматически определен префикс: '{detected_prefix}'")
    else:
        print(
            "Не удалось автоматически определить общий префикс (возможно, файлы начинаются по-разному)."
        )
    print("=" * 40)

    # Интерактивное меню
    while True:
        print("\nВыберите действие:")
        if detected_prefix:
            print(f"  [1] Использовать найденный префикс ('{detected_prefix}')")
        print("  [2] Задать префикс вручную")
        print("  [3] Отмена и выход")

        choice = input("Ваш выбор: ").strip()

        if choice == "1" and detected_prefix:
            prefix_to_remove = detected_prefix
            break
        elif choice == "2":
            prefix_to_remove = input("Введите префикс для удаления: ")
            if not prefix_to_remove:
                print("Префикс не может быть пустым. Попробуйте снова.")
                continue
            break
        elif choice == "3":
            print("Операция отменена.")
            sys.exit(0)
        else:
            print("Неверный ввод. Пожалуйста, выберите существующий пункт.")

    print("\nНачало переименования...")
    renamed_count = 0

    # Важно: all_paths уже отсортированы снизу-вверх (благодаря topdown=False в os.walk)
    for path in all_paths:
        # Проверяем, существует ли путь (его родитель мог быть переименован,
        # но так как мы идем снизу вверх, такая ситуация исключена)
        if rename_item(path, prefix_to_remove):
            renamed_count += 1

    print(f"\n=== Готово! Успешно переименовано элементов: {renamed_count} ===")


if __name__ == "__main__":
    # Защита от прерываний (Ctrl+C)
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Программа прервана пользователем.")
        sys.exit(1)
