import shutil
from pathlib import Path
import random


def copy_random_images():
    # Запрашиваем данные у пользователя
    src_dir_input = input(
        "Введите путь к папке, из которой будут браться изображения: "
    )
    count_input = input("Введите количество изображений для копирования: ")
    dst_dir_input = input(
        "Введите путь к папке, в которую будут сохраняться изображения: "
    )

    # Проверяем корректность ввода количества
    try:
        img_count = int(count_input)
        if img_count <= 0:
            print("Ошибка: Количество изображений должно быть положительным числом.")
            return
    except ValueError:
        print("Ошибка: Введено некорректное число для количества изображений.")
        return

    # Создаем объекты Path
    src_path = Path(src_dir_input)
    dst_path = Path(dst_dir_input)

    # Проверяем существование исходной папки
    if not src_path.is_dir():
        print(
            f"Ошибка: Исходная папка '{src_dir_input}' не существует или не является директорией."
        )
        return

    # Создаем целевую папку, если её нет
    dst_path.mkdir(parents=True, exist_ok=True)
    print(f"Целевая папка: '{dst_path}' (создана/проверена)\n")

    # Расширения файлов, которые считаем изображениями
    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
        ".svg",
    }

    # Шаг 1: Собираем все изображения в список
    print("Поиск изображений в папке и подпапках...")
    all_images = []
    for file_path in src_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            all_images.append(file_path)

    # Если изображения не найдены
    if not all_images:
        print("В указанной папке и её подпапках изображения не найдены.")
        return

    # Шаг 2: Проверяем, достаточно ли изображений найдено
    actual_count = min(img_count, len(all_images))
    if img_count > len(all_images):
        print(
            f"Внимание: Запрошено {img_count}, но найдено всего {len(all_images)}. Будут скопированы все найденные."
        )

    # Шаг 3: Выбираем случайные уникальные элементы
    # random.sample гарантирует выбор без повторений
    selected_images = random.sample(all_images, actual_count)

    # Шаг 4: Копируем выбранные изображения
    copied_count = 0
    for file_path in selected_images:
        destination_file = dst_path / file_path.name

        # Защита от перезаписи, если в целевой папке уже есть файл с таким же именем
        # (например, если скрипт запускается несколько раз в одну и ту же папку)
        if destination_file.exists():
            counter = 1
            while True:
                new_name = f"{file_path.stem}_random_{counter}{file_path.suffix}"
                destination_file = dst_path / new_name
                if not destination_file.exists():
                    break
                counter += 1

        try:
            shutil.copy2(file_path, destination_file)
            copied_count += 1
            print(f"[{copied_count}/{actual_count}] Скопирован: {file_path.name}")
        except Exception as e:
            print(f"Ошибка при копировании файла {file_path.name}: {e}")

    # Итоговый вывод
    print(f"\nУспешно скопировано {copied_count} случайных изображений.")


if __name__ == "__main__":
    copy_random_images()
