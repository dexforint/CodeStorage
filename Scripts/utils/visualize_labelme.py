import os
import json
import glob
import cv2


def visualize_detections(images_dir, annotations_dir, output_dir):
    """
    Читает аннотации LabelMe, отрисовывает боксы на изображениях и сохраняет результат.

    :param images_dir: Путь к папке с исходными изображениями
    :param annotations_dir: Путь к папке с JSON аннотациями LabelMe
    :param output_dir: Путь к папке для сохранения визуализированных изображений
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ищем все JSON файлы в папке аннотаций
    json_files = glob.glob(os.path.join(annotations_dir, "*.json"))

    if not json_files:
        print(f"В папке {annotations_dir} не найдено JSON файлов.")
        return

    print(f"Найдено {len(json_files)} аннотаций. Начинаем визуализацию...")

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Ошибка чтения JSON: {json_path}. Пропуск.")
                continue

        # LabelMe хранит имя картинки в поле imagePath. Берем только имя файла.
        image_filename = os.path.basename(data.get("imagePath", ""))
        if not image_filename:
            # Если imagePath пуст, пробуем взять имя JSON файла и добавить расширения
            base_name = os.path.splitext(os.path.basename(json_path))[0]
        else:
            base_name = os.path.splitext(image_filename)[0]

        # Ищем картинку с таким же именем (проверяя разные расширения)
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp"]:
            candidate = os.path.join(images_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"Изображение для {base_name} не найдено в {images_dir}. Пропуск.")
            continue

        # Загружаем изображение через OpenCV
        image = cv2.imread(img_path)
        if image is None:
            print(f"Не удалось загрузить изображение {img_path}. Пропуск.")
            continue

        # Проходимся по всем фигурам (боксам) в аннотации
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "rectangle":
                continue

            label = shape.get("label", "unknown")
            points = shape.get("points", [])
            if len(points) < 2:
                continue

            # Координаты углов бокса
            x1, y1 = int(points[0][0]), int(points[0][1])
            x2, y2 = int(points[1][0]), int(points[1][1])

            # Выбор цвета и текста в зависимости от класса
            if label == "standing":
                color = (0, 255, 0)  # Зеленый (BGR для OpenCV)
            elif label == "not-standing":
                color = (0, 0, 255)  # Красный (BGR для OpenCV)
            else:
                color = (0, 255, 255)  # Желтый для неизвестных классов

            # Рисуем прямоугольник (толщина 3 пикселя)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

            # Подготовка фона для текста, чтобы он был читаемым
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            # Сдвигаем фон немного над боксом
            y_bg = max(0, y1 - text_h - baseline - 5)
            cv2.rectangle(image, (x1, y_bg), (x1 + text_w, y1), color, -1)

            # Рисуем сам текст (черный цвет на цветном фоне)
            cv2.putText(
                image,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Сохраняем результат
        out_img_name = base_name + "_visualized.jpg"
        out_img_path = os.path.join(output_dir, out_img_name)
        cv2.imwrite(out_img_path, image)

    print(f"\nГотово! Визуализированные изображения сохранены в: {output_dir}")


# ================= ЗАПУСК =================
if __name__ == "__main__":
    # Укажите ваши пути:
    IMAGES_DIR = input("Папка с исходными картинками")
    ANNOTATIONS_DIR = input("Папка с JSON от предыдущего скрипта")  #
    OUTPUT_DIR = input("Куда сохранить картинки с боксами")  #

    visualize_detections(IMAGES_DIR, ANNOTATIONS_DIR, OUTPUT_DIR)
