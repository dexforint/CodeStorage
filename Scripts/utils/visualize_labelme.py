import os
import json
import glob
import cv2


def visualize_detections(images_dir, annotations_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

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

        image_filename = os.path.basename(data.get("imagePath", ""))
        if not image_filename:
            base_name = os.path.splitext(os.path.basename(json_path))[0]
        else:
            base_name = os.path.splitext(image_filename)[0]

        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp"]:
            candidate = os.path.join(images_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"Изображение для {base_name} не найдено. Пропуск.")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Не удалось загрузить изображение {img_path}. Пропуск.")
            continue

        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "rectangle":
                continue

            label = shape.get("label", "unknown")
            description = shape.get(
                "description", ""
            )  # Читаем уверенность из description
            points = shape.get("points", [])
            if len(points) < 2:
                continue

            x1, y1 = int(points[0][0]), int(points[0][1])
            x2, y2 = int(points[1][0]), int(points[1][1])

            # Формируем итоговый текст: "класс (уверенность)"
            if description:
                display_text = f"{label} ({description})"
            else:
                display_text = label

            # Выбор цвета
            if label == "standing":
                color = (0, 255, 0)  # Зеленый
            elif label == "not-standing":
                color = (0, 0, 255)  # Красный
            else:
                color = (0, 255, 255)  # Желтый

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

            # Подготовка фона для текста
            (text_w, text_h), baseline = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            y_bg = max(0, y1 - text_h - baseline - 5)
            cv2.rectangle(image, (x1, y_bg), (x1 + text_w, y1), color, -1)

            # Рисуем текст (черный цвет на цветном фоне)
            cv2.putText(
                image,
                display_text,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        out_img_name = base_name + "_visualized.jpg"
        out_img_path = os.path.join(output_dir, out_img_name)
        cv2.imwrite(out_img_path, image)

    print(f"\nГотово! Визуализированные изображения сохранены в: {output_dir}")


# ================= ЗАПУСК =================
if __name__ == "__main__":
    IMAGES_DIR = input("Images dir: ")
    ANNOTATIONS_DIR = input("Labelme dir: ")
    OUTPUT_DIR = input("Output dir: ")

    visualize_detections(IMAGES_DIR, ANNOTATIONS_DIR, OUTPUT_DIR)
