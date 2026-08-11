import cv2
import numpy as np
from ultralytics import YOLO


def visualize_custom_pose(
    image_path,
    model_path,
    output_path="output.jpg",
    custom_skeleton=None,
    box_conf_threshold=0.5,
    kpt_conf_threshold=0.5,
    draw_kp_numbers=True,
):
    """
    Кастомная визуализация для YOLO Pose.

    :param image_path: Путь к исходному изображению
    :param model_path: Путь к весам кастомной модели (например, 'best.pt')
    :param output_path: Путь для сохранения результата
    :param custom_skeleton: Список кортежей связей, например [(0,1), (1,2)]. Если None - линии не рисуются.
    :param box_conf_threshold: Порог уверенности для объекта в целом (bounding box)
    :param kpt_conf_threshold: Порог уверенности (видимости) для каждой отдельной точки
    :param draw_kp_numbers: Рисовать ли индексы (0, 1, 2...) рядом с точками для отладки
    """
    # 1. Загрузка модели и инференс
    model = YOLO(model_path)
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    # Делаем предсказание
    results = model(img)[0]

    # Если ничего не найдено, сохраняем оригинал и выходим
    if len(results.boxes) == 0:
        print("На изображении ничего не найдено.")
        cv2.imwrite(output_path, img)
        return

    # 2. Извлечение данных в массивы NumPy
    # Координаты рамок [x1, y1, x2, y2]
    boxes = results.boxes.xyxy.cpu().numpy()
    # Уверенность детекции
    box_confs = results.boxes.conf.cpu().numpy()
    # Классы детекции (если класс один, там будут нули)
    classes = results.boxes.cls.cpu().numpy()

    # Ключевые точки. Формат тензора: (Кол-во_объектов, Кол-во_точек, 3)
    # Последняя размерность это: [x, y, confidence/visibility]
    keypoints = results.keypoints.data.cpu().numpy()

    # 3. Отрисовка
    for i in range(len(boxes)):
        # Фильтруем объекты по уверенности рамки
        if box_confs[i] < box_conf_threshold:
            continue

        # --- А) Рисуем Bounding Box и уверенность детекции ---
        x1, y1, x2, y2 = map(int, boxes[i])

        # Рисуем зеленую рамку (B, G, R)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Подготавливаем текст: "Class 0: 0.85"
        label = f"Obj {int(classes[i])}: {box_confs[i]:.2f}"

        # Рисуем подложку под текст (для читаемости)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - 25), (x1 + text_w, y1), (0, 255, 0), -1)
        # Пишем текст
        cv2.putText(
            img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

        # --- Б) Рисуем Ключевые Точки ---
        kpts = keypoints[i]  # Точки конкретного (i-го) объекта

        for kpt_idx, kpt in enumerate(kpts):
            x, y, kpt_conf = kpt

            # Если точка "видна" (уверенность выше порога)
            if kpt_conf > kpt_conf_threshold:
                x_int, y_int = int(x), int(y)

                # Рисуем красную точку
                cv2.circle(img, (x_int, y_int), 5, (0, 0, 255), -1)

                # Рисуем номер точки (полезно для сборки скелета)
                if draw_kp_numbers:
                    cv2.putText(
                        img,
                        str(kpt_idx),
                        (x_int + 5, y_int - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

        # --- В) Рисуем "Скелет" (связи между точками) ---
        if custom_skeleton is not None:
            for pt1_idx, pt2_idx in custom_skeleton:
                # Проверка, чтобы не выйти за пределы массива точек
                if pt1_idx < len(kpts) and pt2_idx < len(kpts):
                    x1_k, y1_k, conf1 = kpts[pt1_idx]
                    x2_k, y2_k, conf2 = kpts[pt2_idx]

                    # Линия рисуется только если ОБЕ точки уверены
                    if conf1 > kpt_conf_threshold and conf2 > kpt_conf_threshold:
                        # Рисуем синюю линию
                        cv2.line(
                            img,
                            (int(x1_k), int(y1_k)),
                            (int(x2_k), int(y2_k)),
                            (255, 0, 0),
                            2,
                        )

    # 4. Сохранение / Вывод
    cv2.imwrite(output_path, img)
    print(f"Готово! Результат сохранен в {output_path}")

    # Если хочешь, чтобы картинка открылась в окне во время работы скрипта, раскомментируй:
    # cv2.imshow("Custom Pose", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# ==========================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ==========================================
if __name__ == "__main__":

    # Допустим, твоя кастомная модель имеет 5 точек.
    # Ты хочешь соединить 0-ю с 1-й, 1-ю со 2-й, 2-ю с 3-й и т.д.
    # Настрой этот список под свою логику! (индексы начинаются с 0)
    MY_SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]

    # Если скелет не нужен, передай MY_SKELETON = None

    model_path = "yolov8n-pose5_0.018.pt"

    visualize_custom_pose(
        image_path="image.png",  # Путь к картинке для теста
        model_path=model_path,  # Путь к твоей обученной модели
        output_path=f"{model_path}.jpg",  # Куда сохранить
        custom_skeleton=MY_SKELETON,
        box_conf_threshold=0.5,  # Показываем детекции увереннее 50%
        kpt_conf_threshold=0.6,  # Рисуем точку, если модель уверена в ней на 60%
        draw_kp_numbers=True,  # Отрисовка циферок рядом с точками
    )
