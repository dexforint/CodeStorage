from ultralytics import YOLO
import cv2

# 1. Загрузка модели для определения поз (Pose model)
# Будет автоматически скачана модель yolov8n-pose.pt (n - nano, самая быстрая)
# Можно использовать более точные: yolov8s-pose.pt, yolov8m-pose.pt и т.д.
# model = YOLO("yolov8n-pose5.pt")
model = YOLO("yolov8n-pose5_005.pt")

# 2. Путь к вашему изображению (замените на свой)
image_path = "image.png"

# 3. Выполнение предсказания
results = model(image_path)

# 4. Обработка результатов
for result in results:
    # Получение объекта с ключевыми точками
    keypoints = result.keypoints

    # Если на фото найдены люди
    if keypoints is not None:
        # Извлечение координат (X, Y) для каждого человека
        # Формат: [количество_людей, 17_точек, 2_координаты]
        coords = keypoints.xy[0]  # берем координаты первого человека
        print("Координаты точек первого человека:\n", coords)

        # Уверенность (confidence) для каждой точки
        confs = keypoints.conf[0]
        # print("Уверенность для каждой точки:\n", confs)

    # 5. Визуализация результата
    # Метод plot() автоматически рисует скелет поверх изображения
    annotated_frame = result.plot()

    # Показ изображения с помощью OpenCV
    cv2.imshow("YOLO Keypoints", annotated_frame)
    cv2.waitKey(0)  # Ждем нажатия любой клавиши
    cv2.destroyAllWindows()
