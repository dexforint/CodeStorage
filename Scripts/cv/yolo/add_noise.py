import torch
from ultralytics import YOLO


def add_noise_to_yolo_pose(model_path, save_path, noise_std=0.01, num_last_params=10):
    """
    Добавляет гауссовский шум к весам последних слоев модели YOLO.

    :param model_path: Путь к исходной модели (например, 'yolov8n-pose.pt')
    :param save_path: Путь для сохранения новой модели
    :param noise_std: Размер шума (стандартное отклонение нормального распределения)
    :param num_last_params: К скольким последним тензорам параметров добавить шум
    """
    print(f"Загрузка модели из {model_path}...")
    model = YOLO(model_path)

    # Получаем доступ к самой PyTorch модели внутри обертки Ultralytics
    pytorch_model = model.model

    # Извлекаем все параметры (веса и смещения) в виде списка
    parameters = list(pytorch_model.named_parameters())
    total_params = len(parameters)
    print(f"Всего обучаемых тензоров (параметров) в модели: {total_params}")

    # Защита от выхода за пределы списка
    num_last_params = min(num_last_params, total_params)

    # Срез последних N параметров
    params_to_modify = parameters[-num_last_params:]

    print(
        f"\nДобавляем шум (std={noise_std}) к последним {num_last_params} тензорам..."
    )

    # Отключаем отслеживание градиентов, так как мы вмешиваемся в веса напрямую
    with torch.no_grad():
        for name, param in params_to_modify:
            # torch.randn_like создает тензор с нормальным (гауссовским) распределением
            # того же размера и на том же устройстве (CPU/GPU), что и оригинальные веса
            noise = torch.randn_like(param) * noise_std

            # Добавляем шум к весам (операция add_ с подчеркиванием изменяет тензор in-place)
            param.add_(noise)

            print(f" -> Шум добавлен к: {name} (размерность: {list(param.shape)})")

    # Сохраняем измененную модель
    model.save(save_path)
    print(f"\nГотово! Модель с шумом сохранена как: {save_path}")


# ==========================================
# ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
# ==========================================
if __name__ == "__main__":
    # 1. Модель (если файла нет локально, ultralytics скачает его автоматически)
    INPUT_MODEL_PATH = "yolov8n-pose5.pt"

    # 3. Размер шума (чем больше, тем сильнее искажения)
    NOISE_MAGNITUDE = 0.018

    # 2. Имя файла для сохранения измененной модели
    OUTPUT_MODEL_PATH = f"yolov8n-pose5_{NOISE_MAGNITUDE}.pt"

    # 4. Количество "слоев" с конца, к которым добавляется шум
    # ВАЖНО: В PyTorch один слой (например, Conv2d) обычно состоит из 2 параметров: weight (веса) и bias (смещение).
    # Поэтому, если вы хотите изменить 5 последних архитектурных слоев,
    # вам, вероятно, нужно указать здесь 10.
    LAYERS_TO_NOISE = 6

    # Запуск функции
    add_noise_to_yolo_pose(
        model_path=INPUT_MODEL_PATH,
        save_path=OUTPUT_MODEL_PATH,
        noise_std=NOISE_MAGNITUDE,
        num_last_params=LAYERS_TO_NOISE,
    )
