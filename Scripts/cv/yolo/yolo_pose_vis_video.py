import cv2
from ultralytics import YOLO
from tqdm.auto import tqdm


def draw_custom_pose(
    img,
    results,
    custom_skeleton=None,
    box_conf_threshold=0.5,
    kpt_conf_threshold=0.5,
    draw_kp_numbers=True,
):
    """Отрисовка боксов, ключевых точек и скелета на одном кадре."""
    if results.boxes is None or len(results.boxes) == 0:
        return img

    boxes = results.boxes.xyxy.cpu().numpy()
    box_confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    keypoints = (
        results.keypoints.data.cpu().numpy() if results.keypoints is not None else None
    )

    for i in range(len(boxes)):
        if box_confs[i] < box_conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"Obj {int(classes[i])}: {box_confs[i]:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - 25), (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(
            img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

        if keypoints is None:
            continue

        kpts = keypoints[i]

        for kpt_idx, kpt in enumerate(kpts):
            x, y, kpt_conf = kpt
            if kpt_conf > kpt_conf_threshold:
                x_int, y_int = int(x), int(y)
                cv2.circle(img, (x_int, y_int), 5, (0, 0, 255), -1)

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

        if custom_skeleton is not None:
            for pt1_idx, pt2_idx in custom_skeleton:
                if pt1_idx < len(kpts) and pt2_idx < len(kpts):
                    x1_k, y1_k, conf1 = kpts[pt1_idx]
                    x2_k, y2_k, conf2 = kpts[pt2_idx]
                    if conf1 > kpt_conf_threshold and conf2 > kpt_conf_threshold:
                        cv2.line(
                            img,
                            (int(x1_k), int(y1_k)),
                            (int(x2_k), int(y2_k)),
                            (255, 0, 0),
                            2,
                        )

    return img


def visualize_custom_pose_video(
    video_path,
    model_path,
    output_path="output.mp4",
    custom_skeleton=None,
    box_conf_threshold=0.5,
    kpt_conf_threshold=0.5,
    draw_kp_numbers=True,
    show_preview=False,
):
    """
    Кастомная визуализация YOLO Pose для видео.

    :param video_path: Путь к исходному видео
    :param model_path: Путь к весам модели (например, 'best.pt')
    :param output_path: Путь для сохранения результата (.mp4 / .avi)
    :param custom_skeleton: Список связей, например [(0, 1), (1, 2)]. None — без линий
    :param box_conf_threshold: Порог уверенности bounding box
    :param kpt_conf_threshold: Порог уверенности ключевой точки
    :param draw_kp_numbers: Рисовать индексы точек
    :param show_preview: Показывать кадры в окне во время обработки
    """
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Не удалось создать выходное видео: {output_path}")

    try:
        pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Обработка")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)[0]
            annotated = draw_custom_pose(
                frame,
                results,
                custom_skeleton=custom_skeleton,
                box_conf_threshold=box_conf_threshold,
                kpt_conf_threshold=kpt_conf_threshold,
                draw_kp_numbers=draw_kp_numbers,
            )

            writer.write(annotated)

            if show_preview:
                cv2.imshow("Custom Pose", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            pbar.update(1)
        pbar.close()
    finally:
        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    print(f"Готово! Результат сохранен в {output_path}")


if __name__ == "__main__":
    MY_SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]

    visualize_custom_pose_video(
        video_path="./data/test_video2.mp4",
        model_path="./data/yolov11n-pose5_0.003.pt",
        output_path="./data/results/test_video2_pose.mp4",
        custom_skeleton=MY_SKELETON,
        box_conf_threshold=0.4,
        kpt_conf_threshold=0.4,
        draw_kp_numbers=True,
        show_preview=False,
    )
