from ultralytics import YOLO

model_path = "data/yolov8n-pose5_v2.pt"

model = YOLO(model_path)
model.export(
    format="engine",  # TensorRT
    half=True,
    imgsz=640,  # зафиксируется в движке
    device=0,
    # int8=True,
    nms=True,  # NMS внутри engine (новые версии Ultralytics)
)
