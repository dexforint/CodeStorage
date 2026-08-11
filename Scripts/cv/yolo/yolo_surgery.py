import torch
from ultralytics import YOLO

KEEP = [0, 5, 6, 11, 12]
idx = torch.tensor([k * 3 + j for k in KEEP for j in range(3)])  # 15 каналов из 51

m = YOLO("yolov8n-pose.pt")
head = m.model.model[-1]  # Pose head

for seq in head.cv4:  # cv4 — ветки kpt-регрессии (по одной на масштаб)
    conv = seq[-1]  # последний Conv2d 1x1
    new = torch.nn.Conv2d(conv.in_channels, len(idx), 1)
    new.weight.data = conv.weight.data[idx].clone()
    new.bias.data = conv.bias.data[idx].clone()
    seq[-1] = new

head.kpt_shape = (5, 3)
head.nk = 15
m.model.kpt_shape = (5, 3)
m.model.yaml["kpt_shape"] = [5, 3]

m.save("yolov8n-pose5.pt")
