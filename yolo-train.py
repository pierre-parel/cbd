#!/usr/bin/env python3

from ultralytics import YOLO

model = YOLO("yolo11s.pt")
results = model.train(data="../datasets/cbd-detection/data.yaml", epochs=100, imgsz=640)
