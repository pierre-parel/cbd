from ultralytics import YOLO

model = YOLO("yolo11s-cls.pt")
results = model.train(data="../datasets/cbd-classification/", epochs=100, imgsz=640, patience=5)
