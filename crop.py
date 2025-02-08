from ultralytics import YOLO
import os
import glob
import cv2

model = YOLO(model="./best.pt")
names = model.names

CROPPED_DIR = "yolo_cropped"
if not os.path.exists(CROPPED_DIR):
    os.mkdir(CROPPED_DIR)


classes = ["Black", "Broken", "Dried_Cherry", "Floater", "Fungus_Damage", "Good", "Insect_Damage", "Sour"]

for classname in classes:
    idx = 0
    PATH = "../../Downloads/GCB_Datset_Yolo/"
    for filename in glob.glob(os.path.join(PATH, classname, "*.jpg")):
        print(os.path.join(PATH, classname, "*.jpg"))
        im = cv2.imread(filename)
        results = model.predict(filename, show=False, conf=0.98)
        boxes = results[0].boxes.xyxy.cpu().tolist()

        if boxes is not None:
            for box in boxes:
                idx += 1
                crop_im = im[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
                CROPPED_PATH = "../../Downloads/GCB_Dataset_Cropped/"
                cv2.imwrite(os.path.join(CROPPED_PATH, classname, str(idx) + ".png"), crop_im)
                print(f"Created file: {os.path.join(CROPPED_PATH, classname, str(idx) + '.png')}")
