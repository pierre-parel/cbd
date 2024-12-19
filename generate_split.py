import tensorflow as tf
from tensorflow import keras
import os
from sklearn.model_selection import train_test_split

DATASET_DIR = "coffee_bean"  
TRAIN_DIR = "coffee_bean_train"  
TEST_DIR = "coffee_bean_test"  

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
SEED = 727

image_paths = []
labels = []
class_names = sorted(os.listdir(DATASET_DIR))

for class_name in class_names:
    class_dir = os.path.join(DATASET_DIR, class_name)
    if os.path.isdir(class_dir):
        for file in os.listdir(class_dir):
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
                image_paths.append(os.path.join(class_dir, file))
                labels.append(class_name)

train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=VALIDATION_SPLIT, stratify=labels, random_state=SEED
)

def save_images(file_paths, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_path, label in zip(file_paths, labels):
        class_dir = os.path.join(output_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        output_path = os.path.join(class_dir, os.path.basename(file_path))
        tf.keras.utils.save_img(output_path, tf.keras.utils.load_img(file_path, target_size=IMG_SIZE))

save_images(train_paths, train_labels, TRAIN_DIR)
save_images(test_paths, test_labels, TEST_DIR)

print(f"Training data saved in: {TRAIN_DIR}")
print(f"Test data saved in: {TEST_DIR}")
