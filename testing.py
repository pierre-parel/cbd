import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from tensorflow import keras

from keras.saving import load_model
loaded_model = load_model("saved_models/51-0.92-0.22.keras")

IMG_SIZE = (224, 224)
ds_test = keras.utils.image_dataset_from_directory(
    "coffee_bean_test",
    seed=1337,
    shuffle=False,
    image_size=IMG_SIZE,
    label_mode="categorical",
)

class_names = ds_test.class_names

for images, labels in ds_test:
    predictions = loaded_model.predict(images)
    print(predictions)
    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels = np.argmax(labels, axis=1)
    
    for i in range(len(images)):
        if predicted_labels[i] != actual_labels[i]:
            plt.imshow(images[i].numpy().astype("uint8"))
            title = f"Predicted: {class_names[predicted_labels[i]]}, Actual: {class_names[actual_labels[i]]}"
            plt.title(title)
            plt.axis("off")
            plt.savefig(f"{title}.jpg")