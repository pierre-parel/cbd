import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from tensorflow import keras

from keras.saving import load_model

loaded_model = load_model("./saved_models/95-0.94.keras") # Replace with model to be tested.

IMG_SIZE = (224, 224)
ds_test = keras.utils.image_dataset_from_directory(
    "coffee_bean_test",
    seed=1337,
    image_size=IMG_SIZE,
    label_mode="categorical",
)

loaded_model.evaluate(
    ds_test
)
