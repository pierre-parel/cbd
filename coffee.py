import numpy as np
import os
import matplotlib.pyplot as plt
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
ds_train, ds_test = keras.utils.image_dataset_from_directory(
    "coffee_bean",
    validation_split=0.2,
    subset="both",
    seed=727,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

model = keras.applications.Xception(
    weights=None,
    input_shape=(224, 224, 3),
    classes=17
)

model.summary()

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy'
)

model.fit(
    ds_train,
    epochs=20,
    validation_data=ds_test
)