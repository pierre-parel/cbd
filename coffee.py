import numpy as np
import os
import matplotlib.pyplot as plt
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf

image_size = (224, 224)
batch_size = 128
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "coffee_bean",
    validation_split=0.2,
    subset="both",
    seed=727,
    image_size=image_size,
    batch_size=batch_size
)


print("train_ds shape:", train_ds.shape)
print("val_ds shape:", val_ds.shape)
print(train_ds.shape[0], "train samples")
print(val_ds.shape[0], "test samples")

model = keras.applications.EfficientNetV2B0(
    include_top=False,
)

model.summary()

epochs = 20

callbacks = [
    keras.callbacks.ModelCheckpoint("model_at_epoch_{epoch}.keras"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]

model.compile(
    optimizer = keras.optimizers.Adam(1e-3),
    loss = keras.losses.CategoricalCrossentropy(),
    metrics = [
        keras.metrics.SparseCategoricalAccuracy(name="acc")
    ]
)

model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)