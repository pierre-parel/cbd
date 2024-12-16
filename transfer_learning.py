import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from tensorflow import keras
from keras.applications.efficientnet_v2 import EfficientNetV2B0

model = EfficientNetV2B0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
    classes=17
)

model.summary()

# Freeze weights of the EfficientNetV2B0 layers for transfer learning
model.trainable = False

from keras.models import Sequential
from keras.layers import *

# Create custom model
my_model = Sequential([
    model,
    Conv2D(1024, 3, 1, activation="relu"),
    GlobalAveragePooling2D(),
    Dense(1024, activation="relu"),
    Dropout(0.2),
    Dense(1024, activation="relu"),
    Dropout(0.2),
    Dense(17, activation="softmax")
])

my_model.summary()

# Import dataset
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 100
ds_train, ds_test = keras.utils.image_dataset_from_directory(
    "coffee_bean",
    validation_split=0.2,
    subset="both",
    seed=727,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

import matplotlib.pyplot as plt
import numpy as np

img_augmentation_layers = [
    RandomRotation(factor=0.15),
    RandomTranslation(height_factor=0.1, width_factor=0.1),
    RandomFlip(),
    RandomContrast(factor=0.1),
]

def img_augmentation(images):
    for layer in img_augmentation_layers:
        images = layer(images)
    return images

def preprocess_input(image, label):
    image = img_augmentation(image)
    return (image, label)

ds_train = ds_train.map(preprocess_input, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

from keras.metrics import *
metrics = [
    "accuracy",
    Precision(),
    Recall(),
]
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

my_model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(learning_rate=0.001),
    metrics=metrics,
)

from keras.callbacks import EarlyStopping
my_model.fit(
    ds_train,
    epochs=EPOCHS,
    validation_data=ds_test,
)