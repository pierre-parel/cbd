import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from tensorflow import keras
from keras.applications.efficientnet_v2 import EfficientNetV2S
import numpy as np

model = EfficientNetV2S(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)

# Freeze weights of the pre-trained model's layers for transfer learning
model.trainable = False

from keras.models import Sequential
from keras.layers import *

# Create custom model
my_model = Sequential([
    model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(8, activation="sigmoid")
])

# Import dataset
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 100

ds_train, ds_test = keras.utils.image_dataset_from_directory(
    "coffee_bean_train",
    validation_split=0.2,
    subset="both",
    seed=np.random.randint(100),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

import matplotlib.pyplot as plt

img_augmentation_layers = [
    RandomRotation(
        factor=0.1,
        fill_mode='constant',
        fill_value = 255),
    RandomTranslation(height_factor=0.1, width_factor=0.1),
    GaussianNoise(0.05),
    RandomContrast(0.1),
    RandomFlip()
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
    F1Score()
]


from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

my_model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(learning_rate=0.001),
    metrics=metrics,
)

from keras.callbacks import ModelCheckpoint, EarlyStopping

model_checkpoint_callbacks = ModelCheckpoint(
    filepath="saved_models/{epoch:02d}-{val_accuracy:.2f}-{val_loss:.2f}.keras",
    monitor="val_accuracy",
    mode="max",
    save_best_only = True
)

early_stopping = EarlyStopping(monitor="val_loss", patience=5)

history = my_model.fit(
    ds_train,
    epochs=EPOCHS,
    validation_data=ds_test,
    callbacks = [
        model_checkpoint_callbacks,
        early_stopping
    ]
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("accuracy.jpg")
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("loss.jpg")