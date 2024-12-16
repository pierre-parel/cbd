import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
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
    label_mode="categorical"
)

import matplotlib.pyplot as plt
import numpy as np
# TEMPORARY
# img_augmentation_layers = [
#     keras.layers.RandomRotation(factor=0.15),
#     keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
#     keras.layers.RandomFlip(),
#     keras.layers.RandomContrast(factor=0.1),
# ]

# def img_augmentation(images):
#     for layer in img_augmentation_layers:
#         images = layer(images)
#     return images

# for image, label in ds_train.take(1):  # Get a batch
#     single_image = image[0]  # Take the first image from the batch
#     single_label = label[0]  # Take the corresponding label
#     for i in range(9):  # Generate and display 9 augmented versions
#         ax = plt.subplot(3, 3, i + 1)
#         aug_img = img_augmentation(np.expand_dims(single_image.numpy(), axis=0))  # Augment the single image
#         plt.imshow(aug_img[0].numpy().astype("uint8"))  # Convert tensor to image
#         plt.title("Class: {}".format(np.argmax(single_label.numpy())))  # Display label
#         plt.axis("off")
# plt.show()
# plt.savefig("augmented_images.png")

# import tensorflow as tf
# NUM_CLASSES=17

# def input_preprocess_train(image, label):
#     image = tf.image.resize(image, (224, 224))  # Ensure the image is resized to (224, 224)
#     image = img_augmentation(image)  # Apply augmentations
#     label = tf.one_hot(tf.cast(label, tf.int32), NUM_CLASSES)  # Cast label to int32
#     return image, label

# def input_preprocess_test(image, label):
#     image = tf.image.resize(image, (224, 224))  # Ensure the image is resized to (224, 224)
#     label = tf.one_hot(tf.cast(label, tf.int32), NUM_CLASSES)  # Cast label to int32
#     return image, label

# # Updated dataset mapping
# ds_train = ds_train.map(input_preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
# ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# ds_test = ds_test.map(input_preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)

# for image_batch, label_batch in ds_train.take(1):
#     print(image_batch.shape)  # Should print (BATCH_SIZE, 224, 224, 3)
#     print(label_batch.shape)  # Should print (BATCH_SIZE, NUM_CLASSES)
# END OF TEMPORARY

model = keras.applications.EfficientNetV2B0(
    weights=None,
    input_shape=(224, 224, 3),
    classes=17
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="checkpoints/model_at_epoch_{epoch}.keras"),
]

model.fit(
    ds_train,
    epochs=EPOCHS,
    validation_data=ds_test,
    callbacks=callbacks,
)