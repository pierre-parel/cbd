import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from tensorflow import keras

from keras.saving import load_model

loaded_model = load_model("./saved_models/60-0.93.keras") # Replace with model to be tested.

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 100
ds_train, ds_test = keras.utils.image_dataset_from_directory(
    "coffee_bean_test",
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

loaded_model.evaluate(
    ds_train
)
loaded_model.evaluate(
    ds_test
)

# Output ranges from 0-16, corresponding to each class (in alphabetical order)
# 0 = Broken
# 1 = Cut
# 2 = Dry Cherry
# 3 = Fade
# 4 = Floater
# 5 = Full Black
# 6 = Full Sour
# 7 = Fungus Damage
# 8 = Husk
# 9 = Immature
# 10 = Parchment
# 11 = Partial Black
# 12 = Partial Sour
# 13 = Severe Insect Damage
# 14 = Shell
# 15 = Slight Insect Damage
# 16 = Withered