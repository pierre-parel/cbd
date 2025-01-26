import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from tensorflow import keras

from keras.saving import load_model
loaded_model = load_model("./saved_models/97-0.98-0.07.keras") # Replace with model to be tested.

IMG_SIZE = (224, 224)
ds_test = keras.utils.image_dataset_from_directory(
    "coffee_bean_test",
    seed=1337,
    shuffle=False,
    image_size=IMG_SIZE,
    label_mode="categorical",
)

evaluate = loaded_model.evaluate(ds_test)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from time import perf_counter

start = perf_counter()
y_pred_probs = loaded_model.predict(ds_test)
end = perf_counter()
print("Elapsed time: ", (end-start)*1000, "ms")

y_pred = np.argmax(y_pred_probs, axis=1)

y_true = np.concatenate([np.argmax(y, axis=1) for _, y in ds_test])

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(12, 12))  
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ds_test.class_names)
disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
plt.xticks(rotation=45)
plt.savefig("confusion_matrix.jpg")
