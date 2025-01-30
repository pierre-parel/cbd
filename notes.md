# Personal Notes
Categories (for now):
- Fungus Damage (partial/severe)
- Insect Damage (partial/severe)
- Black (full/partial)
- Dried Cherry
- Floater/Fade
- Withered
- Broken/Shell
- Sour (Full/partial)

Removed:
- Immature
- Cut
- Parchement
- Husk
-
# Questions
1. What changes the shapes of the models?
2. Does the CrossCategoricalEntropy need a specific shape? If yes, why?
3. What does it mean to `batch` and `prefetch`?
4. What is the difference between **fine-tuning** and just using a **pre-trained model**?
Is fine-tuning necessary?
5. What does the EfficientNetV2B0 take in as input? What are its requirements? What about its shape?
6. Why do we need to do `tf.one_hot` for the labels?
7. See `examples/mnist.py`. Why do we need an `np.expand_dims()`? Does this allow us to answer issues above?
8. How to create a model that creates individual probabilities for each defect? (Possibility of 2 or more defects present in one bean)


# References

* [Webcam Integration](https://stackoverflow.com/questions/73459401/how-to-predict-test-a-trained-model-in-real-timewebcam)
* [Webcam Integration Example](https://github.com/mikkelmedm/Processing-Keras-InceptionV3-Webcam/blob/master/InceptionV3_Webcam.py)
* [Interfacing Python with Arduino](https://projecthub.arduino.cc/ansh2919/serial-communication-between-python-and-arduino-663756)

# Process

Feed first bean -> Python Program (OpenCV webcam + prediction) -> Serial Comm Prediction to Arduino -> Arduino processes logic -> Move motors

# Multi label classification

* Loss Function: Binary Crossentropy 
* Activation at output: Sigmoid
https://datascience.stackexchange.com/questions/65006/multi-label-classification-based-on-single-label-dataset