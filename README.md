# Green Coffee Bean Defect Detection 

A deep learning project to classify 17 types of green coffee bean defects using EfficientNetV2S and transfer learning, achieving high accuracy and precision.

## Model Overview

- Developed a deep learning model using **EfficientNetV2S** with **transfer learning** to classify 17 types of green coffee bean defects.
- Achieved **93.21% validation accuracy** and **93.87% validation precision**.
- Applied advanced **image augmentation** techniques such as random rotation, translation, and contrast adjustment to enhance generalization and reduce overfitting.
- Trained the model with a **validation loss of 0.1952**, using **categorical cross-entropy** loss and the **Adam optimizer**, fine-tuned over 100 epochs for reliable defect detection.

## File Descriptions

### `evaluate.py`
This script evaluates a pre-trained model's performance on training and test datasets. It uses a saved model to classify 17 classes of green coffee bean defects and outputs the evaluation metrics. The script includes dataset preparation and ensures compatibility with TensorFlow and Keras frameworks.

### `generate_augmented_images.py`
Generates additional training data through image augmentation techniques such as rotation at various angles. Processes all images in the training dataset folder and saves the augmented images in the same directory. Useful for improving model generalization and reducing overfitting.

### `generate_split.py`
Splits the dataset into training and testing subsets with a specified validation split and ensures stratified sampling based on class labels. Saves the organized datasets into separate directories for easy access during training and evaluation. Handles image resizing and directory setup.

### `transfer_learning.py`
Implements transfer learning using EfficientNetV2S as the base model, fine-tuned to classify green coffee bean defects. Applies data augmentation during preprocessing and trains the model with categorical cross-entropy loss and Adam optimizer. Includes checkpoints for saving the best-performing model and generates accuracy/loss plots.

### `run.sh`
Automates the complete workflow, including environment setup, dataset preparation, augmentation, model training, and evaluation. Unzips the dataset, organizes it, performs augmentations, trains the model, and evaluates its performance. Ensures a streamlined end-to-end pipeline for running the project.

### `clean.sh`
Removes all generated files and directories, including augmented datasets, models, and plots, to reset the project environment. Prepares the setup for a clean rerun of the workflow.

## Setting Up the Environment 
Tensorflow with GPU access is no longer supported past **TensorFlow 2.10**. Hence, it is recommended to use WSL2. The instructions below are from [NVIDIA's setup docs for CUDA in WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
1. Launch your preferred Windows Terminal/Command Prompt/Powershell and install WSL:
```sh
wsl --install
```
2. Ensure you have the latest WSL kernel
```sh
wsl --update
```
3. Setup your Linux user info by providing a username and password.
> In the future, you can open WSL by typing `wsl` in the search bar in the Windows start menu.

## Enabling CUDA for WSL2
1. Remove the old GPG key:
```sh
sudo apt-key del 7fa2af80
```
2. Run the installation instructions one by one as indicated on the [CUDA download page for WSL-Ubuntu](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

> **NOTE:** It is important to check if your system meets the hardware, software, and system requirements to run TensorFlow. Please check in the [official TensorFlow documentation](https://www.tensorflow.org/install/pip#system_requirements) for more details.

## Getting Started
1. Clone the repository
```sh
git clone https://github.com/pierre-parel/cbd.git
cd cbd
```
2. Create virtual environment and install required modules
```sh
sudo apt install python3 python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Download [Coffee Green Bean with 17 Defects](https://www.kaggle.com/datasets/sujitraarw/coffee-green-bean-with-17-defects-original)
3. Unzip to `coffee_bean` folder. Your directory structure shoud look like:
```
cbd/
    .venv/
    coffee_bean/
        Broken/
        Cut/
        Dry Cherry/
        ...
    examples/
    .py files
```
4. Generate the train/test split by running:
```sh
generate_split.py
```
4. Generate the augmented dataset using the following command:
```sh
python generate_augmented_images.py
```
5. Train the model using the command:
```sh
python transfer_learning.py
```
6. Evaluate the model using the commandL
```sh
python evaluate.py
```
TODO
------
- [X] Use GPU for faster training. See [Install TensorFlow with pip](https://www.tensorflow.org/install/pip#windows-wsl2_1) and [Use a GPU](https://www.tensorflow.org/guide/gpu)
- [X] Change from Xception to EfficientNetV2B0. 
- [X] Use different pretrained model for comparison
- [X] Data augmentation 
- [X] Prefetch dataset 
- [ ] Fine-tuning of the model
- [ ] Experiment with different layers in my_model
- [ ] Try using other image augmentation techniques


References
------
- [Image Classification from Scratch](https://keras.io/examples/vision/image_classification_from_scratch/)
- [EfficientNetV2B0 function](https://keras.io/api/applications/efficientnet_v2/#efficientnetv2b0-function)
- [Load image dataset from directory](https://keras.io/api/data_loading/image/#imagedatasetfromdirectory-function)
- [Image classification via fine-tuning with EfficientNet](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/)
- [Transfer learning & fine tuning](https://keras.io/guides/transfer-learning/)
- [Losses](https://keras.io/api/losses/)
- [Rohan & Lenny #2: Convolutional Neural Networks](https://ayearofai.com/rohan-lenny-2-convolutional-neural-networks-5f4cd480a60b)
- [The Deeplearning Book](https://www.deeplearningbook.org/)
- [Convolution Neural Networks Course by Andrew Ng](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)
- [Deep Learning by 3Blue1Brown](https://www.youtube.com/playlist?list=PLLMP7TazTxHrgVk7w1EKpLBIDoC50QrPS)
- [Understanding Deep Learning Book](https://udlbook.github.io/udlbook/)
- [The Principles of Deep Learning Theory](https://deeplearningtheory.com/)
- [How to Predict New Samples With Your Keras Model](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-predict-new-samples-with-your-keras-model.md)