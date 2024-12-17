# Green Coffee Bean Defect Detection 
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
3. From a windows terminal, enter WSL:
```sh
wsl
```
4. Setup your Linux user info by providing a username and password.

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
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Download [Coffee Green Bean with 17 Defects](https://www.kaggle.com/datasets/sujitraarw/coffee-green-bean-with-17-defects-original)
3. Unzip to `coffee_bean` folder. Your directory structure shoud look like:
```
cbd/
    .venv/
    checkpoints/
    coffee_bean/
        Broken/
        Cut/
        Dry Cherry/
        ...
    examples/
    .py files
```
4. Run the python file.
```sh
python transfer_learning.py
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