Setup
------
1. Clone the repository
```sh
git clone https://github.com/pierre-parel/cbd.git
cd cbd
```
2. Create virtual environment and install modules
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
python EfficientNetV2B0_coffee.py
```

TODO
-------
- [X] Use GPU for faster training. See [Install TensorFlow with pip](https://www.tensorflow.org/install/pip#windows-wsl2_1) and [Use a GPU](https://www.tensorflow.org/guide/gpu)
- [X] Change from Xception to EfficientNetV2B0. 
- [X] Use different pretrained model for comparison
- [ ] Prefetch dataset
- [ ] Data augmentation 


References
------
- [Image Classification from Scratch](https://keras.io/examples/vision/image_classification_from_scratch/)
- [EfficientNetV2B0 function](https://keras.io/api/applications/efficientnet_v2/#efficientnetv2b0-function)
- [Load image dataset from directory](https://keras.io/api/data_loading/image/#imagedatasetfromdirectory-function)
- [Image classification via fine-tuning with EfficientNet](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/)