Setup
------
1. Download [Coffee Green Bean with 17 Defects](https://www.kaggle.com/datasets/sujitraarw/coffee-green-bean-with-17-defects-original)
2. Unzip to `coffee_bean` folder

TODO:
-------
- [X] Use GPU for faster training. See [Install TensorFlow with pip](https://www.tensorflow.org/install/pip#windows-wsl2_1) and [Use a GPU](https://www.tensorflow.org/guide/gpu)
- [ ] Change from Xception to EfficientNetV2B0. 
- [ ] Prefetch dataset
- [ ] Use different pretrained model for comparison
- [ ] Data augmentation 


References
------
- [Image Classification from Scratch](https://keras.io/examples/vision/image_classification_from_scratch/)
- [EfficientNetV2B0 function](https://keras.io/api/applications/efficientnet_v2/#efficientnetv2b0-function)
- [Load image dataset from directory](https://keras.io/api/data_loading/image/#imagedatasetfromdirectory-function)