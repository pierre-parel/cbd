#!/bin/sh

. .venv/bin/activate
set -xe

rm -rf coffee_bean*
rm -rf *.jpg
rm -rf *.png
rm -rf saved_models
unzip dataset.zip -d .
python generate_split.py
python generate_augmented_images.py
python transfer_multiclass.py