#!/bin/sh

. .venv/bin/activate
set -xe

rm -rf coffee_bean*
rm -rf *.jpg
rm -rf saved_models
unzip archive.zip -d .
python generate_split.py
python generate_augmented_images.py
python transfer_learning.py