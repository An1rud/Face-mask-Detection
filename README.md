# Face-mask-Detection
This repository contains code to train and evaluate two Convolutional Neural Network (CNN) models for mask detection using the MobileNetV2 and ResNet50 architectures. These models are trained to classify images into two categories: `with_mask` and `without_mask`.

## Table of Contents

1. Installation
2. Dataset
3. Running the scripts
4. Important Notes

### Installation:

#### Clone the repository:

```bash
git clone https://github.com/An1rud/Face-mask-Detection.git
cd Face-mask-Detection
```
#### Install the required packages:
```bash
pip install -r requirements.txt
```
### Dataset:
-The dataset can be downloaded from [Kaggle: Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset). Ensure that the dataset is organized into the following directory structure:
```bash
dataset/
    with_mask/
        image1.jpg
        image2.jpg
        ...
    without_mask/
        image1.jpg
        image2.jpg
        ...
```
### Running the scripts:
- Run the `resnet50.py` or `mobilenetv2.py.py` to train the using the dataset 
- Then a model will be generated.
- Run the`pred.py` for checking the woring using an image.
- Run the `resnet50pred.py` or the `mobilenetv2pred.py` to see the realtime working

### Important Notes:
- Ensure proper lighting conditions for accurate detection.
- This is a simple mask detection system and may not be suitable for all scenarios. It serves as a basic demonstration of face and object detection in OpenCV.
