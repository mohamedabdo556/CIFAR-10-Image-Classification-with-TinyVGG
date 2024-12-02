# CIFAR-10 Image Classification with TinyVGG

This repository demonstrates the implementation of a CNN model using the TinyVGG architecture to classify images from the CIFAR-10 dataset. The project uses PyTorch for model development, training, and evaluation.

## Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images, categorized into 10 different classes. The model used in this project is TinyVGG, a simplified version of the VGG architecture. It consists of two convolutional blocks followed by a fully connected layer.

## Dataset

- **CIFAR-10**: A dataset of 60,000 32x32 pixel images in 10 classes.
- **Classes**:
  - airplane
  - automobile
  - bird
  - cat
  - deer
  - dog
  - frog
  - horse
  - ship
  - truck

## Features

- Preprocessing and normalization of images using PyTorch transforms.
- Training of the TinyVGG model on the CIFAR-10 dataset.
- Model evaluation on the test dataset with accuracy plotting.
- Saving the trained model for future use.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/mohamedabdo556/CIFAR-10-Image-Classification-with-TinyVGG.git
