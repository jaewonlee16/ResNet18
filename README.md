# ResNet18 - Final Project

This repository contains the implementation of the final project for the Introduction to Computer Vision course. The goal of this project is to gain experience with PyTorch, utilize pre-trained models, and adapt these models to new tasks and losses.

[Report](https://github.com/jaewonlee16/ResNet18/blob/master/Computer_Vision_Final.pdf) / [Notebook](https://github.com/jaewonlee16/ResNet18/blob/master/FP_rotation.ipynb)

## Project Overview

The project involves using a simple self-supervised rotation prediction task to pre-train a feature representation on the CIFAR10 dataset without using class labels, and then fine-tuning this representation for CIFAR10 classification.

### Key Tasks

1. **Self-Supervised Training**: Train a ResNet18 model to predict the rotation of images.
2. **Fine-Tuning**: Fine-tune the pre-trained model on the CIFAR10 classification task.
3. **Supervised Training**: Train the model on CIFAR10 with full supervision.

### Model and Dataset

- **Model Architecture**: ResNet18 using PyTorch's implementation.
- **Dataset**: CIFAR10, containing small (32x32) images from 10 classes.



## Setup Instructions

### Prerequisites

- Python 3.6 or higher
- PyTorch
- torchvision
- numpy
- matplotlib
- jupyter


## Project Details

### Self-Supervised Training

Train ResNet18 on the rotation prediction task. The network is trained to classify images rotated by 0, 90, 180, or 270 degrees.

### Fine-Tuning

Fine-tune only the final layers of the pre-trained model on CIFAR10 classification. Compare the performance with a model initialized with random weights.

### Supervised Training

Train the entire ResNet18 model on CIFAR10 from scratch and compare the results with the fine-tuned model.

## Extra Credit

Recreate the plot from Gidaris et al. (2018) showing CIFAR10 classification performance vs. number of training examples per category, comparing a supervised CIFAR10 model with a RotNet model fine-tuned on CIFAR10.

<img src= "https://github.com/jaewonlee16/ResNet18/assets/73290953/a3260115-5b57-48af-b6ec-2c3650ec7169" width="60%" height="60%">


## Authors

- [jaewonlee16](https://github.com/jaewonlee16)
