# Fashion-MNIST Clothing Classifier with PyTorch

A deep learning image classification project built with **PyTorch** that trains a Multi-Layer Perceptron (MLP) to recognize 10 categories of apparel from the **Fashion-MNIST** dataset.

## Overview

This project walks through an end-to-end PyTorch deep learning workflow:
1. Downloading and visualizing grayscale clothing images from `torchvision.datasets`.
2. Batching and shuffling data using PyTorch `DataLoader`.
3. Automatically selecting hardware acceleration (`CUDA`, `MPS`, or `CPU`).
4. Building a 3-layer feedforward neural network with `ReLU` activations.
5. Training and evaluating the model using `CrossEntropyLoss` and Stochastic Gradient Descent (`SGD`).
6. Saving and reloading trained model weights (`model.pth`) for single-image inference.

## Dataset

The model is trained on the **Fashion-MNIST** dataset, consisting of **70,000 grayscale images** (28×28 pixels) across 10 clothing classes:
* **Training Set:** 60,000 images
* **Test Set:** 10,000 images

| Label | Category | Label | Category |
| :---: | :--- | :---: | :--- |
| `0` | T-shirt/top | `5` | Sandal |
| `1` | Trouser | `6` | Shirt |
| `2` | Pullover | `7` | Sneaker |
| `3` | Dress | `8` | Bag |
| `4` | Coat | `9` | Ankle boot |

## Model Architecture

The classifier flattens each 28×28 image into a 784-dimensional vector and passes it through three fully connected linear layers:

```text
Input (1 × 28 × 28)
   │
   ▼
Flatten (784)
   │
   ▼
Linear (784 → 512) + ReLU
   │
   ▼
Linear (512 → 512) + ReLU
   │
   ▼
Linear (512 → 10) [Logits]
```

## Hyperparameters

* **Batch Size:** `25`
* **Epochs:** `5`
* **Loss Function:** `CrossEntropyLoss`
* **Optimizer:** `SGD` (`lr = 0.001`)

## Prerequisites & Installation

Make sure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install torch torchvision matplotlib
```

## Usage
During execution, the script will:
   * Display a 3×3 sample grid of training images.
   * Print training loss every 100 batches and test accuracy after each epoch.
   * Save the trained weights to `model.pth`.
   * Reload `model.pth` and run a sample prediction on the test set.

## Project Structure

```text
pytorch-fashion-mnist-classifier/
├── data/               # Automatically downloaded Fashion-MNIST dataset
├── main.py             # Complete training, evaluation, and inference script
├── model.pth           # Saved PyTorch model weights (generated after training)
└── README.md           # Project documentation
```
