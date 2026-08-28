# Experiment Log

This file records the implementation, testing, observations, and results
for the CNN and ResNet experiments on CIFAR-10.

## 1. Environment Setup

### Hardware

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- GPU Memory: 8 GB

### Software

- Python: 3.13
- PyTorch: 2.9.1+cu130
- Torchvision: 0.24.1+cu130
- CUDA runtime: 13.0

## 2. Testing

### explore_cifar10.py output

LOADING CIFAR-10 
Dataset loaded successfully. 
CIFAR-10 INFORMATION 
Training samples : 50000 
Test samples : 10000 
Image shape : torch.Size([3, 32, 32]) 
Label : 6 
Class : frog 
Pixel min : 0.0 
Pixel max : 1.0

### data.py

data.py works as a library/module that provides this function,
get_dataloaders()

### test_dataloader.py output

DATASET INFORMATION:

Training samples   : 45000
Validation samples : 5000
Test samples       : 10000

ONE TRAINING BATCH:

Images shape : torch.Size([128, 3, 32, 32])
Labels shape : torch.Size([128])
Image dtype  : torch.float32
Label dtype  : torch.int64
Image min    : -1.9894737005233765
Image max    : 2.12648868560791

DEVICE:

Device: cuda
GPU: NVIDIA GeForce RTX 4060 Laptop GPU

After moving to GPU:
Images device : cuda:0
Labels device : cuda:0

### visualize_augmentation.py output

For the Visualization of different data augmentation like,

┌──────────┬──────────┬──────────┬──────────┐
│ Original │ Aug. 1   │ Aug. 2   │ Aug. 3   │
│          │          │          │          │
├──────────┼──────────┼──────────┼──────────┤
│ Aug. 4   │ Aug. 5   │ Aug. 6   │ Aug. 7   │
│          │          │          │          │
└──────────┴──────────┴──────────┴──────────┘

Using,
RandomCrop(),
RandomHorizontalFlip()


## For testing Besic CNN (cnn.py) before training

### test_cnn.py output

Device: cuda
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
Input shape: torch.Size([128, 3, 32, 32])
Output shape: torch.Size([128, 10])
Total parameters: 288746
Trainable parameters: 288746

### inspect_cnn.py output

Device: cuda
Input shape: torch.Size([1, 3, 32, 32])

0: Conv2d -> torch.Size([1, 32, 32, 32])
1: BatchNorm2d -> torch.Size([1, 32, 32, 32])
2: ReLU -> torch.Size([1, 32, 32, 32])
3: Conv2d -> torch.Size([1, 32, 32, 32])
4: BatchNorm2d -> torch.Size([1, 32, 32, 32])
5: ReLU -> torch.Size([1, 32, 32, 32])
6: MaxPool2d -> torch.Size([1, 32, 16, 16])
7: Conv2d -> torch.Size([1, 64, 16, 16])
8: BatchNorm2d -> torch.Size([1, 64, 16, 16])
9: ReLU -> torch.Size([1, 64, 16, 16])
10: Conv2d -> torch.Size([1, 64, 16, 16])
11: BatchNorm2d -> torch.Size([1, 64, 16, 16])
12: ReLU -> torch.Size([1, 64, 16, 16])
13: MaxPool2d -> torch.Size([1, 64, 8, 8])
14: Conv2d -> torch.Size([1, 128, 8, 8])
15: BatchNorm2d -> torch.Size([1, 128, 8, 8])
16: ReLU -> torch.Size([1, 128, 8, 8])
17: Conv2d -> torch.Size([1, 128, 8, 8])
18: BatchNorm2d -> torch.Size([1, 128, 8, 8])
19: ReLU -> torch.Size([1, 128, 8, 8])
pool: AdaptiveAvgPool2d -> torch.Size([1, 128, 1, 1])
flatten: Flatten -> torch.Size([1, 128])
classifier: Linear -> torch.Size([1, 10])

## CNN Training 

### Training Set up

Dataset: CIFAR-10
Train: 45,000
Validation: 5,000
Test: 10,000

Model: BasicCNN
Batch size: 128

Loss: CrossEntropyLoss
Optimizer: AdamW
Learning rate: 0.001
Weight decay: 0.0005

Epochs: 50

### Sequence

Basic CNN
   ↓
Training loop
   ↓
Validation after every epoch
   ↓
Save best checkpoint
   ↓
Plot loss/accuracy
   ↓
Evaluate on test set
   ↓
Visualize predictions

### CNN experiment 1 result 

Epoch 1   → 53.68%
Epoch 4   → 78.34%
Epoch 8   → 83.08%
Epoch 15  → 87.26%
Epoch 20  → 88.48%
Epoch 27  → 90.60%
Epoch 31  → 92.68%
Epoch 38  → 92.66%
Epoch 41  → 92.92%
Epoch 43  → 93.38%
Epoch 48  → 93.90%
Epoch 50  → 94.24%

Best validation accuracy: 94.24%
Best epoch: 50

Final training accuracy: 93.62%
Final training loss: 0.1791
Final validation loss: 0.1712

Training time: 23.44 minutes
Parameters: 288,746

## CNN Test Results

Test loss: 0.4885
Test accuracy: 86.54%

Weighted precision: 0.8677
Weighted recall: 0.8654
Weighted F1: 0.8655


airplane    Precision: 82.33%  Recall: 91.80%  F1: 86.81%
automobile  Precision: 92.45%  Recall: 94.30%  F1: 93.37%
bird        Precision: 75.76%  Recall: 86.90%  F1: 80.95%
cat         Precision: 76.57%  Recall: 73.20%  F1: 74.85%
deer        Precision: 86.18%  Recall: 84.80%  F1: 85.48%
dog         Precision: 83.94%  Recall: 78.40%  F1: 81.08%
frog        Precision: 92.42%  Recall: 89.00%  F1: 90.68%
horse       Precision: 93.35%  Recall: 84.20%  F1: 88.54%
ship        Precision: 92.71%  Recall: 92.80%  F1: 92.75%
truck       Precision: 92.02%  Recall: 90.00%  F1: 91.00%


### visualize_cnn_filters.py

First convolution weight shape: torch.Size([32, 3, 3, 3])
Number of filters: 32
Channels per filter: 3

### visualize_cnn_features.py

True class: cat
Predicted class: cat
block1 feature map shape: (32, 32, 32)
block2 feature map shape: (32, 32, 32)
block3 feature map shape: (64, 16, 16)
block4 feature map shape: (64, 16, 16)
block5 feature map shape: (128, 8, 8)
block6 feature map shape: (128, 8, 8)