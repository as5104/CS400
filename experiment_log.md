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