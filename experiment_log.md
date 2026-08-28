# Experiment Log

This file records the implementation, testing, observations, and results
for the CNN and ResNet experiments on CIFAR-10.

## Environment Setup

### Hardware

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- GPU Memory: 8 GB

### Software

- Python: 3.13
- PyTorch: 2.9.1+cu130
- Torchvision: 0.24.1+cu130
- CUDA runtime: 13.0

## Testing

### explore_cifar10.py output

```
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
```

### data.py

data.py works as a library/module that provides this function,
get_dataloaders()

### test_dataloader.py output

```
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
```

### visualize_augmentation.py output

For the Visualization of different data augmentation like,

```
┌──────────┬──────────┬──────────┬──────────┐
│ Original │ Aug. 1   │ Aug. 2   │ Aug. 3   │
│          │          │          │          │
├──────────┼──────────┼──────────┼──────────┤
│ Aug. 4   │ Aug. 5   │ Aug. 6   │ Aug. 7   │
│          │          │          │          │
└──────────┴──────────┴──────────┴──────────┘
```

Using,
RandomCrop(),
RandomHorizontalFlip()

## For testing Besic CNN (cnn.py) before training

### test_cnn.py output

```
Device: cuda
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
Input shape: torch.Size([128, 3, 32, 32])
Output shape: torch.Size([128, 10])
Total parameters: 288746
Trainable parameters: 288746
```

### inspect_cnn.py output

```
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
```

## CNN Training

### Training Set up

- Dataset: CIFAR-10
- Train: 45,000
- Validation: 5,000
- Test: 10,000

- Model: BasicCNN
- Batch size: 128

- Loss: CrossEntropyLoss
- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.0005

- Epochs: 50

### Sequence

```
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
```

### CNN experiment 1 result

```
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
```

Best validation accuracy: 94.24%
Best epoch: 50

Final training accuracy: 93.62%
Final training loss: 0.1791
Final validation loss: 0.1712

Training time: 23.44 minutes
Parameters: 288,746

## CNN Test Results

Test Result: 
- Test loss: 0.4885 
- Test accuracy: 86.54% 
- Weighted precision: 0.8677 
- Weighted recall: 0.8654
- Weighted F1: 0.8655

```
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
```

### visualize_cnn_filters.py

```
First convolution weight shape: torch.Size([32, 3, 3, 3])
Number of filters: 32
Channels per filter: 3
```

### visualize_cnn_features.py

```
True class: cat
Predicted class: cat
block1 feature map shape: (32, 32, 32)
block2 feature map shape: (32, 32, 32)
block3 feature map shape: (64, 16, 16)
block4 feature map shape: (64, 16, 16)
block5 feature map shape: (128, 8, 8)
block6 feature map shape: (128, 8, 8)
```

## For testing ResNet18 (resnet.py) before training

### Architecture

For a single image,

```
Input
[3, 32, 32]

Initial Conv
[64, 32, 32]

Stage 1
[64, 32, 32]

Stage 2
[128, 16, 16]

Stage 3
[256, 8, 8]

Stage 4
[512, 4, 4]

Global Average Pool
[512, 1, 1]

Flatten
[512]

Linear
[10]
```

### test_resnet.py output

```
Device: cuda
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
Input shape: torch.Size([128, 3, 32, 32])
Output shape: torch.Size([128, 10])
Total parameters: 11173962
Trainable parameters: 11173962
```

### inspect_resnet.py output

```
Device: cuda
Input shape: torch.Size([1, 3, 32, 32])

Initial Conv: torch.Size([1, 64, 32, 32])
Initial BatchNorm: torch.Size([1, 64, 32, 32])
Initial ReLU: torch.Size([1, 64, 32, 32])

Layer 1, Block 1: torch.Size([1, 64, 32, 32])
Layer 1, Block 2: torch.Size([1, 64, 32, 32])

Layer 2, Block 1: torch.Size([1, 128, 16, 16])
Layer 2, Block 2: torch.Size([1, 128, 16, 16])

Layer 3, Block 1: torch.Size([1, 256, 8, 8])
Layer 3, Block 2: torch.Size([1, 256, 8, 8])

Layer 4, Block 1: torch.Size([1, 512, 4, 4])
Layer 4, Block 2: torch.Size([1, 512, 4, 4])

Global Average Pool: torch.Size([1, 512, 1, 1])
Flatten: torch.Size([1, 512])
Linear: torch.Size([1, 10])
```

## ResNet18 Training

### Training Set up

- Dataset: CIFAR-10
- Train: 45,000
- Validation: 5,000
- Test: 10,000

- Model: BasicCNN
- Batch size: 128

- Loss: CrossEntropyLoss
- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.0005

- Epochs: 50

### CNN experiment 1 result

```
Epoch 1  → 57.58%
Epoch 2  → 68.32%
Epoch 3  → 70.22%
Epoch 4  → 79.36%
Epoch 6  → 82.54%
Epoch 7  → 84.04%
Epoch 8  → 85.50%
Epoch 9  → 87.90%
Epoch 11 → 88.06%
Epoch 13 → 91.28%
Epoch 15 → 91.50%
Epoch 16 → 92.56%
Epoch 17 → 93.12%
Epoch 18 → 93.76%
Epoch 20 → 95.06%
Epoch 22 → 95.12%
Epoch 24 → 95.96%
Epoch 29 → 96.64%
Epoch 30 → 97.26%
Epoch 37 → 97.74%
Epoch 39 → 97.90%
Epoch 40 → 98.12%
Epoch 43 → 98.36%
Epoch 47 → 98.42%
```

Total training time: 44.07 minutes
Best validation accuracy: 98.42%
Best epoch: 47

## ResNet Test Results

Test results:
- Test loss: 0.4292
- Test accuracy: 91.51%
- Weighted precision: 0.9166
- Weighted recall: 0.9151
- Weighted F1: 0.9153

```
airplane     Precision: 90.16%  Recall: 95.30%  F1: 92.66%
automobile   Precision: 96.95%  Recall: 95.30%  F1: 96.12%
bird         Precision: 85.03%  Recall: 90.30%  F1: 87.58%
cat          Precision: 85.84%  Recall: 81.20%  F1: 83.45%
deer         Precision: 92.65%  Recall: 92.00%  F1: 92.32%
dog          Precision: 83.40%  Recall: 89.90%  F1: 86.53%
frog         Precision: 96.16%  Recall: 90.20%  F1: 93.09%
horse        Precision: 96.08%  Recall: 93.20%  F1: 94.62%
ship         Precision: 96.76%  Recall: 92.50%  F1: 94.58%
truck        Precision: 93.61%  Recall: 95.20%  F1: 94.40%
```

### visualize_resnet_filters.py

```
First convolution weight shape: torch.Size([64, 3, 3, 3])
Number of filters: 64
Channels per filter: 3
```

### visualize_resnet_features.py

```
True class: cat
Predicted class: cat

initial_conv feature map shape: (64, 32, 32)
layer1 feature map shape: (64, 32, 32)
layer2 feature map shape: (128, 16, 16)
layer3 feature map shape: (256, 8, 8)
layer4 feature map shape: (512, 4, 4)
```

### visualize_resnet_residual_block.py

A residual block from Layer 1 was visualized using a correctly classified CIFAR-10 test image.

The block received an input of shape 64 × 32 × 32. Since the input and output dimensions matched, an identity shortcut was used.

The visualization showed the main-path representation F(x), the shortcut representation S(x), their element-wise sum F(x) + S(x), and the final representation after ReLU.

This demonstrates the core residual operation:

y = ReLU(F(x) + S(x))

For this block, S(x) = x.

```
Input to residual block: torch.Size([1, 64, 32, 32])
Main path output: torch.Size([1, 64, 32, 32])
Shortcut output: torch.Size([1, 64, 32, 32])
Combined output: torch.Size([1, 64, 32, 32])
Residual block output: torch.Size([1, 64, 32, 32])
```

## Comparison

### compare_models.py

CNN test accuracy: 86.54%
ResNet test accuracy: 91.51%
Test accuracy improvement: 4.97 percentage points

| Metric | Basic CNN | ResNet-18 |
|---|---:|---:|
| Parameters | 288,746 | 11,173,962 |
| Training Time | 23.44 min | 44.07 min |
| Best Validation Accuracy | 94.24% | 98.42% |
| Test Accuracy | 86.54% | 91.51% |
| Test Loss | 0.4885 | 0.4292 |
| Weighted Precision | 0.8677 | 0.9166 |
| Weighted Recall | 0.8654 | 0.9151 |
| Weighted F1 | 0.8655 | 0.9153 |

ResNet-18 improved test accuracy from 86.54% to 91.51%, an improvement of 4.97 percentage points. It also achieved higher precision, recall, and F1 scores, with improved F1 performance across all ten CIFAR-10 classes.

The Basic CNN contains 288,746 parameters, while ResNet-18 contains 11,173,962 parameters, approximately 38.7× more. Therefore, the performance improvement cannot be attributed solely to residual/skip connections; the greater depth and model capacity also contribute.

The experiments demonstrate that ResNet-18 provided substantially better classification performance than the Basic CNN under the chosen CIFAR-10 training setup, at the cost of increased training time and model complexity.