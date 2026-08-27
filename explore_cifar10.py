import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Configuration

DATA_DIR = "./data"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# 2. Transform

transform = transforms.ToTensor()


# 3. Load CIFAR-10 from local files

print("=" * 60)
print("LOADING CIFAR-10")
print("=" * 60)

train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=False,
    transform=transform
)

print("Dataset loaded successfully.")


# 4. Basic dataset information

print("\n" + "=" * 60)
print("CIFAR-10 INFORMATION")
print("=" * 60)

print("Training samples :", len(train_dataset))
print("Test samples     :", len(test_dataset))

image, label = train_dataset[0]

print("Image shape      :", image.shape)
print("Label            :", label)
print("Class            :", CLASS_NAMES[label])

print("Pixel min        :", image.min().item())
print("Pixel max        :", image.max().item())


# 5. Display 16 sample images

images = []
labels = []

for i in range(16):
    image, label = train_dataset[i]
    images.append(image)
    labels.append(label)


fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flat):

    # PyTorch format:
    # C × H × W
    #
    # Matplotlib expects:
    # H × W × C

    image = images[i].permute(1, 2, 0)

    ax.imshow(image)
    ax.set_title(CLASS_NAMES[labels[i]])
    ax.axis("off")


plt.suptitle(
    "CIFAR-10 Sample Images",
    fontsize=16
)

plt.tight_layout()
plt.show()