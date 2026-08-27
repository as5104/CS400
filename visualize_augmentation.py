import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


DATA_DIR = "./data"

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)

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


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


display_transform = transforms.ToTensor()


def denormalize(image):
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    image = image * std + mean

    return image.clamp(0, 1)


dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=None,
)


image, label = dataset[0]

print("Original image size:", image.size)
print("Class:", CLASS_NAMES[label])

original_tensor = display_transform(image)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

axes[0, 0].imshow(original_tensor.permute(1, 2, 0))
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")


for i in range(1, 8):
    augmented_image = train_transform(image)

    augmented_image = denormalize(augmented_image)

    row = i // 4
    col = i % 4

    axes[row, col].imshow(
        augmented_image.permute(1, 2, 0)
    )

    axes[row, col].set_title(f"Augmented {i}")
    axes[row, col].axis("off")


plt.suptitle(
    f"Data Augmentation: {CLASS_NAMES[label]}",
    fontsize=14
)

plt.tight_layout()
plt.show()