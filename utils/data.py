import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Configuration

DATA_DIR = "./data"
BATCH_SIZE = 128
NUM_WORKERS = 4
SEED = 42


# CIFAR-10 normalization statistics

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


# Training transform

train_transform = transforms.Compose([
    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    ),
])


# Validation / Test transform

test_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    ),
])


# Load CIFAR-10

def get_dataloaders():

    # Full training dataset

    full_train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=train_transform,
    )

    # Create a separate dataset for validation.
    # Validation images won't receive random augmentation.


    validation_dataset_full = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=test_transform,
    )

    # Reproducible train/validation split

    generator = torch.Generator().manual_seed(SEED)

    train_size = 45_000
    validation_size = 5_000

    train_dataset, _ = random_split(
        full_train_dataset,
        [train_size, validation_size],
        generator=generator,
    )

    _, validation_dataset = random_split(
        validation_dataset_full,
        [train_size, validation_size],
        generator=generator,
    )

    # Test dataset

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=test_transform,
    )

    # DataLoaders

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )