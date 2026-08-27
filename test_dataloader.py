import torch

from utils.data import get_dataloaders


def main():
    train_loader, validation_loader, test_loader = get_dataloaders()

    print("DATASET INFORMATION: ")

    print("Training samples   :", len(train_loader.dataset))
    print("Validation samples :", len(validation_loader.dataset))
    print("Test samples       :", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("\n")
    print("ONE TRAINING BATCH: ")

    print("Images shape :", images.shape)
    print("Labels shape :", labels.shape)

    print("Image dtype  :", images.dtype)
    print("Label dtype  :", labels.dtype)

    print("Image min    :", images.min().item())
    print("Image max    :", images.max().item())

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n")
    print("DEVICE: ")

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    images = images.to(device)
    labels = labels.to(device)

    print("\nAfter moving to GPU:")
    print("Images device :", images.device)
    print("Labels device :", labels.device)


if __name__ == "__main__":
    main()