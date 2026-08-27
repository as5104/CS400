import torch

from models.cnn import BasicCNN


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = BasicCNN(num_classes=10).to(device)
    model.eval()

    x = torch.randn(1, 3, 32, 32).to(device)

    print("Device:", device)
    print("Input shape:", x.shape)
    print()

    current = x

    for name, layer in model.features.named_children():
        current = layer(current)
        print(f"{name}: {layer.__class__.__name__} -> {current.shape}")

    current = model.pool(current)
    print(f"pool: AdaptiveAvgPool2d -> {current.shape}")

    current = torch.flatten(current, 1)
    print(f"flatten: Flatten -> {current.shape}")

    current = model.classifier(current)
    print(f"classifier: Linear -> {current.shape}")


if __name__ == "__main__":
    main()