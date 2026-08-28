import torch

from models.resnet import ResNet18


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ResNet18(num_classes=10).to(device)
    model.eval()

    x = torch.randn(
        1,
        3,
        32,
        32,
        device=device,
    )

    print("Device:", device)
    print("Input shape:", x.shape)
    print()

    current = x

    current = model.conv1(current)
    print("Initial Conv:", current.shape)

    current = model.bn1(current)
    print("Initial BatchNorm:", current.shape)

    current = model.relu(current)
    print("Initial ReLU:", current.shape)

    for layer_number, layer in enumerate(
        [
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ],
        start=1,
    ):
        print()

        for block_number, block in enumerate(
            layer,
            start=1,
        ):
            current = block(current)

            print(
                f"Layer {layer_number}, "
                f"Block {block_number}: "
                f"{current.shape}"
            )

    current = model.avg_pool(current)
    print()
    print("Global Average Pool:", current.shape)

    current = torch.flatten(current, 1)
    print("Flatten:", current.shape)

    current = model.fc(current)
    print("Linear:", current.shape)


if __name__ == "__main__":
    main()