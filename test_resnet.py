import torch

from models.resnet import ResNet18


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ResNet18(num_classes=10).to(device)

    x = torch.randn(
        128,
        3,
        32,
        32,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        output = model(x)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Total parameters:", total_parameters)
    print(
        "Trainable parameters:",
        trainable_parameters,
    )


if __name__ == "__main__":
    main()