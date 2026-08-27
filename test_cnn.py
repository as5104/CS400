import torch

from models.cnn import BasicCNN


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = BasicCNN(num_classes=10)
    model = model.to(device)

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    x = torch.randn(128, 3, 32, 32).to(device)

    print("Input shape:", x.shape)

    with torch.no_grad():
        output = model(x)

    print("Output shape:", output.shape)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Total parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)


if __name__ == "__main__":
    main()