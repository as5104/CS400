import matplotlib.pyplot as plt
import torch

from torchvision import datasets, transforms

from models.resnet import ResNet18


DATA_DIR = "./data"
CHECKPOINT_PATH = "checkpoints/resnet18_best.pth"
RESULTS_DIR = "results/resnet18"

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


def denormalize(image):
    mean = torch.tensor(
        CIFAR10_MEAN
    ).view(3, 1, 1)

    std = torch.tensor(
        CIFAR10_STD
    ).view(3, 1, 1)

    return (
        image * std + mean
    ).clamp(0, 1)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ResNet18(
        num_classes=10
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD,
        ),
    ])

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=transform,
    )

    image, label = dataset[0]

    x = image.unsqueeze(0).to(device)

    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)

        block = model.layer1[0]

        identity = block.shortcut(x)

        main_path = block.conv1(x)
        main_path = block.bn1(main_path)
        main_path = block.relu(main_path)

        main_path = block.conv2(main_path)
        main_path = block.bn2(main_path)

        combined = main_path + identity

        output = block.relu(combined)

    print("Input to residual block:", x.shape)
    print(
        "Main path output:",
        main_path.shape,
    )

    print(
        "Shortcut output:",
        identity.shape,
    )

    print(
        "Combined output:",
        combined.shape,
    )

    print(
        "Residual block output:",
        output.shape,
    )

    main_map = main_path[0, 0].cpu()
    shortcut_map = identity[0, 0].cpu()
    combined_map = combined[0, 0].cpu()
    output_map = output[0, 0].cpu()

    maps = [
        (
            main_map,
            "Main Path F(x)",
        ),
        (
            shortcut_map,
            "Shortcut S(x)",
        ),
        (
            combined_map,
            "F(x) + S(x)",
        ),
        (
            output_map,
            "After ReLU",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    for ax, (feature_map, title) in zip(
        axes,
        maps,
    ):
        ax.imshow(
            feature_map,
            cmap="viridis",
        )

        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        "ResNet-18 - Residual Block Components",
        fontsize=14,
    )

    fig.tight_layout()

    output_path = (
        f"{RESULTS_DIR}/residual_block_components.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

    plt.close(fig)

    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()