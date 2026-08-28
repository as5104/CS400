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


def denormalize(image):
    mean = torch.tensor(
        CIFAR10_MEAN
    ).view(3, 1, 1)

    std = torch.tensor(
        CIFAR10_STD
    ).view(3, 1, 1)

    image = image * std + mean

    return image.clamp(0, 1)


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

    image_batch = image.unsqueeze(0).to(device)

    activations = {}

    def save_activation(name):
        def hook(model, inputs, output):
            activations[name] = output.detach().cpu()

        return hook

    model.conv1.register_forward_hook(
        save_activation("initial_conv")
    )

    model.layer1.register_forward_hook(
        save_activation("layer1")
    )

    model.layer2.register_forward_hook(
        save_activation("layer2")
    )

    model.layer3.register_forward_hook(
        save_activation("layer3")
    )

    model.layer4.register_forward_hook(
        save_activation("layer4")
    )

    with torch.no_grad():
        output = model(image_batch)

    predicted_class = output.argmax(
        dim=1
    ).item()

    print(
        "True class:",
        CLASS_NAMES[label],
    )

    print(
        "Predicted class:",
        CLASS_NAMES[predicted_class],
    )

    print()

    original_image = denormalize(image)

    plt.figure(figsize=(4, 4))

    plt.imshow(
        original_image.permute(1, 2, 0)
    )

    plt.title(
        f"True: {CLASS_NAMES[label]} | "
        f"Predicted: {CLASS_NAMES[predicted_class]}"
    )

    plt.axis("off")
    plt.tight_layout()

    output_path = (
        f"{RESULTS_DIR}/feature_input_image.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

    for layer_name in [
        "initial_conv",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    ]:
        feature_maps = activations[
            layer_name
        ][0]

        print(
            f"{layer_name} feature map shape:",
            tuple(feature_maps.shape),
        )

        number_to_show = min(
            16,
            feature_maps.shape[0],
        )

        fig, axes = plt.subplots(
            4,
            4,
            figsize=(10, 10),
        )

        for i, ax in enumerate(
            axes.flat
        ):
            if i < number_to_show:
                ax.imshow(
                    feature_maps[i],
                    cmap="viridis",
                )

                ax.set_title(
                    f"Channel {i}"
                )

            ax.axis("off")

        fig.suptitle(
            f"ResNet-18 - {layer_name} Feature Maps",
            fontsize=14,
        )

        fig.tight_layout()

        output_path = (
            f"{RESULTS_DIR}/feature_maps_{layer_name}.png"
        )

        fig.savefig(
            output_path,
            dpi=200,
        )

        plt.show()

        plt.close(fig)

    print()
    print(
        "Feature-map visualizations saved."
    )


if __name__ == "__main__":
    main()