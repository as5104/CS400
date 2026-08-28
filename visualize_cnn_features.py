import matplotlib.pyplot as plt
import torch

from torchvision import datasets, transforms

from models.cnn import BasicCNN


DATA_DIR = "./data"
CHECKPOINT_PATH = "checkpoints/cnn_best.pth"
RESULTS_DIR = "results/cnn"

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
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    image = image * std + mean

    return image.clamp(0, 1)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = BasicCNN(num_classes=10).to(device)

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
        def hook(model, input, output):
            activations[name] = output.detach().cpu()

        return hook

    model.features[2].register_forward_hook(
        save_activation("block1")
    )

    model.features[5].register_forward_hook(
        save_activation("block2")
    )

    model.features[9].register_forward_hook(
        save_activation("block3")
    )

    model.features[12].register_forward_hook(
        save_activation("block4")
    )

    model.features[16].register_forward_hook(
        save_activation("block5")
    )

    model.features[19].register_forward_hook(
        save_activation("block6")
    )

    with torch.no_grad():
        output = model(image_batch)

    predicted_class = output.argmax(dim=1).item()

    print("True class:", CLASS_NAMES[label])
    print("Predicted class:", CLASS_NAMES[predicted_class])

    original_image = denormalize(image)

    plt.figure(figsize=(4, 4))
    plt.imshow(original_image.permute(1, 2, 0))
    plt.title(
        f"Input Image\n"
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

    selected_layers = [
        "block1",
        "block2",
        "block3",
        "block4",
        "block5",
        "block6",
    ]

    for layer_name in selected_layers:
        feature_maps = activations[layer_name][0]

        print(
            f"{layer_name} feature map shape:",
            tuple(feature_maps.shape),
        )

        number_to_show = min(16, feature_maps.shape[0])

        rows = 4
        cols = 4

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(10, 10),
        )

        for i, ax in enumerate(axes.flat):
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
            f"Basic CNN - {layer_name} Feature Maps",
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

    print("\nFeature-map visualizations saved.")


if __name__ == "__main__":
    main()