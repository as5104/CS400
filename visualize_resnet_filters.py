import matplotlib.pyplot as plt
import torch

from models.resnet import ResNet18


CHECKPOINT_PATH = "checkpoints/resnet18_best.pth"
RESULTS_DIR = "results/resnet18"


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ResNet18(num_classes=10).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    weights = model.conv1.weight.detach().cpu()

    print("First convolution weight shape:", weights.shape)
    print("Number of filters:", weights.shape[0])
    print("Channels per filter:", weights.shape[1])

    fig, axes = plt.subplots(
        8,
        8,
        figsize=(12, 12),
    )

    for i, ax in enumerate(axes.flat):
        filter_weights = weights[i]

        filter_min = filter_weights.min()
        filter_max = filter_weights.max()

        filter_weights = (
            filter_weights - filter_min
        ) / (
            filter_max - filter_min + 1e-8
        )

        filter_weights = filter_weights.permute(1, 2, 0)

        ax.imshow(filter_weights)
        ax.set_title(f"Filter {i + 1}")
        ax.axis("off")

    plt.suptitle(
        "ResNet-18 - Learned First-Layer Filters",
        fontsize=14,
    )

    plt.tight_layout()

    output_path = (
        f"{RESULTS_DIR}/learned_first_layer_filters.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

    print("Saved:", output_path)


if __name__ == "__main__":
    main()