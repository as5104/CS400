import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torchvision import datasets, transforms

from models.cnn import BasicCNN


DATA_DIR = "./data"
CHECKPOINT_PATH = "checkpoints/cnn_best.pth"
RESULTS_DIR = Path("results/cnn")

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
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    image = image * std + mean

    return image.clamp(0, 1)


def evaluate_model(model, dataset, device):
    model.eval()

    loss_function = nn.CrossEntropyLoss()

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            loss = loss_function(outputs, labels)

            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    test_loss = total_loss / total_samples

    return (
        test_loss,
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities),
    )


def save_prediction_csv(labels, predictions, probabilities):
    output_path = RESULTS_DIR / "test_predictions.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "index",
            "true_label",
            "true_class",
            "predicted_label",
            "predicted_class",
            "confidence",
            "correct",
        ])

        for index in range(len(labels)):
            predicted_label = int(predictions[index])
            confidence = float(probabilities[index][predicted_label])

            writer.writerow([
                index,
                int(labels[index]),
                CLASS_NAMES[int(labels[index])],
                predicted_label,
                CLASS_NAMES[predicted_label],
                confidence,
                bool(labels[index] == predictions[index]),
            ])

    return output_path


def save_classification_report(labels, predictions):
    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    output_path = RESULTS_DIR / "classification_report.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "class",
            "precision",
            "recall",
            "f1_score",
            "support",
        ])

        for class_name in CLASS_NAMES:
            row = report[class_name]

            writer.writerow([
                class_name,
                row["precision"],
                row["recall"],
                row["f1-score"],
                row["support"],
            ])

        for average_name in ["macro avg", "weighted avg"]:
            row = report[average_name]

            writer.writerow([
                average_name,
                row["precision"],
                row["recall"],
                row["f1-score"],
                row["support"],
            ])

        writer.writerow([
            "accuracy",
            report["accuracy"],
            "",
            "",
            len(labels),
        ])

    return output_path


def save_confusion_matrix(labels, predictions):
    matrix = confusion_matrix(
        labels,
        predictions,
    )

    np.savetxt(
        RESULTS_DIR / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
    )

    plt.figure(figsize=(10, 8))

    plt.imshow(matrix)

    plt.xticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES,
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Basic CNN - Confusion Matrix")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(
                col,
                row,
                matrix[row, col],
                ha="center",
                va="center",
            )

    plt.colorbar()
    plt.tight_layout()

    output_path = RESULTS_DIR / "confusion_matrix.png"

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    return matrix


def save_prediction_examples(dataset, labels, predictions, probabilities):
    correct_indices = np.where(labels == predictions)[0]
    incorrect_indices = np.where(labels != predictions)[0]

    rng = np.random.default_rng(42)

    correct_count = min(16, len(correct_indices))
    incorrect_count = min(16, len(incorrect_indices))

    correct_indices = rng.choice(
        correct_indices,
        size=correct_count,
        replace=False,
    )

    incorrect_indices = rng.choice(
        incorrect_indices,
        size=incorrect_count,
        replace=False,
    )

    for indices, filename, title in [
        (
            correct_indices,
            "correct_predictions.png",
            "Basic CNN - Correct Predictions",
        ),
        (
            incorrect_indices,
            "incorrect_predictions.png",
            "Basic CNN - Incorrect Predictions",
        ),
    ]:
        rows = 4
        cols = 4

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(12, 12),
        )

        for ax in axes.flat:
            ax.axis("off")

        for position, index in enumerate(indices):
            image, _ = dataset[int(index)]

            image = denormalize(image)

            axes.flat[position].imshow(
                image.permute(1, 2, 0)
            )

            true_class = CLASS_NAMES[int(labels[index])]
            predicted_class = CLASS_NAMES[int(predictions[index])]

            confidence = float(
                probabilities[index][predictions[index]]
            )

            axes.flat[position].set_title(
                f"True: {true_class}\n"
                f"Pred: {predicted_class}\n"
                f"Conf: {confidence:.2%}",
                fontsize=9,
            )

            axes.flat[position].axis("off")

        plt.suptitle(
            title,
            fontsize=15,
        )

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / filename,
            dpi=200,
        )

        plt.close()


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD,
        ),
    ])

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=test_transform,
    )

    model = BasicCNN(num_classes=10).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Loaded checkpoint from epoch:", checkpoint["epoch"])
    print(
        "Checkpoint validation accuracy:",
        f"{checkpoint['validation_accuracy']:.2f}%",
    )

    test_loss, labels, predictions, probabilities = evaluate_model(
        model,
        test_dataset,
        device,
    )

    test_accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\nTest results:")
    print("Test loss:", f"{test_loss:.4f}")
    print("Test accuracy:", f"{test_accuracy * 100:.2f}%")
    print("Weighted precision:", f"{precision:.4f}")
    print("Weighted recall:", f"{recall:.4f}")
    print("Weighted F1:", f"{f1:.4f}")

    save_prediction_csv(
        labels,
        predictions,
        probabilities,
    )

    save_classification_report(
        labels,
        predictions,
    )

    save_confusion_matrix(
        labels,
        predictions,
    )

    save_prediction_examples(
        test_dataset,
        labels,
        predictions,
        probabilities,
    )

    metrics = {
        "model": "BasicCNN",
        "checkpoint": CHECKPOINT_PATH,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_accuracy": float(
            checkpoint["validation_accuracy"]
        ),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_accuracy_percent": float(test_accuracy * 100),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "test_samples": len(test_dataset),
    }

    with (
        RESULTS_DIR / "test_metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("\nSaved evaluation results to:", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()