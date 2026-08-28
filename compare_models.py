import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


CNN_DIR = Path("results/cnn")
RESNET_DIR = Path("results/resnet18")
COMPARISON_DIR = Path("results/comparison")

COMPARISON_DIR.mkdir(
    parents=True,
    exist_ok=True,
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


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_csv(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def save_csv(path, headers, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for row in rows:
            writer.writerow(row)


def create_model_comparison(cnn_config, resnet_config):
    headers = [
        "metric",
        "Basic CNN",
        "ResNet-18",
        "difference",
        "relative_difference_percent",
    ]

    metrics = [
        (
            "Parameters",
            cnn_config["parameters"],
            resnet_config["parameters"],
        ),
        (
            "Best validation accuracy (%)",
            cnn_config["best_validation_accuracy"],
            resnet_config["best_validation_accuracy"],
        ),
        (
            "Best epoch",
            cnn_config["best_epoch"],
            resnet_config["best_epoch"],
        ),
        (
            "Final training accuracy (%)",
            cnn_config["final_train_accuracy"],
            resnet_config["final_train_accuracy"],
        ),
        (
            "Final validation accuracy (%)",
            cnn_config["final_validation_accuracy"],
            resnet_config["final_validation_accuracy"],
        ),
        (
            "Final training loss",
            cnn_config["final_train_loss"],
            resnet_config["final_train_loss"],
        ),
        (
            "Final validation loss",
            cnn_config["final_validation_loss"],
            resnet_config["final_validation_loss"],
        ),
        (
            "Training time (minutes)",
            cnn_config["total_training_time_minutes"],
            resnet_config["total_training_time_minutes"],
        ),
    ]

    rows = []

    for metric, cnn_value, resnet_value in metrics:
        difference = resnet_value - cnn_value

        if cnn_value != 0:
            relative_difference = (
                difference / cnn_value
            ) * 100
        else:
            relative_difference = ""

        rows.append([
            metric,
            cnn_value,
            resnet_value,
            difference,
            relative_difference,
        ])

    save_csv(
        COMPARISON_DIR / "model_comparison.csv",
        headers,
        rows,
    )


def create_test_comparison(cnn_metrics, resnet_metrics):
    headers = [
        "metric",
        "Basic CNN",
        "ResNet-18",
        "difference",
    ]

    metrics = [
        (
            "Test loss",
            cnn_metrics["test_loss"],
            resnet_metrics["test_loss"],
        ),
        (
            "Test accuracy (%)",
            cnn_metrics["test_accuracy_percent"],
            resnet_metrics["test_accuracy_percent"],
        ),
        (
            "Weighted precision",
            cnn_metrics["weighted_precision"],
            resnet_metrics["weighted_precision"],
        ),
        (
            "Weighted recall",
            cnn_metrics["weighted_recall"],
            resnet_metrics["weighted_recall"],
        ),
        (
            "Weighted F1",
            cnn_metrics["weighted_f1"],
            resnet_metrics["weighted_f1"],
        ),
    ]

    rows = []

    for metric, cnn_value, resnet_value in metrics:
        rows.append([
            metric,
            cnn_value,
            resnet_value,
            resnet_value - cnn_value,
        ])

    save_csv(
        COMPARISON_DIR / "test_metrics_comparison.csv",
        headers,
        rows,
    )


def create_per_class_comparison():
    cnn_rows = load_csv(
        CNN_DIR / "classification_report.csv"
    )

    resnet_rows = load_csv(
        RESNET_DIR / "classification_report.csv"
    )

    cnn_data = {
        row["class"]: row
        for row in cnn_rows
        if row["class"] in CLASS_NAMES
    }

    resnet_data = {
        row["class"]: row
        for row in resnet_rows
        if row["class"] in CLASS_NAMES
    }

    headers = [
        "class",
        "cnn_precision",
        "resnet_precision",
        "precision_difference",
        "cnn_recall",
        "resnet_recall",
        "recall_difference",
        "cnn_f1",
        "resnet_f1",
        "f1_difference",
    ]

    rows = []

    for class_name in CLASS_NAMES:
        cnn_precision = float(
            cnn_data[class_name]["precision"]
        )

        resnet_precision = float(
            resnet_data[class_name]["precision"]
        )

        cnn_recall = float(
            cnn_data[class_name]["recall"]
        )

        resnet_recall = float(
            resnet_data[class_name]["recall"]
        )

        cnn_f1 = float(
            cnn_data[class_name]["f1_score"]
        )

        resnet_f1 = float(
            resnet_data[class_name]["f1_score"]
        )

        rows.append([
            class_name,
            cnn_precision,
            resnet_precision,
            resnet_precision - cnn_precision,
            cnn_recall,
            resnet_recall,
            resnet_recall - cnn_recall,
            cnn_f1,
            resnet_f1,
            resnet_f1 - cnn_f1,
        ])

    save_csv(
        COMPARISON_DIR / "per_class_comparison.csv",
        headers,
        rows,
    )

    return rows


def plot_training_history():
    cnn_history = load_csv(
        CNN_DIR / "training_history.csv"
    )

    resnet_history = load_csv(
        RESNET_DIR / "training_history.csv"
    )

    epochs_cnn = [
        int(row["epoch"])
        for row in cnn_history
    ]

    epochs_resnet = [
        int(row["epoch"])
        for row in resnet_history
    ]

    cnn_train_loss = [
        float(row["train_loss"])
        for row in cnn_history
    ]

    resnet_train_loss = [
        float(row["train_loss"])
        for row in resnet_history
    ]

    cnn_val_loss = [
        float(row["validation_loss"])
        for row in cnn_history
    ]

    resnet_val_loss = [
        float(row["validation_loss"])
        for row in resnet_history
    ]

    cnn_train_accuracy = [
        float(row["train_accuracy"])
        for row in cnn_history
    ]

    resnet_train_accuracy = [
        float(row["train_accuracy"])
        for row in resnet_history
    ]

    cnn_val_accuracy = [
        float(row["validation_accuracy"])
        for row in cnn_history
    ]

    resnet_val_accuracy = [
        float(row["validation_accuracy"])
        for row in resnet_history
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs_cnn,
        cnn_train_loss,
        label="Basic CNN",
    )

    plt.plot(
        epochs_resnet,
        resnet_train_loss,
        label="ResNet-18",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(
        "Training Loss Comparison"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        COMPARISON_DIR / "training_loss_comparison.png",
        dpi=200,
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs_cnn,
        cnn_val_loss,
        label="Basic CNN",
    )

    plt.plot(
        epochs_resnet,
        resnet_val_loss,
        label="ResNet-18",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(
        "Validation Loss Comparison"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        COMPARISON_DIR / "validation_loss_comparison.png",
        dpi=200,
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs_cnn,
        cnn_train_accuracy,
        label="Basic CNN",
    )

    plt.plot(
        epochs_resnet,
        resnet_train_accuracy,
        label="ResNet-18",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy (%)")
    plt.title(
        "Training Accuracy Comparison"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        COMPARISON_DIR / "training_accuracy_comparison.png",
        dpi=200,
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs_cnn,
        cnn_val_accuracy,
        label="Basic CNN",
    )

    plt.plot(
        epochs_resnet,
        resnet_val_accuracy,
        label="ResNet-18",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title(
        "Validation Accuracy Comparison"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        COMPARISON_DIR / "validation_accuracy_comparison.png",
        dpi=200,
    )

    plt.close()


def plot_test_metrics(cnn_metrics, resnet_metrics):
    metric_names = [
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 (%)",
    ]

    cnn_values = [
        cnn_metrics["test_accuracy_percent"],
        cnn_metrics["weighted_precision"] * 100,
        cnn_metrics["weighted_recall"] * 100,
        cnn_metrics["weighted_f1"] * 100,
    ]

    resnet_values = [
        resnet_metrics["test_accuracy_percent"],
        resnet_metrics["weighted_precision"] * 100,
        resnet_metrics["weighted_recall"] * 100,
        resnet_metrics["weighted_f1"] * 100,
    ]

    positions = range(len(metric_names))

    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        [position - width / 2 for position in positions],
        cnn_values,
        width,
        label="Basic CNN",
    )

    plt.bar(
        [position + width / 2 for position in positions],
        resnet_values,
        width,
        label="ResNet-18",
    )

    plt.xticks(
        list(positions),
        metric_names,
    )

    plt.ylabel("Score (%)")
    plt.title(
        "Test Performance Comparison"
    )

    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        COMPARISON_DIR / "test_metrics_comparison.png",
        dpi=200,
    )

    plt.close()


def plot_model_size_and_time(cnn_config, resnet_config):
    models = [
        "Basic CNN",
        "ResNet-18",
    ]

    parameters = [
        cnn_config["parameters"],
        resnet_config["parameters"],
    ]

    training_times = [
        cnn_config[
            "total_training_time_minutes"
        ],
        resnet_config[
            "total_training_time_minutes"
        ],
    ]

    plt.figure(figsize=(8, 6))

    plt.bar(
        models,
        parameters,
    )

    plt.ylabel("Number of Parameters")
    plt.title(
        "Model Parameter Count"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        COMPARISON_DIR / "parameter_comparison.png",
        dpi=200,
    )

    plt.close()

    plt.figure(figsize=(8, 6))

    plt.bar(
        models,
        training_times,
    )

    plt.ylabel("Training Time (minutes)")
    plt.title(
        "Training Time Comparison"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        COMPARISON_DIR / "training_time_comparison.png",
        dpi=200,
    )

    plt.close()


def plot_per_class_f1(per_class_rows):
    classes = [
        row[0]
        for row in per_class_rows
    ]

    cnn_f1 = [
        row[7] * 100
        for row in per_class_rows
    ]

    resnet_f1 = [
        row[8] * 100
        for row in per_class_rows
    ]

    positions = range(len(classes))
    width = 0.35

    plt.figure(figsize=(13, 7))

    plt.bar(
        [position - width / 2 for position in positions],
        cnn_f1,
        width,
        label="Basic CNN",
    )

    plt.bar(
        [position + width / 2 for position in positions],
        resnet_f1,
        width,
        label="ResNet-18",
    )

    plt.xticks(
        list(positions),
        classes,
        rotation=45,
        ha="right",
    )

    plt.ylabel("F1 Score (%)")
    plt.title(
        "Per-Class F1 Score Comparison"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        COMPARISON_DIR / "per_class_f1_comparison.png",
        dpi=200,
    )

    plt.close()


def main():
    cnn_config = load_json(
        CNN_DIR / "config.json"
    )

    resnet_config = load_json(
        RESNET_DIR / "config.json"
    )

    cnn_metrics = load_json(
        CNN_DIR / "test_metrics.json"
    )

    resnet_metrics = load_json(
        RESNET_DIR / "test_metrics.json"
    )

    create_model_comparison(
        cnn_config,
        resnet_config,
    )

    create_test_comparison(
        cnn_metrics,
        resnet_metrics,
    )

    per_class_rows = (
        create_per_class_comparison()
    )

    plot_training_history()

    plot_test_metrics(
        cnn_metrics,
        resnet_metrics,
    )

    plot_model_size_and_time(
        cnn_config,
        resnet_config,
    )

    plot_per_class_f1(
        per_class_rows,
    )

    print(
        "Model comparison generated successfully."
    )

    print(
        "Results saved to:",
        COMPARISON_DIR.resolve(),
    )

    print(
        "CNN test accuracy:",
        f"{cnn_metrics['test_accuracy_percent']:.2f}%",
    )

    print(
        "ResNet test accuracy:",
        f"{resnet_metrics['test_accuracy_percent']:.2f}%",
    )

    print(
        "Test accuracy improvement:",
        f"{resnet_metrics['test_accuracy_percent'] - cnn_metrics['test_accuracy_percent']:.2f} percentage points",
    )


if __name__ == "__main__":
    main()