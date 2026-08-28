import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from models.resnet import ResNet18
from utils.data import get_dataloaders


NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results/resnet18")

CHECKPOINT_PATH = CHECKPOINT_DIR / "resnet18_best.pth"


def train_one_epoch(model, loader, loss_function, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def evaluate(model, loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            loss = loss_function(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def save_history(
    epochs,
    train_losses,
    train_accuracies,
    validation_losses,
    validation_accuracies,
    epoch_times,
):
    history_path = RESULTS_DIR / "training_history.csv"

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
            "epoch_time_seconds",
        ])

        for index in range(len(epochs)):
            writer.writerow([
                epochs[index],
                train_losses[index],
                train_accuracies[index],
                validation_losses[index],
                validation_accuracies[index],
                epoch_times[index],
            ])


def save_plots(
    epochs,
    train_losses,
    train_accuracies,
    validation_losses,
    validation_accuracies,
    epoch_times,
):
    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_losses,
        label="Training Loss",
    )

    plt.plot(
        epochs,
        validation_losses,
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet-18 - Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "loss_curve.png",
        dpi=200,
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_accuracies,
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        validation_accuracies,
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("ResNet-18 - Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "accuracy_curve.png",
        dpi=200,
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        epoch_times,
        marker="o",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.title("ResNet-18 - Time per Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "epoch_time.png",
        dpi=200,
    )

    plt.close()


def save_summary(
    best_epoch,
    best_validation_accuracy,
    best_validation_loss,
    train_losses,
    train_accuracies,
    validation_losses,
    validation_accuracies,
    total_training_time,
    parameter_count,
):
    summary = {
        "experiment": "ResNet-18 Experiment 1",
        "dataset": "CIFAR-10",
        "train_samples": 45000,
        "validation_samples": 5000,
        "test_samples": 10000,
        "model": "ResNet18",
        "parameters": parameter_count,
        "trainable_parameters": parameter_count,
        "batch_size": 128,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "loss_function": "CrossEntropyLoss",
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_loss": best_validation_loss,
        "final_train_loss": train_losses[-1],
        "final_train_accuracy": train_accuracies[-1],
        "final_validation_loss": validation_losses[-1],
        "final_validation_accuracy": validation_accuracies[-1],
        "total_training_time_minutes": total_training_time / 60,
        "gpu": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU",
    }

    with (
        RESULTS_DIR / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    with (
        RESULTS_DIR / "training_summary.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("ResNet-18 Experiment 1\n\n")
        file.write("Dataset: CIFAR-10\n")
        file.write("Training samples: 45000\n")
        file.write("Validation samples: 5000\n")
        file.write("Test samples: 10000\n\n")

        file.write("Model: ResNet18\n")
        file.write(
            f"Parameters: {parameter_count}\n"
        )
        file.write(
            f"Trainable parameters: {parameter_count}\n\n"
        )

        file.write("Batch size: 128\n")
        file.write(f"Epochs: {NUM_EPOCHS}\n")
        file.write("Optimizer: AdamW\n")
        file.write(
            f"Learning rate: {LEARNING_RATE}\n"
        )
        file.write(
            f"Weight decay: {WEIGHT_DECAY}\n"
        )
        file.write(
            "Loss function: CrossEntropyLoss\n\n"
        )

        file.write(
            f"Best epoch: {best_epoch}\n"
        )
        file.write(
            "Best validation accuracy: "
            f"{best_validation_accuracy:.2f}%\n"
        )
        file.write(
            "Best validation loss: "
            f"{best_validation_loss:.4f}\n\n"
        )

        file.write(
            "Final training loss: "
            f"{train_losses[-1]:.4f}\n"
        )
        file.write(
            "Final training accuracy: "
            f"{train_accuracies[-1]:.2f}%\n"
        )
        file.write(
            "Final validation loss: "
            f"{validation_losses[-1]:.4f}\n"
        )
        file.write(
            "Final validation accuracy: "
            f"{validation_accuracies[-1]:.2f}%\n\n"
        )

        file.write(
            "Total training time: "
            f"{total_training_time / 60:.2f} minutes\n"
        )


def main():
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    train_loader, validation_loader, test_loader = (
        get_dataloaders()
    )

    model = ResNet18(
        num_classes=10
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_validation_accuracy = 0.0
    best_validation_loss = None
    best_epoch = 0

    epochs = []
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []
    epoch_times = []

    print("\nTraining configuration:")
    print("Model: ResNet-18")
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Batch size: 128")
    print("Training samples:", len(train_loader.dataset))
    print(
        "Validation samples:",
        len(validation_loader.dataset),
    )
    print(
        "Test samples:",
        len(test_loader.dataset),
    )
    print("Parameters:", parameter_count)

    total_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
        )

        epoch_number = epoch + 1

        epoch_time = time.time() - epoch_start_time

        epochs.append(epoch_number)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)
        epoch_times.append(epoch_time)

        save_history(
            epochs,
            train_losses,
            train_accuracies,
            validation_losses,
            validation_accuracies,
            epoch_times,
        )

        save_plots(
            epochs,
            train_losses,
            train_accuracies,
            validation_losses,
            validation_accuracies,
            epoch_times,
        )

        print(
            f"Epoch {epoch_number:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_validation_loss = validation_loss
            best_epoch = epoch_number

            torch.save(
                {
                    "epoch": epoch_number,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "validation_loss": validation_loss,
                },
                CHECKPOINT_PATH,
            )

            print(
                "Saved best model with validation "
                f"accuracy: {validation_accuracy:.2f}%"
            )

    total_training_time = (
        time.time() - total_start_time
    )

    save_summary(
        best_epoch,
        best_validation_accuracy,
        best_validation_loss,
        train_losses,
        train_accuracies,
        validation_losses,
        validation_accuracies,
        total_training_time,
        parameter_count,
    )

    print("\nTraining completed.")
    print(
        "Total training time:",
        f"{total_training_time / 60:.2f} minutes",
    )
    print(
        "Best validation accuracy:",
        f"{best_validation_accuracy:.2f}%",
    )
    print("Best epoch:", best_epoch)
    print(
        "Results saved to:",
        RESULTS_DIR.resolve(),
    )


if __name__ == "__main__":
    main()