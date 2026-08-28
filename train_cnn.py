import time

import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import BasicCNN
from utils.data import get_dataloaders


NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005

CHECKPOINT_PATH = "checkpoints/cnn_best.pth"


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


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, validation_loader, test_loader = get_dataloaders()

    model = BasicCNN(num_classes=10).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_validation_accuracy = 0.0

    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []

    print("\nTraining configuration:")
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(validation_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

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

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "validation_loss": validation_loss,
                },
                CHECKPOINT_PATH,
            )

            print(
                f"Saved best model with "
                f"validation accuracy: {validation_accuracy:.2f}%"
            )

    total_time = time.time() - total_start_time

    print("\nTraining completed.")
    print(f"Total training time: {total_time / 60:.2f} minutes")
    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()