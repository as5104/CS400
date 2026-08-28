import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("results/cnn")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


epochs = list(range(1, 51))

train_loss = [
    1.3540, 0.9552, 0.7867, 0.6921, 0.6237,
    0.5770, 0.5371, 0.5105, 0.4826, 0.4614,
    0.4436, 0.4280, 0.4105, 0.3966, 0.3840,
    0.3691, 0.3593, 0.3474, 0.3392, 0.3293,
    0.3156, 0.3068, 0.3033, 0.2953, 0.2887,
    0.2814, 0.2736, 0.2665, 0.2586, 0.2578,
    0.2503, 0.2492, 0.2390, 0.2336, 0.2268,
    0.2271, 0.2228, 0.2156, 0.2128, 0.2078,
    0.2044, 0.2028, 0.1956, 0.1909, 0.1903,
    0.1860, 0.1816, 0.1803, 0.1827, 0.1791,
]

train_accuracy = [
    50.83, 66.24, 72.40, 75.96, 78.53,
    79.89, 81.56, 82.36, 83.31, 83.98,
    84.57, 85.09, 85.75, 86.37, 86.67,
    87.27, 87.65, 87.99, 88.23, 88.50,
    88.88, 89.41, 89.54, 89.72, 89.90,
    90.22, 90.43, 90.76, 91.05, 90.82,
    91.26, 91.37, 91.68, 91.78, 92.07,
    91.95, 91.97, 92.37, 92.49, 92.75,
    92.79, 92.83, 93.06, 93.28, 93.16,
    93.41, 93.67, 93.56, 93.45, 93.62,
]

val_loss = [
    1.3179, 1.1369, 0.9621, 0.6291, 0.8190,
    0.6481, 0.6566, 0.4991, 0.4816, 0.4826,
    0.4793, 0.4466, 0.4966, 0.4074, 0.3750,
    0.4115, 0.3733, 0.3476, 0.3983, 0.3293,
    0.3168, 0.3635, 0.3189, 0.3029, 0.3195,
    0.2988, 0.2737, 0.2712, 0.2602, 0.2610,
    0.2079, 0.2951, 0.2436, 0.2105, 0.2400,
    0.2748, 0.2181, 0.2036, 0.2196, 0.2123,
    0.2026, 0.2346, 0.1901, 0.2448, 0.2052,
    0.2035, 0.1991, 0.1780, 0.1986, 0.1712,
]

val_accuracy = [
    53.68, 59.96, 66.78, 78.34, 72.72,
    78.34, 77.50, 83.08, 83.24, 83.54,
    83.42, 84.44, 83.58, 86.32, 87.26,
    86.66, 87.28, 87.90, 86.12, 88.48,
    89.16, 87.70, 89.08, 89.46, 88.84,
    89.72, 90.60, 90.72, 90.86, 91.14,
    92.68, 89.66, 91.68, 92.36, 91.68,
    90.84, 92.34, 92.66, 91.98, 92.52,
    92.92, 91.84, 93.38, 90.98, 92.76,
    92.62, 93.02, 93.90, 92.80, 94.24,
]

epoch_time = [
    28.3, 28.1, 26.8, 26.9, 27.0,
    27.2, 27.0, 26.9, 33.5, 30.8,
    31.6, 28.5, 27.4, 27.4, 27.3,
    27.8, 27.2, 27.2, 27.4, 27.6,
    29.4, 27.2, 27.8, 27.2, 27.2,
    27.2, 27.3, 28.2, 30.9, 29.8,
    30.8, 30.2, 32.2, 32.3, 27.8,
    26.8, 27.2, 26.9, 26.8, 27.4,
    27.6, 27.5, 27.2, 27.4, 26.9,
    26.8, 27.0, 27.6, 27.6, 27.6,
]


history_path = RESULTS_DIR / "training_history.csv"

with history_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "epoch_time_seconds",
    ])

    for i in range(50):
        writer.writerow([
            epochs[i],
            train_loss[i],
            train_accuracy[i],
            val_loss[i],
            val_accuracy[i],
            epoch_time[i],
        ])


best_epoch = max(
    range(50),
    key=lambda i: val_accuracy[i],
)

best_validation_accuracy = val_accuracy[best_epoch]
best_validation_loss = val_loss[best_epoch]

summary = {
    "experiment": "Basic CNN - Experiment 1",
    "dataset": "CIFAR-10",
    "train_samples": 45000,
    "validation_samples": 5000,
    "test_samples": 10000,
    "model": "BasicCNN",
    "parameters": 288746,
    "trainable_parameters": 288746,
    "batch_size": 128,
    "epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 0.0005,
    "optimizer": "AdamW",
    "loss_function": "CrossEntropyLoss",
    "best_epoch": best_epoch + 1,
    "best_validation_accuracy": best_validation_accuracy,
    "best_validation_loss": best_validation_loss,
    "final_train_loss": train_loss[-1],
    "final_train_accuracy": train_accuracy[-1],
    "final_validation_loss": val_loss[-1],
    "final_validation_accuracy": val_accuracy[-1],
    "total_training_time_minutes": 23.44,
    "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
}


config_path = RESULTS_DIR / "config.json"

with config_path.open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)


summary_path = RESULTS_DIR / "training_summary.txt"

with summary_path.open("w", encoding="utf-8") as file:
    file.write("Basic CNN - Experiment 1\n")
    file.write("\n")
    file.write("Dataset: CIFAR-10\n")
    file.write("Training samples: 45000\n")
    file.write("Validation samples: 5000\n")
    file.write("Test samples: 10000\n")
    file.write("\n")
    file.write("Model: BasicCNN\n")
    file.write("Parameters: 288746\n")
    file.write("Trainable parameters: 288746\n")
    file.write("\n")
    file.write("Batch size: 128\n")
    file.write("Epochs: 50\n")
    file.write("Optimizer: AdamW\n")
    file.write("Learning rate: 0.001\n")
    file.write("Weight decay: 0.0005\n")
    file.write("Loss function: CrossEntropyLoss\n")
    file.write("\n")
    file.write("Best epoch: 50\n")
    file.write("Best validation accuracy: 94.24%\n")
    file.write("Best validation loss: 0.1712\n")
    file.write("\n")
    file.write("Final training loss: 0.1791\n")
    file.write("Final training accuracy: 93.62%\n")
    file.write("Final validation loss: 0.1712\n")
    file.write("Final validation accuracy: 94.24%\n")
    file.write("\n")
    file.write("Total training time: 23.44 minutes\n")
    file.write("GPU: NVIDIA GeForce RTX 4060 Laptop GPU\n")


plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Basic CNN - Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=200)
plt.close()


plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy, label="Training Accuracy")
plt.plot(epochs, val_accuracy, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Basic CNN - Training and Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=200)
plt.close()


plt.figure(figsize=(10, 6))
plt.plot(epochs, epoch_time, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Time (seconds)")
plt.title("Basic CNN - Time per Epoch")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "epoch_time.png", dpi=200)
plt.close()


print("CNN experiment results recovered successfully.")
print("Results directory:", RESULTS_DIR.resolve())
print("Best epoch:", best_epoch + 1)
print("Best validation accuracy:", f"{best_validation_accuracy:.2f}%")
print("Best validation loss:", f"{best_validation_loss:.4f}")