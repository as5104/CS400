"""Saved complementary splits. Search loaders never load the test set."""
import hashlib
from pathlib import Path
import numpy as np
from .config import digest, SCHEMA_VERSION
from .storage import read_json, write_json

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def stratified_indices(labels, validation_per_class, seed):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("Labels must be a nonempty one-dimensional array")
    if type(validation_per_class) is not int or validation_per_class < 1:
        raise ValueError("validation_per_class must be a positive integer")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) <= validation_per_class:
            raise ValueError("Each class needs training and validation examples")
        rng.shuffle(indices)
        validation.extend(indices[:validation_per_class].tolist())
        train.extend(indices[validation_per_class:].tolist())
    rng.shuffle(train)
    rng.shuffle(validation)
    validate_indices(train, validation, len(labels))
    return train, validation


def validate_indices(train, validation, count):
    combined = list(train) + list(validation)
    if any(type(i) is not int for i in combined):
        raise ValueError("Split indices must be integers")
    if not train or not validation or len(combined) != count or set(combined) != set(range(count)):
        raise ValueError("Training and validation must cover the dataset exactly once without overlap")


def training_fingerprint(dataset):
    h = hashlib.sha256()
    h.update(memoryview(np.ascontiguousarray(dataset.data)))
    h.update(np.asarray(dataset.targets, dtype="<i8").tobytes())
    return h.hexdigest()


def prepare_cifar10(root, manifest_path, seed=42, download=False):
    from torchvision.datasets import CIFAR10
    destination = Path(manifest_path)
    if destination.exists():
        raise FileExistsError("Manifest exists; choose a new path instead of changing an experiment's splits")
    dataset = CIFAR10(str(root), train=True, download=download)
    train, val = stratified_indices(dataset.targets, 500, seed)
    manifest = dict(schema=SCHEMA_VERSION, dataset="cifar10", seed=seed,
                    classes=dataset.classes, train=train, validation=val,
                    fingerprint=training_fingerprint(dataset),
                    mean=list(CIFAR_MEAN), std=list(CIFAR_STD), image_size=32)
    manifest["id"] = digest(manifest)
    write_json(destination, manifest)
    return manifest


def load_manifest(path):
    manifest = read_json(path)
    content = {k: v for k, v in manifest.items() if k != "id"}
    if manifest.get("schema") != SCHEMA_VERSION or digest(content) != manifest.get("id"):
        raise ValueError("Manifest schema or digest mismatch")
    if manifest.get("dataset") == "image-folder":
        from .folder_data import validate_folder_manifest
        validate_folder_manifest(manifest)
        return manifest
    if manifest.get("dataset") != "cifar10":
        raise ValueError("This initial adapter supports CIFAR-10; custom data needs a separate validated adapter")
    validate_indices(manifest["train"], manifest["validation"], 50_000)
    if len(manifest["train"]) != 45_000 or len(manifest["validation"]) != 5_000:
        raise ValueError("CIFAR-10 requires a saved 45,000/5,000 split")
    if len(manifest["classes"]) != 10 or manifest["image_size"] != 32:
        raise ValueError("Invalid CIFAR-10 metadata")
    return manifest


def transform(manifest, augmentation=None):
    from torchvision import transforms as T
    steps = []
    if manifest.get("dataset") == "image-folder":
        steps.append(T.Resize((32, 32), interpolation=T.InterpolationMode.BILINEAR, antialias=True))
    if augmentation is not None:
        if augmentation not in ("basic", "erasing"):
            raise ValueError("Unknown augmentation policy")
        steps += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(0.5)]
    steps += [T.ToTensor(), T.Normalize(manifest["mean"], manifest["std"])]
    if augmentation == "erasing":
        steps += [T.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0)]
    return T.Compose(steps)


def search_datasets(root, manifest, augmentation):
    if manifest.get("dataset") == "image-folder":
        from .folder_data import FolderDataset
        return (FolderDataset(root, manifest, "train", transform(manifest, augmentation)),
                FolderDataset(root, manifest, "validation", transform(manifest)))
    from torch.utils.data import Subset
    from torchvision.datasets import CIFAR10
    training = CIFAR10(str(root), train=True, download=False, transform=transform(manifest, augmentation))
    if training_fingerprint(training) != manifest["fingerprint"] or training.classes != manifest["classes"]:
        raise ValueError("Dataset content or class mapping changed since split preparation")
    validation = CIFAR10(str(root), train=True, download=False, transform=transform(manifest))
    return Subset(training, manifest["train"]), Subset(validation, manifest["validation"])

