"""Post-training evaluation and portable exports; never imported by the sampler."""
from dataclasses import asdict
from pathlib import Path
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .config import Candidate, Protocol, digest, SCHEMA_VERSION
from .data import load_manifest, transform
from .model import ResNet
from .storage import read_json, write_json, run_lock
from .trainer import load_checkpoint, save_checkpoint, validation_accuracy

CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
)


def robustness_metrics(clean_accuracy, condition_accuracies):
    expected = {(c, s) for c in CORRUPTIONS for s in range(1, 6)}
    if set(condition_accuracies) != expected:
        raise ValueError("Robustness needs exactly the 15 declared corruptions at all five severities")
    scores = [clean_accuracy, *condition_accuracies.values()]
    if any(not math.isfinite(a) or not 0 <= a <= 1 for a in scores):
        raise ValueError("Accuracies must be finite fractions in [0, 1]")
    average = sum(condition_accuracies.values()) / 75
    return dict(clean_accuracy=clean_accuracy, corruption_accuracy=average,
                mean_corruption_error=1 - average,
                accuracy_drop_percentage_points=100 * (clean_accuracy - average),
                reference_normalized=False)


def completed_model(run_dir, manifest_path, device_name):
    directory = Path(run_dir)
    meta = read_json(directory / "run.json")
    search_root = directory.parent.parent
    if (search_root / "search.sqlite").exists():
        report = read_json(search_root / "search-results.json")
        identity = Candidate(**meta["candidate"]).id
        if not report["frozen"] or identity not in {r["metrics"]["candidate_id"] for r in report["completed"]}:
            raise ValueError("Search evaluation requires a frozen archive and an eligible completed candidate")
    manifest = load_manifest(manifest_path)
    if manifest["id"] != meta["manifest_id"]:
        raise ValueError("Evaluation manifest differs from the training manifest")
    protocol = Protocol(**meta["protocol"])
    checkpoint = load_checkpoint(directory / "checkpoint.pt", digest(meta))
    if checkpoint["epoch"] != protocol.full_epochs:
        raise ValueError("Only full-budget checkpoints qualify for final evaluation/export")
    device = torch.device(device_name)
    model = ResNet(Candidate(**meta["candidate"]), len(manifest["classes"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), meta, manifest, checkpoint


class CorruptionDataset(Dataset):
    def __init__(self, array, labels, severity, preprocess):
        self.array = array
        self.labels = labels
        self.offset = (severity - 1) * 10000
        self.preprocess = preprocess

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        from PIL import Image
        return self.preprocess(Image.fromarray(self.array[self.offset + index])), int(self.labels[index])


def evaluate_run(run_dir, manifest_path, data_root, corruption_root, output, device_name="cuda"):
    from torchvision.datasets import CIFAR10
    if load_manifest(manifest_path)["dataset"] != "cifar10":
        raise ValueError("CIFAR-10-C is only valid for CIFAR-10. Custom data supports --clean-only; define a domain corruption benchmark separately")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("Evaluation output exists; use a new output path")
    root = Path(corruption_root)
    # Check the whole benchmark before spending time on inference.
    labels = np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False)
    if labels.shape == (50000,):
        if any(not np.array_equal(labels[:10000], labels[i:i+10000]) for i in range(10000, 50000, 10000)):
            raise ValueError("CIFAR-10-C label blocks differ across severities")
        labels = labels[:10000]
    if labels.shape != (10000,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Unexpected CIFAR-10-C labels")
    for name in CORRUPTIONS:
        values = np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if values.shape != (50000, 32, 32, 3) or values.dtype != np.uint8:
            raise ValueError(f"Unexpected corruption data: {name}")
    with run_lock(run_dir):
        model, meta, manifest, checkpoint = completed_model(run_dir, manifest_path, device_name)
        device = torch.device(device_name)
        clean = CIFAR10(str(data_root), train=False, download=False, transform=transform(manifest))
        if not np.array_equal(np.asarray(clean.targets), labels):
            raise ValueError("Corruption labels do not match the clean test labels")
        batch_size = meta["protocol"]["batch_size"]
        clean_accuracy = validation_accuracy(model, DataLoader(clean, batch_size=batch_size), device)
        scores, rows = {}, []
        for name in CORRUPTIONS:
            array = np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for severity in range(1, 6):
                dataset = CorruptionDataset(array, labels, severity, transform(manifest))
                accuracy = validation_accuracy(model, DataLoader(dataset, batch_size=batch_size), device)
                scores[name, severity] = accuracy
                rows.append(dict(corruption=name, severity=severity, accuracy=accuracy, samples=10000))
        report = dict(schema=SCHEMA_VERSION, phase="post_search", training_identity=checkpoint["identity"],
                      artifact_id=checkpoint_fingerprint(Path(run_dir) / "checkpoint.pt"),
                      epoch=checkpoint["epoch"], manifest_id=manifest["id"], candidate=meta["candidate"],
                      metrics=robustness_metrics(clean_accuracy, scores), conditions=rows)
        write_json(destination, report)
        return report


def checkpoint_fingerprint(path):
    import hashlib
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate_clean(run_dir, manifest_path, data_root, output, device_name="cuda"):
    if Path(output).exists():
        raise FileExistsError("Evaluation output exists; use a new output path")
    with run_lock(run_dir):
        model, meta, manifest, checkpoint = completed_model(run_dir, manifest_path, device_name)
        if manifest["dataset"] == "image-folder":
            from .folder_data import FolderDataset
            dataset = FolderDataset(data_root, manifest, "test", transform(manifest))
        else:
            from torchvision.datasets import CIFAR10
            dataset = CIFAR10(str(data_root), train=False, download=False, transform=transform(manifest))
        score = validation_accuracy(model, DataLoader(dataset, batch_size=meta["protocol"]["batch_size"]), torch.device(device_name))
        report = dict(schema=SCHEMA_VERSION, phase="post_search", evaluation="clean_only",
                      artifact_id=checkpoint_fingerprint(Path(run_dir) / "checkpoint.pt"),
                      training_identity=checkpoint["identity"], epoch=checkpoint["epoch"],
                      manifest_id=manifest["id"], candidate=meta["candidate"],
                      metrics=dict(clean_accuracy=score), samples=len(dataset), conditions=[])
        write_json(output, report)
        return report


def export_run(run_dir, manifest_path, destination):
    output = Path(destination)
    if output.exists():
        raise FileExistsError("Export exists; select a new destination")
    with run_lock(run_dir):
        model, meta, manifest, checkpoint = completed_model(run_dir, manifest_path, "cpu")
        save_checkpoint(output, dict(schema=SCHEMA_VERSION, kind="cs400-model-export", model=model.state_dict(),
                                    candidate=meta["candidate"], classes=manifest["classes"],
                                    preprocessing={**{k: manifest[k] for k in ("image_size", "mean", "std")},
                                                   "dataset": manifest["dataset"], "resize": "bilinear_stretch",
                                                   "color": "exif_transposed_rgb"},
                                    epoch=checkpoint["epoch"], training_identity=checkpoint["identity"]))

