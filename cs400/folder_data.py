"""Explicit train/validation/test image folders with immutable content records."""
import hashlib
from pathlib import Path, PurePosixPath
import re
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from .config import SCHEMA_VERSION, digest
from .storage import write_json

SPLITS = ("train", "validation", "test")
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def image_record(path):
    raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as original:
        if original.format not in ("PNG", "JPEG") or getattr(original, "n_frames", 1) != 1:
            raise ValueError(f"Expected a single PNG/JPEG image: {path}")
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.load()
        pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    return raw_hash, pixels


def safe_image(root, relative):
    item = PurePosixPath(relative)
    if item.is_absolute() or len(item.parts) != 3 or any(p in (".", "..") or ":" in p or "\\" in p for p in item.parts):
        raise ValueError("Image paths must be split/class/file relative paths")
    base = Path(root).resolve()
    path = base.joinpath(*item.parts)
    if any(p.is_symlink() for p in (path, path.parent, path.parent.parent)) or not path.resolve().is_relative_to(base):
        raise ValueError("Symbolic links and paths outside the dataset are unsupported")
    return path


def validate_folder_manifest(manifest):
    classes = manifest.get("classes", [])
    if len(classes) < 2 or classes != sorted(set(classes)) or any(not isinstance(c, str) or not c or any(x in c for x in "/\\:") or c in (".", "..") for c in classes):
        raise ValueError("Custom classification needs at least two distinct, sorted class names")
    if manifest.get("image_size") != 32 or manifest.get("mean") != [0.5]*3 or manifest.get("std") != [0.5]*3:
        raise ValueError("Unsupported custom preprocessing")
    if manifest.get("augmentation_confirmed") is not True:
        raise ValueError("Crop, horizontal flip and erasing must be confirmed label-preserving")
    seen_paths, seen_pixels = set(), {}
    for split in SPLITS:
        rows = manifest.get(split, [])
        if not rows or {r.get("label") for r in rows} != set(range(len(classes))):
            raise ValueError(f"Each class must occur in {split}")
        for row in rows:
            label = row.get("label")
            if type(label) is not int or not 0 <= label < len(classes):
                raise ValueError("Invalid class index")
            path = row.get("path", "")
            safe_image(".", path)  # structural validation only; no image is read
            parts = PurePosixPath(path).parts
            if parts[:2] != (split, classes[label]) or PurePosixPath(path).suffix.lower() not in EXTENSIONS or path in seen_paths:
                raise ValueError("Invalid or duplicate image path")
            seen_paths.add(path)
            for key in ("sha256", "pixel_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", row.get(key, "")):
                    raise ValueError("Invalid image fingerprint")
            previous = seen_pixels.setdefault(row["pixel_sha256"], (split, label))
            if previous != (split, label):
                raise ValueError("Identical decoded images cross dataset splits or labels")


def prepare_folder(root, destination, augmentation_confirmed=False):
    if Path(destination).exists():
        raise FileExistsError("Manifest exists; choose a new path")
    if not augmentation_confirmed:
        raise ValueError("Confirm that 32x32 resizing, crop, horizontal flip and erasing preserve your labels")
    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)
    classes = sorted(p.name for p in (base / "train").iterdir() if p.is_dir())
    result = dict(schema=SCHEMA_VERSION, dataset="image-folder", classes=classes,
                  image_size=32, mean=[0.5]*3, std=[0.5]*3, augmentation_confirmed=True,
                  resize="bilinear_stretch", color="exif_transposed_rgb")
    for split in SPLITS:
        directory = base / split
        if sorted(p.name for p in directory.iterdir() if p.is_dir()) != classes:
            raise ValueError(f"Class folders differ in {split}")
        if any(not p.is_dir() for p in directory.iterdir()):
            raise ValueError(f"Place images inside class folders in {split}")
        rows = []
        for label, name in enumerate(classes):
            for file in sorted((directory / name).iterdir()):
                if not file.is_file() or file.suffix.lower() not in EXTENSIONS:
                    raise ValueError(f"Unsupported image entry: {file}")
                relative = file.relative_to(base).as_posix()
                path = safe_image(base, relative)
                raw, pixels = image_record(path)
                rows.append(dict(path=relative, label=label, sha256=raw, pixel_sha256=pixels))
        result[split] = rows
    validate_folder_manifest(result)
    result["id"] = digest(result)
    write_json(destination, result)
    return result


class FolderDataset(Dataset):
    def __init__(self, root, manifest, split, preprocess):
        if split not in SPLITS:
            raise ValueError("Unknown dataset split")
        self.root, self.rows, self.preprocess = Path(root), manifest[split], preprocess
        # Verify only the requested split. Training never reads test images.
        for row in self.rows:
            path = safe_image(self.root, row["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Dataset changed after preparation: {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(safe_image(self.root, row["path"])) as image:
            value = ImageOps.exif_transpose(image).convert("RGB")
            return self.preprocess(value), row["label"]
