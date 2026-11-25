"""Inference from a portable export; never constructs an optimizer."""
from PIL import Image, ImageOps
import torch
from .config import Candidate, SCHEMA_VERSION
from .model import ResNet


def predict_image(model_path, image_path, device_name="cpu"):
    from torchvision import transforms as T
    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    if artifact.get("schema") != SCHEMA_VERSION or artifact.get("kind") != "cs400-model-export":
        raise ValueError("Expected a CS400 model export")
    classes = artifact["classes"]
    settings = artifact["preprocessing"]
    model = ResNet(Candidate(**artifact["candidate"]), len(classes))
    model.load_state_dict(artifact["model"])
    device = torch.device(device_name)
    model.to(device).eval()
    preprocess = T.Compose([T.Resize((settings["image_size"],)*2, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                            T.ToTensor(), T.Normalize(settings["mean"], settings["std"])])
    with Image.open(image_path) as source:
        tensor = preprocess(ImageOps.exif_transpose(source).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores = model(tensor)
        if not torch.isfinite(scores).all():
            raise ValueError("Model produced non-finite outputs")
        probabilities = scores.softmax(1)[0].cpu().tolist()
    ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
    return dict(prediction=ranked[0][0], top_predictions=[dict(label=label, probability=p) for label, p in ranked[:5]],
                note="Softmax scores are not calibrated confidence or an out-of-distribution detector")
