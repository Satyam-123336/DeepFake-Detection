from __future__ import annotations

import json
from pathlib import Path

import cv2
import torch
from PIL import Image

from models.cnn.architecture import LightweightArtifactCNN


def _calibration_path(weights_path: Path) -> Path:
    return weights_path.with_suffix(".calibration.json")


def load_fake_threshold(weights_path: Path) -> float:
    calibration_path = _calibration_path(weights_path)
    if not calibration_path.exists():
        return 0.5
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0.5

    threshold = payload.get("fake_probability_threshold", 0.5)
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, threshold_value))


def load_model(weights_path: Path) -> LightweightArtifactCNN:
    model = LightweightArtifactCNN()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_face_image(face_image_path: Path, weights_path: Path) -> tuple[int, float, float]:
    model = load_model(weights_path)
    fake_threshold = load_fake_threshold(weights_path)
    image = cv2.imread(str(face_image_path))
    if image is None:
        raise ValueError(f"Unable to read face image: {face_image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = Image.fromarray(rgb).resize((128, 128))
    tensor = torch.tensor(list(resized.getdata()), dtype=torch.float32).view(128, 128, 3)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        if probabilities.shape[1] > 1:
            fake_probability = float(probabilities[0, 1].item())
            predicted_class = 1 if fake_probability >= fake_threshold else 0
            confidence = fake_probability if predicted_class == 1 else (1.0 - fake_probability)
        else:
            predicted_class = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(torch.max(probabilities).item())
            fake_probability = confidence

    return predicted_class, confidence, fake_probability