from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CNNInferenceResult:
    predicted_class: int
    confidence: float
    fake_probability: float


def run_cnn_inference(face_image_path: Path, weights_path: Path) -> CNNInferenceResult:
    from models.cnn.infer import predict_face_image

    predicted_class, confidence, fake_probability = predict_face_image(face_image_path, weights_path)
    return CNNInferenceResult(predicted_class, confidence, fake_probability)