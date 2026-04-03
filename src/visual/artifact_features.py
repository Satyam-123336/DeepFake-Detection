from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class ArtifactFeatures:
    sharpness_score: float
    texture_score: float
    brightness_variance: float


def extract_artifact_features(face_image_path: Path) -> ArtifactFeatures:
    image = cv2.imread(str(face_image_path))
    if image is None:
        raise ValueError(f"Unable to read face image: {face_image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    texture = float(np.std(gray))
    brightness_variance = float(np.var(gray))
    return ArtifactFeatures(sharpness, texture, brightness_variance)