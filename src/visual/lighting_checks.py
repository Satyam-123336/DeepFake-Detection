from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def compute_lighting_asymmetry(face_image_path: Path) -> float:
    image = cv2.imread(str(face_image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read face image: {face_image_path}")

    midpoint = image.shape[1] // 2
    left = image[:, :midpoint]
    right = image[:, midpoint:]
    right = cv2.flip(right, 1)

    min_width = min(left.shape[1], right.shape[1])
    left = left[:, :min_width]
    right = right[:, :min_width]

    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))