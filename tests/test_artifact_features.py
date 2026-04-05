from pathlib import Path

import cv2
import numpy as np

from src.visual.artifact_features import extract_artifact_features


def test_artifact_features_extracts_basic_statistics(tmp_path: Path) -> None:
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    image_path = tmp_path / "face.jpg"
    cv2.imwrite(str(image_path), image)

    features = extract_artifact_features(image_path)
    assert features.brightness_variance >= 0.0