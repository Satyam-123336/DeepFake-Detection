from pathlib import Path

import csv
import numpy as np
from PIL import Image

from src.data.cnn_dataset import FaceImageDataset


def test_cnn_dataset_loads_face_records(tmp_path: Path) -> None:
    image_path = tmp_path / "face.jpg"
    Image.fromarray(np.full((32, 32, 3), 200, dtype=np.uint8)).save(image_path)

    csv_path = tmp_path / "faces.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "label",
                "label_name",
                "source_video",
                "frame_timestamp",
                "sharpness_score",
                "texture_score",
                "brightness_variance",
                "lighting_asymmetry",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": str(image_path),
                "label": 1,
                "label_name": "fake",
                "source_video": "video.mp4",
                "frame_timestamp": 0.5,
                "sharpness_score": 1.0,
                "texture_score": 1.0,
                "brightness_variance": 1.0,
                "lighting_asymmetry": 1.0,
            }
        )

    dataset = FaceImageDataset(csv_path, image_size=64)
    image_tensor, label = dataset[0]
    assert tuple(image_tensor.shape) == (3, 64, 64)
    assert int(label.item()) == 1