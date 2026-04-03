from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(slots=True)
class FaceRecord:
    image_path: str
    label: int
    label_name: str
    source_video: str
    frame_timestamp: float
    sharpness_score: float
    texture_score: float
    brightness_variance: float
    lighting_asymmetry: float


class FaceImageDataset(Dataset):
    def __init__(self, csv_path: Path, image_size: int = 128, validate_paths: bool = False) -> None:
        self.csv_path = csv_path
        self.image_size = image_size
        self.records = self._load_records(csv_path, validate_paths=validate_paths)

    def _load_records(self, csv_path: Path, validate_paths: bool = False) -> list[FaceRecord]:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows: list[FaceRecord] = []
            for row in reader:
                if validate_paths and not Path(row["image_path"]).exists():
                    continue
                rows.append(
                    FaceRecord(
                        image_path=row["image_path"],
                        label=int(row["label"]),
                        label_name=row["label_name"],
                        source_video=row["source_video"],
                        frame_timestamp=float(row["frame_timestamp"]),
                        sharpness_score=float(row["sharpness_score"]),
                        texture_score=float(row["texture_score"]),
                        brightness_variance=float(row["brightness_variance"]),
                        lighting_asymmetry=float(row["lighting_asymmetry"]),
                    )
                )
        return rows

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        label = torch.tensor(record.label, dtype=torch.long)
        return tensor, label