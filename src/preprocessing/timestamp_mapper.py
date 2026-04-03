from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from src.preprocessing.keyframe_extractor import KeyFrame
from src.utils.io import write_json


@dataclass(slots=True)
class TimestampMapping:
    frame_index: int
    timestamp_seconds: float
    image_path: str


def save_timestamp_map(key_frames: list[KeyFrame], output_path: Path) -> None:
    payload = {
        "frames": [
            asdict(TimestampMapping(k.frame_index, k.timestamp_seconds, str(k.image_path)))
            for k in key_frames
        ]
    }
    write_json(output_path, payload)