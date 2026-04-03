from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from src.utils.io import ensure_dir


@dataclass(slots=True)
class KeyFrame:
    frame_index: int
    timestamp_seconds: float
    image_path: Path


def extract_key_frames(video_path: Path, output_dir: Path, interval_seconds: float = 0.5) -> list[KeyFrame]:
    ensure_dir(output_dir)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        capture.release()
        raise ValueError("Video FPS could not be determined.")

    frame_step = max(int(fps * interval_seconds), 1)
    key_frames: list[KeyFrame] = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % frame_step == 0:
            output_path = output_dir / f"{video_path.stem}_frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            key_frames.append(KeyFrame(frame_index, frame_index / fps, output_path))

        frame_index += 1

    capture.release()
    return key_frames