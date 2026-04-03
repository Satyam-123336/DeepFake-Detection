from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from src.utils.io import ensure_dir

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(slots=True)
class VideoRecord:
    video_path: str
    label: int
    label_name: str
    language: str = "unknown"
    lighting_quality: str = "unknown"
    face_visibility: str = "unknown"
    speaking_state: str = "unknown"


def _safe_video_metadata(video_path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0, 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return width, height


def _infer_speaking_state(video_path: Path) -> str:
    name = video_path.stem.lower()
    speaking_tokens = ("talk", "speech", "podium", "meeting", "interview")
    nonspeaking_tokens = ("still", "walk", "hugging", "pan")
    if any(token in name for token in speaking_tokens):
        return "speaking"
    if any(token in name for token in nonspeaking_tokens):
        return "non_speaking"
    return "unknown"


def _infer_lighting_quality(video_path: Path) -> str:
    name = video_path.stem.lower()
    if any(token in name for token in ("dark", "night", "lowlight")):
        return "low"
    if any(token in name for token in ("bright", "outdoor", "sun")):
        return "high"
    return "normal"


def _infer_face_visibility(video_path: Path, width: int, height: int) -> str:
    name = video_path.stem.lower()
    if any(token in name for token in ("close", "portrait", "selfie")):
        return "high"
    if width >= 1280 or height >= 720:
        return "medium"
    return "unknown"


def scan_raw_video_dataset(raw_dir: Path, max_videos: int | None = None) -> list[VideoRecord]:
    """Scan raw/real and raw/fake directories.  ``max_videos`` is applied *per class*."""
    records: list[VideoRecord] = []
    for label_name, label in (("real", 0), ("fake", 1)):
        split_dir = raw_dir / label_name
        if not split_dir.exists():
            continue

        class_count = 0
        for path in sorted(split_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            width, height = _safe_video_metadata(path)
            records.append(
                VideoRecord(
                    video_path=str(path.resolve()),
                    label=label,
                    label_name=label_name,
                    language="unknown",
                    lighting_quality=_infer_lighting_quality(path),
                    face_visibility=_infer_face_visibility(path, width, height),
                    speaking_state=_infer_speaking_state(path),
                )
            )
            class_count += 1
            if max_videos is not None and class_count >= max_videos:
                break
    return records


def write_video_records_csv(output_path: Path, records: list[VideoRecord]) -> Path:
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_path",
                "label",
                "label_name",
                "language",
                "lighting_quality",
                "face_visibility",
                "speaking_state",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return output_path


def read_video_records_csv(manifest_path: Path) -> list[VideoRecord]:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            VideoRecord(
                video_path=row["video_path"],
                label=int(row["label"]),
                label_name=row["label_name"],
                language=row.get("language", "unknown"),
                lighting_quality=row.get("lighting_quality", "unknown"),
                face_visibility=row.get("face_visibility", "unknown"),
                speaking_state=row.get("speaking_state", "unknown"),
            )
            for row in reader
        ]
