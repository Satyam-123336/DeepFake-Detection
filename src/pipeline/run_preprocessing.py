from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.preprocessing.audio_extractor import extract_audio
from src.preprocessing.keyframe_extractor import KeyFrame, extract_key_frames
from src.preprocessing.timestamp_mapper import save_timestamp_map
from src.preprocessing.video_loader import VideoMetadata, load_video_metadata
from src.utils.io import ensure_dir


@dataclass(slots=True)
class PreprocessingResult:
    metadata: VideoMetadata
    audio_path: Path | None
    key_frames: list[KeyFrame]
    timestamp_map_path: Path
    artifact_dir: Path


def build_artifact_dir(video_path: Path, processed_dir: Path) -> Path:
    digest = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:8]
    safe_name = f"{video_path.stem}_{digest}"
    return ensure_dir(processed_dir / safe_name)


def run_preprocessing(video_path: Path, processed_dir: Path) -> PreprocessingResult:
    metadata = load_video_metadata(video_path)
    artifact_dir = build_artifact_dir(video_path, processed_dir)
    audio_dir = ensure_dir(artifact_dir / "audio")
    frames_dir = ensure_dir(artifact_dir / "frames")
    mapping_dir = ensure_dir(artifact_dir / "landmarks")

    audio_path = extract_audio(video_path, audio_dir)
    # Denser temporal sampling improves behavioral signals (blink and lip-sync).
    key_frames = extract_key_frames(video_path, frames_dir, interval_seconds=0.2)
    timestamp_map_path = mapping_dir / f"{video_path.stem}_timestamp_map.json"
    save_timestamp_map(key_frames, timestamp_map_path)
    return PreprocessingResult(metadata, audio_path, key_frames, timestamp_map_path, artifact_dir)