from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.utils.io import ensure_dir

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(slots=True)
class IngestionStats:
    source_name: str
    real_count: int
    fake_count: int
    skipped_count: int


def _copy_with_unique_name(source_path: Path, destination_dir: Path, prefix: str) -> Path:
    ensure_dir(destination_dir)
    stem = source_path.stem
    suffix = source_path.suffix.lower()
    destination_path = destination_dir / f"{prefix}_{stem}{suffix}"
    index = 1
    while destination_path.exists():
        destination_path = destination_dir / f"{prefix}_{stem}_{index}{suffix}"
        index += 1
    shutil.copy2(source_path, destination_path)
    return destination_path


def _list_video_files(root_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(root_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]


def ingest_labeled_directory(
    source_dir: Path,
    label_name: str,
    raw_dir: Path,
    max_videos: int | None = None,
    prefix: str = "custom",
) -> IngestionStats:
    if label_name not in {"real", "fake"}:
        raise ValueError("label_name must be either 'real' or 'fake'.")

    destination = ensure_dir(raw_dir / label_name)
    files = _list_video_files(source_dir)
    copied = 0

    for file_path in files:
        _copy_with_unique_name(file_path, destination, prefix)
        copied += 1
        if max_videos is not None and copied >= max_videos:
            break

    if label_name == "real":
        return IngestionStats(prefix, copied, 0, max(0, len(files) - copied))
    return IngestionStats(prefix, 0, copied, max(0, len(files) - copied))


def ingest_faceforensicspp(
    ffpp_root: Path,
    raw_dir: Path,
    max_real: int | None = None,
    max_fake: int | None = None,
    prefix: str = "ffpp",
) -> IngestionStats:
    original_dir = ffpp_root / "original_sequences"
    manipulated_dir = ffpp_root / "manipulated_sequences"

    if not original_dir.exists() or not manipulated_dir.exists():
        raise FileNotFoundError(
            "FaceForensics++ root must contain original_sequences and manipulated_sequences."
        )

    real_files = _list_video_files(original_dir)
    fake_files = _list_video_files(manipulated_dir)

    real_destination = ensure_dir(raw_dir / "real")
    fake_destination = ensure_dir(raw_dir / "fake")

    real_count = 0
    for file_path in real_files:
        _copy_with_unique_name(file_path, real_destination, prefix)
        real_count += 1
        if max_real is not None and real_count >= max_real:
            break

    fake_count = 0
    for file_path in fake_files:
        _copy_with_unique_name(file_path, fake_destination, prefix)
        fake_count += 1
        if max_fake is not None and fake_count >= max_fake:
            break

    skipped = max(0, len(real_files) - real_count) + max(0, len(fake_files) - fake_count)
    return IngestionStats("faceforensicspp", real_count, fake_count, skipped)


def ingest_dfdc(
    dfdc_root: Path,
    raw_dir: Path,
    max_real: int | None = None,
    max_fake: int | None = None,
    prefix: str = "dfdc",
) -> IngestionStats:
    metadata_files = sorted(dfdc_root.rglob("metadata.json"))
    if not metadata_files:
        raise FileNotFoundError("No metadata.json files found under the provided DFDC root.")

    real_destination = ensure_dir(raw_dir / "real")
    fake_destination = ensure_dir(raw_dir / "fake")

    real_count = 0
    fake_count = 0
    skipped = 0

    for metadata_path in metadata_files:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        base_dir = metadata_path.parent
        for file_name, record in metadata.items():
            label = str(record.get("label", "")).upper()
            source_video = base_dir / file_name
            if not source_video.exists() or source_video.suffix.lower() not in VIDEO_EXTENSIONS:
                skipped += 1
                continue

            if label == "REAL":
                if max_real is not None and real_count >= max_real:
                    skipped += 1
                    continue
                _copy_with_unique_name(source_video, real_destination, prefix)
                real_count += 1
            elif label == "FAKE":
                if max_fake is not None and fake_count >= max_fake:
                    skipped += 1
                    continue
                _copy_with_unique_name(source_video, fake_destination, prefix)
                fake_count += 1
            else:
                skipped += 1

    return IngestionStats("dfdc", real_count, fake_count, skipped)