from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.data.dataset_manifest import read_video_records_csv
from src.pipeline.run_preprocessing import run_preprocessing
from src.pipeline.run_visual import run_visual_analysis
from src.utils.io import ensure_dir

_FACE_CSV_FIELDS = [
    "image_path",
    "label",
    "label_name",
    "source_video",
    "frame_timestamp",
    "sharpness_score",
    "texture_score",
    "brightness_variance",
    "lighting_asymmetry",
]


def _read_existing_manifest_state(face_manifest_path: Path) -> tuple[set[str], int]:
    """Return (already_processed_source_videos, existing_face_rows)."""
    if not face_manifest_path.exists() or face_manifest_path.stat().st_size == 0:
        return set(), 0

    processed_videos: set[str] = set()
    face_rows = 0
    with face_manifest_path.open("r", newline="", encoding="utf-8") as csv_handle:
        reader = csv.DictReader(csv_handle)
        for row in reader:
            source_video = row.get("source_video")
            if source_video:
                processed_videos.add(source_video)
            face_rows += 1
    return processed_videos, face_rows


def _write_status(status_path: Path | None, payload: dict) -> None:
    if status_path is None:
        return
    ensure_dir(status_path.parent)
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass(slots=True)
class FaceManifestRecord:
    image_path: str
    label: int
    label_name: str
    source_video: str
    frame_timestamp: float
    sharpness_score: float
    texture_score: float
    brightness_variance: float
    lighting_asymmetry: float


def preprocess_manifest_to_faces(
    manifest_path: Path,
    processed_dir: Path,
    face_manifest_path: Path,
    *,
    split_name: str | None = None,
    status_path: Path | None = None,
) -> dict:
    records = read_video_records_csv(manifest_path)
    total_videos = len(records)
    ensure_dir(face_manifest_path.parent)
    processed_videos = 0
    skipped_videos = 0
    already_processed_videos, existing_faces = _read_existing_manifest_state(face_manifest_path)
    total_faces = existing_faces

    _write_status(
        status_path,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "starting",
            "split": split_name,
            "manifest_path": str(manifest_path),
            "face_manifest_path": str(face_manifest_path),
            "total_videos_in_split": total_videos,
            "already_processed_videos": len(already_processed_videos),
            "processed_videos_this_run": processed_videos,
            "skipped_videos_this_run": skipped_videos,
            "total_faces": total_faces,
        },
    )

    if already_processed_videos:
        print(
            f"Resuming from existing manifest: {len(already_processed_videos)} videos, {existing_faces} faces.",
            flush=True,
        )

    # Open CSV once and write incrementally so progress is saved even if interrupted.
    file_exists = face_manifest_path.exists() and face_manifest_path.stat().st_size > 0
    write_mode = "a" if file_exists else "w"
    with face_manifest_path.open(write_mode, newline="", encoding="utf-8") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=_FACE_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
            csv_handle.flush()

        for index, record in enumerate(records, start=1):
            video_name = Path(record.video_path).name
            print(
                f"[{index}/{total_videos}] {record.label_name} | {video_name}",
                flush=True,
            )

            _write_status(
                status_path,
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "stage": "processing",
                    "split": split_name,
                    "manifest_path": str(manifest_path),
                    "face_manifest_path": str(face_manifest_path),
                    "current_video_index": index,
                    "total_videos_in_split": total_videos,
                    "current_video": video_name,
                    "processed_videos_this_run": processed_videos,
                    "skipped_videos_this_run": skipped_videos,
                    "total_faces": total_faces,
                },
            )

            if record.video_path in already_processed_videos:
                print("  already processed in previous run; skipping", flush=True)
                _write_status(
                    status_path,
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "stage": "processing",
                        "split": split_name,
                        "manifest_path": str(manifest_path),
                        "face_manifest_path": str(face_manifest_path),
                        "current_video_index": index,
                        "total_videos_in_split": total_videos,
                        "current_video": video_name,
                        "processed_videos_this_run": processed_videos,
                        "skipped_videos_this_run": skipped_videos,
                        "total_faces": total_faces,
                        "note": "already processed in previous run; skipped",
                    },
                )
                continue

            try:
                preprocessing_result = run_preprocessing(Path(record.video_path), processed_dir)
            except Exception as exc:
                print(f"  SKIP (preprocessing error): {exc}", flush=True)
                skipped_videos += 1
                _write_status(
                    status_path,
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "stage": "processing",
                        "split": split_name,
                        "manifest_path": str(manifest_path),
                        "face_manifest_path": str(face_manifest_path),
                        "current_video_index": index,
                        "total_videos_in_split": total_videos,
                        "current_video": video_name,
                        "processed_videos_this_run": processed_videos,
                        "skipped_videos_this_run": skipped_videos,
                        "total_faces": total_faces,
                        "note": f"preprocessing error: {exc}",
                    },
                )
                continue

            video_faces = 0
            for key_frame in preprocessing_result.key_frames:
                try:
                    visual_result = run_visual_analysis(
                        key_frame.image_path,
                        preprocessing_result.artifact_dir / "faces",
                    )
                except Exception as exc:
                    print(f"  frame {key_frame.frame_index} skipped: {exc}", flush=True)
                    continue

                if (
                    visual_result.face_path is None
                    or visual_result.artifact_features is None
                    or visual_result.lighting_asymmetry is None
                ):
                    continue

                row = FaceManifestRecord(
                    image_path=str(visual_result.face_path.resolve()),
                    label=record.label,
                    label_name=record.label_name,
                    source_video=record.video_path,
                    frame_timestamp=key_frame.timestamp_seconds,
                    sharpness_score=visual_result.artifact_features.sharpness_score,
                    texture_score=visual_result.artifact_features.texture_score,
                    brightness_variance=visual_result.artifact_features.brightness_variance,
                    lighting_asymmetry=visual_result.lighting_asymmetry,
                )
                writer.writerow(asdict(row))
                video_faces += 1

            csv_handle.flush()
            processed_videos += 1
            total_faces += video_faces
            already_processed_videos.add(record.video_path)
            print(f"  faces found: {video_faces}  |  running total: {total_faces}", flush=True)
            _write_status(
                status_path,
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "stage": "processing",
                    "split": split_name,
                    "manifest_path": str(manifest_path),
                    "face_manifest_path": str(face_manifest_path),
                    "current_video_index": index,
                    "total_videos_in_split": total_videos,
                    "current_video": video_name,
                    "processed_videos_this_run": processed_videos,
                    "skipped_videos_this_run": skipped_videos,
                    "total_faces": total_faces,
                    "last_video_faces": video_faces,
                },
            )

    print(
        f"\nDone. processed={processed_videos}  skipped={skipped_videos}  total_faces={total_faces}",
        file=sys.stderr,
        flush=True,
    )
    _write_status(
        status_path,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "completed",
            "split": split_name,
            "manifest_path": str(manifest_path),
            "face_manifest_path": str(face_manifest_path),
            "total_videos_in_split": total_videos,
            "processed_videos_this_run": processed_videos,
            "skipped_videos_this_run": skipped_videos,
            "total_faces": total_faces,
        },
    )
    return {
        "manifest_path": str(face_manifest_path),
        "processed_videos": processed_videos,
        "face_rows": total_faces,
    }