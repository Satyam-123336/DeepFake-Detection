from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class WatermarkDetectionResult:
    matched_signatures: list[str]
    metadata_trace_score: float
    frame_pattern_score: float
    confidence: float


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _metadata_score(video_path: Path) -> tuple[list[str], float]:
    name = video_path.name.lower()
    signatures = [
        "deepfake",
        "face2face",
        "faceswap",
        "neuraltextures",
        "dfdc",
        "synth",
        "generated",
        "fake",
    ]
    matched = [token for token in signatures if token in name]
    if not matched:
        return [], 0.0
    # Multiple signature hits increase confidence, capped at 1.
    return matched, _clamp01(0.35 + 0.2 * len(matched))


def _overlay_pattern_score(frame_paths: list[Path], max_frames: int = 8) -> float:
    if not frame_paths:
        return 0.0

    scores: list[float] = []
    for frame_path in frame_paths[:max_frames]:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < 20 or w < 20:
            continue

        strip_h = max(1, int(h * 0.12))
        top = gray[:strip_h, :]
        center = gray[h // 3 : (2 * h) // 3, :]
        bottom = gray[h - strip_h :, :]

        top_edges = cv2.Canny(top, 100, 200)
        center_edges = cv2.Canny(center, 100, 200)
        bottom_edges = cv2.Canny(bottom, 100, 200)

        top_density = float(np.mean(top_edges > 0))
        center_density = float(np.mean(center_edges > 0))
        bottom_density = float(np.mean(bottom_edges > 0))

        # Visible watermarks commonly increase high-contrast patterns in top/bottom overlays.
        edge_delta = max(top_density, bottom_density) - center_density
        scores.append(_clamp01(edge_delta * 6.0))

    if not scores:
        return 0.0
    return float(np.mean(scores))


def detect_watermark_traces(video_path: Path, frame_paths: list[Path]) -> WatermarkDetectionResult:
    matched, metadata_score = _metadata_score(video_path)
    overlay_score = _overlay_pattern_score(frame_paths)
    confidence = _clamp01(0.65 * metadata_score + 0.35 * overlay_score)
    return WatermarkDetectionResult(
        matched_signatures=matched,
        metadata_trace_score=metadata_score,
        frame_pattern_score=overlay_score,
        confidence=confidence,
    )
