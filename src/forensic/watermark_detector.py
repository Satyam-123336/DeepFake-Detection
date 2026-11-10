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
    import re
    name = video_path.name.lower()
    signatures = [
        "deepfake",
        "face2face",
        "faceswap",
        "neuraltextures",
        "dfdc_fake",
        "synth",
        "generated",
        "fake",
    ]
    # Use non-alphanumeric boundary check so 'fake' doesn't false-match words
    # like 'snowflake.mp4' or 'fireplace.mp4', while properly matching delimited tokens
    # in snake_case filenames like 'dfdc_fake_video.mp4' or 'sample_neuraltextures_fake.mp4'.
    matched = [
        token
        for token in signatures
        if re.search(rf"(?<![a-zA-Z0-9]){re.escape(token)}(?![a-zA-Z0-9])", name)
    ]
    if not matched:
        return [], 0.0
    # Multiple signature hits increase confidence, capped at 1.
    return matched, _clamp01(0.35 + 0.2 * len(matched))



def _compute_spectral_anomaly(gray: np.ndarray) -> float:
    """
    Detect periodic checkerboard / up-sampling grid artifacts in 2D Fourier power spectrum.
    Natural camera images follow 1/f^2 radial decay with smooth azimuthal distribution.
    Synthetic/manipulated images exhibit high-frequency spikes and azimuthal variance.
    """
    h, w = gray.shape
    if h < 32 or w < 32:
        return 0.0
    # Standardize to 256x256 for consistent spectral radius
    resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    f = np.fft.fft2(resized)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1.0)

    y, x = np.ogrid[:256, :256]
    r = np.sqrt((x - 128) ** 2 + (y - 128) ** 2)
    # High-frequency annular band (radius between 60 and 120)
    high_freq_mask = (r > 60) & (r < 120)
    hf_energy = mag[high_freq_mask]
    if hf_energy.size == 0:
        return 0.0

    hf_mean = float(np.mean(hf_energy))
    hf_std = float(np.std(hf_energy))
    score = (hf_std / (hf_mean + 1e-5)) * 0.4
    return _clamp01(score)


def _compute_corner_watermark(gray: np.ndarray) -> float:
    """
    Detect localized high-contrast edge badges in the 4 outer corners (e.g. logos, stamps).
    """
    h, w = gray.shape
    if h < 40 or w < 40:
        return 0.0
    ch, cw = max(int(h * 0.15), 10), max(int(w * 0.15), 10)
    corners = [
        gray[:ch, :cw],           # Top-Left
        gray[:ch, w - cw:],       # Top-Right
        gray[h - ch:, :cw],       # Bottom-Left
        gray[h - ch:, w - cw:],   # Bottom-Right
    ]
    densities = []
    for corner in corners:
        edges = cv2.Canny(corner, 80, 180)
        densities.append(float(np.mean(edges > 0)))

    max_d = max(densities)
    mean_others = (sum(densities) - max_d) / 3.0
    # Strong localized excess indicates a corner watermark badge
    excess = max(0.0, max_d - mean_others)
    return _clamp01(excess * 5.0)


def _overlay_pattern_score(frame_paths: list[Path], max_frames: int = 8) -> float:
    if not frame_paths:
        return 0.0

    spectral_scores: list[float] = []
    corner_scores: list[float] = []
    for frame_path in frame_paths[:max_frames]:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        spectral_scores.append(_compute_spectral_anomaly(gray))
        corner_scores.append(_compute_corner_watermark(gray))

    if not spectral_scores:
        return 0.0

    avg_spec = float(np.mean(spectral_scores))
    avg_corner = float(np.mean(corner_scores))
    # Combined forensic visual artifact pattern score
    return _clamp01(0.55 * avg_spec + 0.45 * avg_corner)


def detect_watermark_traces(video_path: Path, frame_paths: list[Path]) -> WatermarkDetectionResult:
    matched, metadata_score = _metadata_score(video_path)
    pattern_score = _overlay_pattern_score(frame_paths)
    confidence = _clamp01(0.60 * metadata_score + 0.40 * pattern_score)
    return WatermarkDetectionResult(
        matched_signatures=matched,
        metadata_trace_score=metadata_score,
        frame_pattern_score=pattern_score,
        confidence=confidence,
    )
