from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.behavioral.blink.blink_detector import BlinkDetectionResult, BlinkDetector
from src.behavioral.blink.ear_calculator import calculate_eye_aspect_ratio
from src.behavioral.lipsync.audio_features import extract_audio_envelope
from src.behavioral.lipsync.sync_analyzer import analyze_sync
from src.behavioral.lipsync.viseme_features import compute_mouth_openness
from src.preprocessing.landmark_extractor import LandmarkExtractor
from src.utils.constants import LEFT_EYE_LANDMARKS, LOWER_LIP_LANDMARKS, RIGHT_EYE_LANDMARKS, UPPER_LIP_LANDMARKS


@dataclass(slots=True)
class BehavioralResult:
    blink_result: BlinkDetectionResult
    audio_available: bool
    lipsync_offset_seconds: float
    lipsync_correlation: float
    lipsync_error: float


def run_behavioral_analysis(frame_paths: list[Path], frame_timestamps: list[float], audio_path: Path | None) -> BehavioralResult:
    extractor = LandmarkExtractor()
    ear_timeline: list[tuple[float, float]] = []
    mouth_signal: list[float] = []
    mouth_timestamps: list[float] = []

    for frame_path, timestamp in zip(frame_paths, frame_timestamps, strict=True):
        landmarks = extractor.extract(frame_path)
        if landmarks is None:
            continue

        left_eye = [landmarks[index] for index in LEFT_EYE_LANDMARKS]
        right_eye = [landmarks[index] for index in RIGHT_EYE_LANDMARKS]
        left_ear = calculate_eye_aspect_ratio(left_eye)
        right_ear = calculate_eye_aspect_ratio(right_eye)
        ear_timeline.append((timestamp, (left_ear + right_ear) / 2.0))

        upper_lip = [landmarks[index] for index in UPPER_LIP_LANDMARKS]
        lower_lip = [landmarks[index] for index in LOWER_LIP_LANDMARKS]
        mouth_signal.append(compute_mouth_openness(upper_lip, lower_lip))
        mouth_timestamps.append(timestamp)

    blink_result = BlinkDetector().detect(ear_timeline)

    if audio_path is None:
        return BehavioralResult(blink_result, False, 0.0, 0.0, 1.0)

    audio_timestamps, audio_envelope = extract_audio_envelope(audio_path)
    if len(audio_timestamps) < 2 or not mouth_signal or not mouth_timestamps:
        return BehavioralResult(blink_result, True, 0.0, 0.0, 1.0)

    sampled_audio = np.interp(np.array(mouth_timestamps, dtype=np.float32), audio_timestamps, audio_envelope)
    frame_step = mouth_timestamps[1] - mouth_timestamps[0] if len(mouth_timestamps) > 1 else 0.04
    sync_metrics = analyze_sync(sampled_audio, np.array(mouth_signal, dtype=np.float32), frame_step)
    return BehavioralResult(
        blink_result,
        True,
        sync_metrics.best_offset_seconds,
        sync_metrics.correlation_score,
        sync_metrics.average_absolute_error,
    )