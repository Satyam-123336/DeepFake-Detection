from __future__ import annotations

from dataclasses import dataclass

from src.behavioral.blink.blink_features import BlinkFeatures, summarize_blinks


@dataclass(slots=True)
class BlinkDetectionResult:
    ear_timeline: list[tuple[float, float]]
    blink_windows: list[tuple[float, float]]
    features: BlinkFeatures


class BlinkDetector:
    def __init__(self, ear_threshold: float = 0.21, min_consecutive_frames: int = 2) -> None:
        self.ear_threshold = ear_threshold
        self.min_consecutive_frames = min_consecutive_frames

    def detect(self, ear_timeline: list[tuple[float, float]]) -> BlinkDetectionResult:
        blink_windows: list[tuple[float, float]] = []
        current_start: float | None = None
        below_threshold_frames = 0

        for timestamp, ear_value in ear_timeline:
            if ear_value < self.ear_threshold:
                below_threshold_frames += 1
                if current_start is None:
                    current_start = timestamp
            else:
                if current_start is not None and below_threshold_frames >= self.min_consecutive_frames:
                    blink_windows.append((current_start, timestamp))
                current_start = None
                below_threshold_frames = 0

        if current_start is not None and below_threshold_frames >= self.min_consecutive_frames:
            blink_windows.append((current_start, ear_timeline[-1][0]))

        return BlinkDetectionResult(ear_timeline, blink_windows, summarize_blinks(blink_windows))