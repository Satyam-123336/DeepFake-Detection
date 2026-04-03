from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BlinkFeatures:
    blink_count: int
    average_blink_duration: float
    average_inter_blink_interval: float
    irregularity_score: float


def summarize_blinks(blink_windows: list[tuple[float, float]]) -> BlinkFeatures:
    if not blink_windows:
        return BlinkFeatures(0, 0.0, 0.0, 1.0)

    durations = [end - start for start, end in blink_windows]
    intervals = [
        blink_windows[index][0] - blink_windows[index - 1][1]
        for index in range(1, len(blink_windows))
    ]

    average_duration = sum(durations) / len(durations)
    average_interval = sum(intervals) / len(intervals) if intervals else 0.0
    irregularity = 0.0
    if intervals and average_interval > 0:
        irregularity = sum(abs(interval - average_interval) for interval in intervals) / len(intervals)
        irregularity /= average_interval

    return BlinkFeatures(len(blink_windows), average_duration, average_interval, irregularity)