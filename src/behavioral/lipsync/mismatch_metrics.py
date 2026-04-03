from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LipSyncMetrics:
    best_offset_seconds: float
    correlation_score: float
    average_absolute_error: float