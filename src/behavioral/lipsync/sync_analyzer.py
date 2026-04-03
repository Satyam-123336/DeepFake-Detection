from __future__ import annotations

import numpy as np

from src.behavioral.lipsync.mismatch_metrics import LipSyncMetrics


def analyze_sync(audio_signal: np.ndarray, mouth_signal: np.ndarray, frame_step_seconds: float) -> LipSyncMetrics:
    if len(audio_signal) == 0 or len(mouth_signal) == 0:
        return LipSyncMetrics(0.0, 0.0, 1.0)

    min_len = min(len(audio_signal), len(mouth_signal))
    audio_signal = audio_signal[:min_len]
    mouth_signal = mouth_signal[:min_len]

    audio_norm = audio_signal - np.mean(audio_signal)
    mouth_norm = mouth_signal - np.mean(mouth_signal)

    correlation = np.correlate(audio_norm, mouth_norm, mode="full")
    best_index = int(np.argmax(correlation))
    lag = best_index - (min_len - 1)
    best_offset = lag * frame_step_seconds

    if np.std(audio_norm) == 0 or np.std(mouth_norm) == 0:
        corr_score = 0.0
    else:
        corr_score = float(np.max(correlation) / (min_len * np.std(audio_norm) * np.std(mouth_norm)))

    absolute_error = float(np.mean(np.abs(audio_norm - mouth_norm)))
    return LipSyncMetrics(best_offset, corr_score, absolute_error)