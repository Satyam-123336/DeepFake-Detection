from __future__ import annotations

import numpy as np

from src.behavioral.lipsync.mismatch_metrics import LipSyncMetrics


def analyze_sync(
    audio_signal: np.ndarray,
    mouth_signal: np.ndarray,
    frame_step_seconds: float,
    max_lag_seconds: float = 0.60,
) -> LipSyncMetrics:
    if len(audio_signal) == 0 or len(mouth_signal) == 0:
        return LipSyncMetrics(0.0, 0.0, 1.0)

    min_len = min(len(audio_signal), len(mouth_signal))
    audio_signal = audio_signal[:min_len]
    mouth_signal = mouth_signal[:min_len]

    audio_norm = audio_signal - np.mean(audio_signal)
    mouth_norm = mouth_signal - np.mean(mouth_signal)

    std_audio = np.std(audio_norm)
    std_mouth = np.std(mouth_norm)
    if std_audio == 0 or std_mouth == 0:
        return LipSyncMetrics(0.0, 0.0, 1.0)

    # Compute full cross-correlation
    correlation = np.correlate(audio_norm, mouth_norm, mode="full")
    lags = np.arange(-(min_len - 1), min_len) * frame_step_seconds

    # Constrain search window to physiologically plausible audio-visual delay:
    # Human audio-visual desync (due to codec delay, dubbing, or speech) is within +-0.60s.
    # Searching across 30+ seconds unconstrained produces spurious pseudo-peaks on long clips.
    max_lag = max(frame_step_seconds * 2, max_lag_seconds)
    valid_mask = np.abs(lags) <= max_lag

    windowed_corr = np.where(valid_mask, correlation, -np.inf)
    best_index = int(np.argmax(windowed_corr))
    best_offset = float(lags[best_index])

    denom = min_len * std_audio * std_mouth
    corr_score = float(max(0.0, windowed_corr[best_index] / denom))

    # If correlation in the plausible window is negligible (< 0.08), the signal is
    # inconclusive (ambient noise / non-speaking), so do not report a phantom offset.
    if corr_score < 0.08:
        best_offset = 0.0

    absolute_error = float(np.mean(np.abs(audio_norm - mouth_norm)))
    return LipSyncMetrics(best_offset, corr_score, absolute_error)