import numpy as np

from src.behavioral.lipsync.sync_analyzer import analyze_sync


def test_sync_analyzer_reports_strong_match_for_same_signal() -> None:
    signal = np.array([0.0, 1.0, 0.2, 0.7, 0.1], dtype=np.float32)
    result = analyze_sync(signal, signal, 0.04)
    assert result.correlation_score > 0.9