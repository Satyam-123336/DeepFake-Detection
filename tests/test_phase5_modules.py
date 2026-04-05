from pathlib import Path

import cv2
import numpy as np

from src.forensic.watermark_detector import detect_watermark_traces
from src.nlp.suspicion import score_nlp_suspicion
from src.nlp.transcription import TranscriptResult


def test_watermark_detector_uses_filename_signatures(tmp_path: Path) -> None:
    frame = np.full((64, 64, 3), 127, dtype=np.uint8)
    frame_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(frame_path), frame)

    video_path = tmp_path / "sample_neuraltextures_fake.mp4"
    video_path.write_bytes(b"placeholder")

    result = detect_watermark_traces(video_path, [frame_path])
    assert result.metadata_trace_score > 0.0
    assert result.confidence > 0.0


def test_nlp_suspicion_marks_uniform_speech_segments() -> None:
    transcript = TranscriptResult(
        transcript_text="proxy",
        method="energy-proxy",
        confidence=0.7,
        voiced_ratio=0.95,
        speech_segments=6,
        segment_durations=[0.8, 0.82, 0.79, 0.81, 0.8, 0.83],
        duration_seconds=10.0,
    )
    result = score_nlp_suspicion(transcript)
    assert result.score > 0.2
