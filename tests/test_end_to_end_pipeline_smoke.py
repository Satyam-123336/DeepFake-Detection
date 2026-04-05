from pathlib import Path

import cv2
import numpy as np

from src.pipeline.run_full_pipeline import run_phase_four_pipeline


def test_end_to_end_pipeline_smoke_on_synthetic_video(tmp_path: Path) -> None:
    video_path = tmp_path / "smoke.mp4"
    processed_dir = tmp_path / "processed"

    width, height, fps, frame_count = 320, 240, 10, 20
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for idx in range(frame_count):
        frame = np.full((height, width, 3), 30 + idx, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    result = run_phase_four_pipeline(video_path, processed_dir)
    payload = result.to_dict()

    assert "preprocessing" in payload
    assert "behavioral" in payload
    assert "visual" in payload
    assert "watermark" in payload
    assert "nlp" in payload
    assert "scoring" in payload
    assert payload["scoring"]["risk_level"] in {"low", "medium", "high"}
