from pathlib import Path
import numpy as np

from src.scoring.engine import _visual_suspicion
from src.visual.lighting_checks import compute_lighting_asymmetry
from src.forensic.watermark_detector import _metadata_score
from src.nlp.transcription import _transcribe_with_energy_proxy


def test_visual_suspicion_returns_zero_when_no_face_detected():
    # When no face is detected in any keyframe, all metrics are None
    suspicion = _visual_suspicion(None, None, None, None)
    assert suspicion == 0.0, f"Expected 0.0 for no-face visual suspicion, got {suspicion}"


def test_watermark_metadata_does_not_flag_dfdc_real_filename():
    # Real videos from DFDC dataset should not trigger false positive watermark confidence
    matched, score = _metadata_score(Path("dfdc_real_video_001.mp4"))
    assert "dfdc" not in matched
    assert score == 0.0

    # Synthetic videos with explicitly fake names should trigger matched signatures
    matched_fake, score_fake = _metadata_score(Path("dfdc_fake_video_001.mp4"))
    assert "dfdc_fake" in matched_fake
    assert score_fake > 0.0


def test_lighting_asymmetry_handles_small_image_without_nan(tmp_path):
    # Create a 1x1 image that previously caused NaN due to empty array splits
    import cv2
    tiny_img_path = tmp_path / "tiny_face.png"
    cv2.imwrite(str(tiny_img_path), np.ones((1, 1), dtype=np.uint8) * 128)

    score = compute_lighting_asymmetry(tiny_img_path)
    assert not np.isnan(score)
    assert score == 0.0


def test_transcribe_handles_invalid_audio_path():
    result = _transcribe_with_energy_proxy(Path("non_existent_audio_file.wav"))
    assert result.method == "energy-proxy"
    assert result.duration_seconds == 0.0
    assert result.confidence == 0.0
