from pathlib import Path

from src.behavioral.blink.blink_detector import BlinkDetectionResult
from src.behavioral.blink.blink_features import BlinkFeatures
from src.pipeline.run_behavioral import BehavioralResult
from src.pipeline.run_full_pipeline import run_phase_four_pipeline
from src.pipeline.run_preprocessing import PreprocessingResult
from src.preprocessing.keyframe_extractor import KeyFrame
from src.preprocessing.video_loader import VideoMetadata
from src.pipeline.run_visual import VisualResult
from src.visual.artifact_features import ArtifactFeatures
from src.visual.cnn_inference import CNNInferenceResult


def test_run_full_pipeline_output_has_phase5_and_scoring(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "sample_fake_video.mp4"
    video_path.write_bytes(b"x")
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"x")

    metadata = VideoMetadata(
        path=video_path,
        fps=25.0,
        frame_count=250,
        width=640,
        height=360,
        duration_seconds=10.0,
    )
    preprocessing_result = PreprocessingResult(
        metadata=metadata,
        audio_path=None,
        key_frames=[KeyFrame(frame_index=0, timestamp_seconds=0.0, image_path=frame_path)],
        timestamp_map_path=tmp_path / "map.json",
        artifact_dir=tmp_path,
    )

    behavioral_result = BehavioralResult(
        blink_result=BlinkDetectionResult(
            ear_timeline=[(0.0, 0.2), (0.04, 0.19)],
            blink_windows=[(0.0, 0.04)],
            features=BlinkFeatures(
                blink_count=1,
                average_blink_duration=0.04,
                average_inter_blink_interval=0.0,
                irregularity_score=0.3,
            ),
        ),
        audio_available=False,
        lipsync_offset_seconds=0.0,
        lipsync_correlation=0.8,
        lipsync_error=0.2,
    )

    visual_result = VisualResult(
        face_path=frame_path,
        artifact_features=ArtifactFeatures(10.0, 20.0, 30.0),
        lighting_asymmetry=0.1,
        cnn_result=CNNInferenceResult(predicted_class=1, confidence=0.7, fake_probability=0.9),
    )

    monkeypatch.setattr("src.pipeline.run_full_pipeline.run_preprocessing", lambda *_args, **_kwargs: preprocessing_result)
    monkeypatch.setattr("src.pipeline.run_full_pipeline.run_behavioral_analysis", lambda *_args, **_kwargs: behavioral_result)
    monkeypatch.setattr("src.pipeline.run_full_pipeline.run_visual_analysis", lambda *_args, **_kwargs: visual_result)

    result = run_phase_four_pipeline(video_path, tmp_path)
    payload = result.to_dict()

    assert "watermark" in payload
    assert "transcript" in payload
    assert "nlp" in payload
    assert "scoring" in payload
    assert "risk_level" in payload["scoring"]
    assert "module_scores" in payload["scoring"]
    assert "cnn_fake_probability" in payload["visual"]
