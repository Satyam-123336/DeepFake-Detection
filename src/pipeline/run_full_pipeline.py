from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.forensic.watermark_detector import WatermarkDetectionResult, detect_watermark_traces
from src.nlp.suspicion import NLPSuspicionResult, score_nlp_suspicion
from src.nlp.transcription import TranscriptResult, transcribe_audio_proxy
from src.pipeline.run_behavioral import BehavioralResult, run_behavioral_analysis
from src.pipeline.run_preprocessing import PreprocessingResult, run_preprocessing
from src.pipeline.run_visual import VisualResult, run_visual_analysis
from src.scoring.engine import FinalScoreResult, compute_final_score
from src.utils.io import ensure_dir


def _resolve_weights_path() -> Path | None:
    candidates = [
        Path("models/cnn/weights/lightweight_artifact_cnn_1000_fresh.pt"),
        Path("models/cnn/weights/lightweight_artifact_cnn.pt"),
        Path("models/cnn/weights/demo_lightweight_artifact_cnn.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@dataclass(slots=True)
class PhaseFourPipelineResult:
    preprocessing: PreprocessingResult
    behavioral: BehavioralResult
    visual: VisualResult
    watermark: WatermarkDetectionResult
    transcript: TranscriptResult
    nlp: NLPSuspicionResult
    scoring: FinalScoreResult

    def to_dict(self) -> dict:
        metadata = self.preprocessing.metadata
        return {
            "preprocessing": {
                "metadata": {
                    "path": str(metadata.path),
                    "fps": metadata.fps,
                    "frame_count": metadata.frame_count,
                    "width": metadata.width,
                    "height": metadata.height,
                    "duration_seconds": metadata.duration_seconds,
                },
                "audio_path": str(self.preprocessing.audio_path) if self.preprocessing.audio_path else None,
                "key_frame_count": len(self.preprocessing.key_frames),
                "timestamp_map_path": str(self.preprocessing.timestamp_map_path),
                "artifact_dir": str(self.preprocessing.artifact_dir),
            },
            "behavioral": {
                "blink_count": self.behavioral.blink_result.features.blink_count,
                "blink_irregularity": self.behavioral.blink_result.features.irregularity_score,
                "audio_available": self.behavioral.audio_available,
                "lipsync_offset_seconds": self.behavioral.lipsync_offset_seconds,
                "lipsync_correlation": self.behavioral.lipsync_correlation,
                "lipsync_error": self.behavioral.lipsync_error,
            },
            "visual": {
                "face_path": str(self.visual.face_path) if self.visual.face_path else None,
                "lighting_asymmetry": self.visual.lighting_asymmetry,
                "sharpness_score": self.visual.artifact_features.sharpness_score if self.visual.artifact_features else None,
                "texture_score": self.visual.artifact_features.texture_score if self.visual.artifact_features else None,
                "brightness_variance": self.visual.artifact_features.brightness_variance if self.visual.artifact_features else None,
                "cnn_confidence": self.visual.cnn_result.confidence if self.visual.cnn_result else None,
                "cnn_fake_probability": self.visual.cnn_result.fake_probability if self.visual.cnn_result else None,
            },
            "watermark": {
                "matched_signatures": self.watermark.matched_signatures,
                "metadata_trace_score": self.watermark.metadata_trace_score,
                "frame_pattern_score": self.watermark.frame_pattern_score,
                "confidence": self.watermark.confidence,
            },
            "transcript": {
                "text": self.transcript.transcript_text,
                "method": self.transcript.method,
                "confidence": self.transcript.confidence,
                "voiced_ratio": self.transcript.voiced_ratio,
                "speech_segments": self.transcript.speech_segments,
            },
            "nlp": {
                "suspicion_score": self.nlp.score,
                "reasons": self.nlp.reasons,
            },
            "scoring": {
                "confidence_score": self.scoring.confidence_score,
                "risk_level": self.scoring.risk_level,
                "module_scores": self.scoring.module_scores,
                "reasons": self.scoring.reasons,
            },
        }


def run_phase_four_pipeline(video_path: Path, processed_dir: Path) -> PhaseFourPipelineResult:
    processed_dir = ensure_dir(processed_dir)
    preprocessing_result = run_preprocessing(video_path, processed_dir)

    frame_paths = [item.image_path for item in preprocessing_result.key_frames]
    frame_timestamps = [item.timestamp_seconds for item in preprocessing_result.key_frames]
    behavioral_result = run_behavioral_analysis(frame_paths, frame_timestamps, preprocessing_result.audio_path)

    weights_path = _resolve_weights_path()
    visual_result = VisualResult(None, None, None, None)
    visual_candidates: list[VisualResult] = []
    for frame_path in frame_paths[:12]:
        candidate = run_visual_analysis(
            frame_path,
            preprocessing_result.artifact_dir / "faces",
            weights_path,
        )
        if candidate.face_path is not None:
            visual_candidates.append(candidate)

    if visual_candidates:
        def _candidate_score(item: VisualResult) -> float:
            cnn_fake = item.cnn_result.fake_probability if item.cnn_result else 0.0
            light = min(max((item.lighting_asymmetry or 0.0) / 80.0, 0.0), 1.0)
            return max(cnn_fake, 0.35 * light)

        visual_result = max(visual_candidates, key=_candidate_score)

    watermark_result = detect_watermark_traces(video_path, frame_paths)
    transcript_result = transcribe_audio_proxy(preprocessing_result.audio_path)
    nlp_result = score_nlp_suspicion(transcript_result)

    scoring_result = compute_final_score(
        duration_seconds=preprocessing_result.metadata.duration_seconds,
        blink_count=behavioral_result.blink_result.features.blink_count,
        blink_irregularity=behavioral_result.blink_result.features.irregularity_score,
        lipsync_offset_seconds=behavioral_result.lipsync_offset_seconds,
        lipsync_correlation=behavioral_result.lipsync_correlation,
        cnn_confidence=visual_result.cnn_result.fake_probability if visual_result.cnn_result else None,
        lighting_asymmetry=visual_result.lighting_asymmetry,
        sharpness_score=visual_result.artifact_features.sharpness_score if visual_result.artifact_features else None,
        texture_score=visual_result.artifact_features.texture_score if visual_result.artifact_features else None,
        watermark_confidence=watermark_result.confidence,
        nlp_suspicion_score=nlp_result.score,
    )

    return PhaseFourPipelineResult(
        preprocessing_result,
        behavioral_result,
        visual_result,
        watermark_result,
        transcript_result,
        nlp_result,
        scoring_result,
    )