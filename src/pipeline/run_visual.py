from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.preprocessing.face_detector import FaceDetector
from src.visual.artifact_features import ArtifactFeatures, extract_artifact_features
from src.visual.cnn_inference import CNNInferenceResult, run_cnn_inference
from src.visual.face_cropper import crop_face
from src.visual.lighting_checks import compute_lighting_asymmetry


@dataclass(slots=True)
class VisualResult:
    face_path: Path | None
    artifact_features: ArtifactFeatures | None
    lighting_asymmetry: float | None
    cnn_result: CNNInferenceResult | None


def run_visual_analysis(frame_path: Path, faces_dir: Path, weights_path: Path | None = None) -> VisualResult:
    detector = FaceDetector()
    face_box = detector.detect(frame_path)
    if face_box is None:
        return VisualResult(None, None, None, None)

    face_path = crop_face(frame_path, face_box, faces_dir)
    features = extract_artifact_features(face_path)
    lighting = compute_lighting_asymmetry(face_path)

    cnn_result = None
    if weights_path is not None and weights_path.exists():
        cnn_result = run_cnn_inference(face_path, weights_path)

    return VisualResult(face_path, features, lighting, cnn_result)