from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(slots=True)
class LandmarkPoint:
    x: float
    y: float


class LandmarkExtractor:
    def __init__(self, max_num_faces: int = 1) -> None:
        self._mesh = None
        self._face_detector = None

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if cascade_path.exists():
            detector = cv2.CascadeClassifier(str(cascade_path))
            self._face_detector = detector if not detector.empty() else None

        try:
            import mediapipe as mp

            if hasattr(mp, "solutions"):
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=max_num_faces,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                )
        except ImportError:
            self._mesh = None

    def extract(self, image_path: Path) -> list[LandmarkPoint] | None:
        if self._mesh is None:
            return None

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Unable to read frame: {image_path}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe can miss small faces on very large/portrait frames.
        # Run mesh on a downscaled copy to improve detector stability.
        frame_h, frame_w = frame.shape[:2]
        max_dim = max(frame_h, frame_w)
        if max_dim > 960:
            scale = 960.0 / float(max_dim)
            resized_rgb = cv2.resize(rgb, (max(int(frame_w * scale), 1), max(int(frame_h * scale), 1)))
            results = self._mesh.process(resized_rgb)
        else:
            results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            if self._face_detector is None:
                return None

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            if len(faces) == 0:
                return None

            x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.2)
            x0 = max(x - margin_x, 0)
            y0 = max(y - margin_y, 0)
            x1 = min(x + w + margin_x, frame_w)
            y1 = min(y + h + margin_y, frame_h)

            face_crop = rgb[y0:y1, x0:x1]
            if face_crop.size == 0:
                return None

            resized_crop = cv2.resize(face_crop, (256, 256))
            crop_results = self._mesh.process(resized_crop)
            if not crop_results.multi_face_landmarks:
                return None

            face_landmarks = crop_results.multi_face_landmarks[0].landmark
            return [
                LandmarkPoint(
                    (x0 + (point.x * (x1 - x0))) / max(frame_w, 1),
                    (y0 + (point.y * (y1 - y0))) / max(frame_h, 1),
                )
                for point in face_landmarks
            ]

        face_landmarks = results.multi_face_landmarks[0].landmark
        return [LandmarkPoint(point.x, point.y) for point in face_landmarks]