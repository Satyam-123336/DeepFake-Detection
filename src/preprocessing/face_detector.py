from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(slots=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.6) -> None:
        self._face_detection = None
        self._haar_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        try:
            import mediapipe as mp

            if hasattr(mp, "solutions"):
                self._face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=min_detection_confidence,
                )
        except ImportError:
            self._face_detection = None

    def detect(self, image_path: Path) -> FaceBox | None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Unable to read frame: {image_path}")

        if self._face_detection is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_detection.process(rgb)
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                return FaceBox(
                    x=max(int(bbox.xmin * w), 0),
                    y=max(int(bbox.ymin * h), 0),
                    width=int(bbox.width * w),
                    height=int(bbox.height * h),
                    confidence=float(detection.score[0]),
                )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None

        x, y, width, height = faces[0]
        return FaceBox(x=int(x), y=int(y), width=int(width), height=int(height), confidence=0.5)