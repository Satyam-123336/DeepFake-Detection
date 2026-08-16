from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from facenet_pytorch import MTCNN


@dataclass(slots=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.6) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(keep_all=False, device=device)
        self.min_detection_confidence = min_detection_confidence

    def detect(self, image_path: Path) -> FaceBox | None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Unable to read frame: {image_path}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, probs = self.mtcnn.detect(rgb)
        
        if boxes is None or probs is None:
            return None

        best_box = None
        best_prob = -1.0
        
        for box, prob in zip(boxes, probs):
            if prob is None or prob < self.min_detection_confidence:
                continue
            if prob > best_prob:
                best_prob = float(prob)
                best_box = box

        if best_box is None:
            return None

        # box format is [xmin, ymin, xmax, ymax]
        x1, y1, x2, y2 = best_box
        x = max(int(x1), 0)
        y = max(int(y1), 0)
        w = max(int(x2 - x1), 0)
        h = max(int(y2 - y1), 0)

        h_frame, w_frame = frame.shape[:2]
        if w <= 0 or h <= 0 or x >= w_frame or y >= h_frame:
            return None

        return FaceBox(x=x, y=y, width=w, height=h, confidence=best_prob)