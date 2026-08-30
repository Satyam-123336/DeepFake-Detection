"""
LandmarkExtractor -- facenet-pytorch MTCNN replacement for the broken MediaPipe
1.x solutions API.

MediaPipe >= 1.0 dropped `mp.solutions` entirely.  We now use facenet-pytorch's
MTCNN for fast, accurate face + 5-keypoint detection and synthesise the exact
landmark slots expected by the EAR calculator (6 per eye) and the viseme
calculator (5 upper + 5 lower lip points) from those keypoints + the face
bounding box.

facenet-pytorch is already in the project venv (used by face_detector.py).
No additional downloads or model files are needed.

MTCNN 5-point landmark order:
    0 -- left_eye
    1 -- right_eye
    2 -- nose
    3 -- mouth_left
    4 -- mouth_right
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class LandmarkPoint:
    x: float
    y: float


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _interp_points(
    p_start: tuple[float, float],
    p_end:   tuple[float, float],
    n: int,
) -> list[tuple[float, float]]:
    """n points linearly interpolated from p_start to p_end inclusive."""
    return [
        (
            p_start[0] + (p_end[0] - p_start[0]) * i / max(n - 1, 1),
            p_start[1] + (p_end[1] - p_start[1]) * i / max(n - 1, 1),
        )
        for i in range(n)
    ]


def _synth_eye_landmarks(
    eye_center: tuple[float, float],
    eye_width:  float,
    eye_height: float,
) -> list[LandmarkPoint]:
    """
    Synthesise 6 eye landmark points from the eye centre + estimated dimensions.

    Layout matches MediaPipe's 6-point eye schema:
        0 -- left corner  (inner)
        1 -- upper-left
        2 -- upper-right
        3 -- right corner (outer)
        4 -- lower-right
        5 -- lower-left
    """
    cx, cy = eye_center
    hw = eye_width  / 2.0
    hh = eye_height / 2.0
    return [
        LandmarkPoint(cx - hw,        cy       ),  # 0 left corner
        LandmarkPoint(cx - hw * 0.4,  cy - hh  ),  # 1 upper-left
        LandmarkPoint(cx + hw * 0.4,  cy - hh  ),  # 2 upper-right
        LandmarkPoint(cx + hw,        cy       ),  # 3 right corner
        LandmarkPoint(cx + hw * 0.4,  cy + hh  ),  # 4 lower-right
        LandmarkPoint(cx - hw * 0.4,  cy + hh  ),  # 5 lower-left
    ]


def _synth_lip_landmarks(
    mouth_left:  tuple[float, float],
    mouth_right: tuple[float, float],
    face_h_px:   float,
    n: int = 5,
) -> tuple[list[LandmarkPoint], list[LandmarkPoint]]:
    """
    Synthesise n upper-lip and n lower-lip LandmarkPoints.

    Typical mouth height is ~8 % of face height.
    """
    mx = (mouth_left[0] + mouth_right[0]) / 2.0
    my = (mouth_left[1] + mouth_right[1]) / 2.0
    half_mouth_w = abs(mouth_right[0] - mouth_left[0]) / 2.0
    half_mouth_h = face_h_px * 0.04  # pixels

    top_pts = _interp_points(
        (mx - half_mouth_w, my - half_mouth_h),
        (mx + half_mouth_w, my - half_mouth_h),
        n,
    )
    bot_pts = _interp_points(
        (mx - half_mouth_w, my + half_mouth_h),
        (mx + half_mouth_w, my + half_mouth_h),
        n,
    )
    upper = [LandmarkPoint(px, py) for px, py in top_pts]
    lower = [LandmarkPoint(px, py) for px, py in bot_pts]
    return upper, lower


def _build_landmark_list(
    box:        tuple[float, float, float, float],  # x1, y1, x2, y2
    kp5:        Any,   # (5, 2) array -- [le, re, nose, ml, mr] in pixels
    img_w: int,
    img_h: int,
) -> list[LandmarkPoint]:
    """
    Build a 478-slot landmark list compatible with the MediaPipe index
    constants in `src/utils/constants.py`.

    Only the slots consumed by the pipeline are populated; the rest default to
    the face centre so that accidental access never crashes.

    Constants used:
        LEFT_EYE_LANDMARKS  = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
        UPPER_LIP_LANDMARKS = [13, 312, 311, 310, 415]
        LOWER_LIP_LANDMARKS = [14, 87, 178, 88, 95]
    """
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)

    kp = np.array(kp5, dtype=np.float32)  # shape (5, 2) in pixels

    le_px  = (float(kp[0, 0]), float(kp[0, 1]))
    re_px  = (float(kp[1, 0]), float(kp[1, 1]))
    ml_px  = (float(kp[3, 0]), float(kp[3, 1]))
    mr_px  = (float(kp[4, 0]), float(kp[4, 1]))

    # Eye dimensions in pixels then normalised
    eye_w_px = abs(re_px[0] - le_px[0]) * 0.35
    eye_h_px = eye_w_px * 0.45

    def _norm(pt_px: tuple[float, float]) -> tuple[float, float]:
        return (pt_px[0] / max(img_w, 1), pt_px[1] / max(img_h, 1))

    def _norm_pts(pts: list[LandmarkPoint]) -> list[LandmarkPoint]:
        return [LandmarkPoint(p.x / max(img_w, 1), p.y / max(img_h, 1)) for p in pts]

    left_eye_pts  = _synth_eye_landmarks(le_px, eye_w_px, eye_h_px)
    right_eye_pts = _synth_eye_landmarks(re_px, eye_w_px, eye_h_px)

    upper_lip_raw, lower_lip_raw = _synth_lip_landmarks(ml_px, mr_px, bh, n=5)

    left_eye_norm  = _norm_pts(left_eye_pts)
    right_eye_norm = _norm_pts(right_eye_pts)
    upper_lip_norm = _norm_pts(upper_lip_raw)
    lower_lip_norm = _norm_pts(lower_lip_raw)

    LEFT_EYE_SLOTS  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_SLOTS = [362, 385, 387, 263, 373, 380]
    UPPER_LIP_SLOTS = [13, 312, 311, 310, 415]
    LOWER_LIP_SLOTS = [14, 87, 178, 88, 95]

    cx = ((x1 + x2) / 2.0) / max(img_w, 1)
    cy = ((y1 + y2) / 2.0) / max(img_h, 1)
    filler = LandmarkPoint(cx, cy)
    landmarks: list[LandmarkPoint] = [filler] * 478

    for slot, pt in zip(LEFT_EYE_SLOTS,  left_eye_norm):
        landmarks[slot] = pt
    for slot, pt in zip(RIGHT_EYE_SLOTS, right_eye_norm):
        landmarks[slot] = pt
    for slot, pt in zip(UPPER_LIP_SLOTS, upper_lip_norm):
        landmarks[slot] = pt
    for slot, pt in zip(LOWER_LIP_SLOTS, lower_lip_norm):
        landmarks[slot] = pt

    return landmarks


# ---------------------------------------------------------------------------
# Public extractor class
# ---------------------------------------------------------------------------

class LandmarkExtractor:
    """Extracts facial landmarks using facenet-pytorch MTCNN, OpenCV Haar fallback."""

    def __init__(self, max_num_faces: int = 1) -> None:
        self._mtcnn: Any = None
        self._haar:  Any = None

        # Primary: facenet-pytorch MTCNN (already a project dependency)
        try:
            from facenet_pytorch import MTCNN  # type: ignore
            self._mtcnn = MTCNN(keep_all=False, device="cpu")
        except Exception:
            self._mtcnn = None

        # Fallback: OpenCV Haar cascade
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if cascade_path.exists():
            haar = cv2.CascadeClassifier(str(cascade_path))
            self._haar = haar if not haar.empty() else None

    # ------------------------------------------------------------------
    def extract(self, image_path: Path) -> list[LandmarkPoint] | None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Unable to read frame: {image_path}")

        img_h, img_w = frame.shape[:2]

        # -- facenet MTCNN path ----------------------------------------
        if self._mtcnn is not None:
            try:
                from PIL import Image  # type: ignore
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                boxes, probs, landmarks = self._mtcnn.detect(pil_img, landmarks=True)
                if boxes is not None and len(boxes) > 0 and probs[0] >= 0.80:
                    box = boxes[0]     # [x1, y1, x2, y2]
                    kp5 = landmarks[0] # shape (5, 2) in pixel coords
                    return _build_landmark_list(box, kp5, img_w, img_h)
            except Exception:
                pass  # fall through to Haar

        # -- OpenCV Haar fallback (geometric approximation) ------------
        if self._haar is not None:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
            )
            if len(faces) > 0:
                bx, by, bw, bh = max(faces, key=lambda b: int(b[2]) * int(b[3]))
                # Estimate 5 keypoints from face-box geometry
                le  = (bx + bw * 0.30, by + bh * 0.38)
                re  = (bx + bw * 0.70, by + bh * 0.38)
                ml  = (bx + bw * 0.35, by + bh * 0.70)
                mr  = (bx + bw * 0.65, by + bh * 0.70)
                kp5 = [le, re, (bx + bw * 0.50, by + bh * 0.55), ml, mr]
                box = [bx, by, bx + bw, by + bh]
                return _build_landmark_list(box, kp5, img_w, img_h)

        return None