from __future__ import annotations

import math

from src.preprocessing.landmark_extractor import LandmarkPoint


def _distance(point_a: LandmarkPoint, point_b: LandmarkPoint) -> float:
    return math.dist((point_a.x, point_a.y), (point_b.x, point_b.y))


def calculate_eye_aspect_ratio(eye_points: list[LandmarkPoint]) -> float:
    if len(eye_points) != 6:
        raise ValueError("Eye aspect ratio requires exactly 6 eye landmarks.")

    vertical_a = _distance(eye_points[1], eye_points[5])
    vertical_b = _distance(eye_points[2], eye_points[4])
    horizontal = _distance(eye_points[0], eye_points[3])
    if horizontal == 0:
        return 0.0
    return (vertical_a + vertical_b) / (2.0 * horizontal)