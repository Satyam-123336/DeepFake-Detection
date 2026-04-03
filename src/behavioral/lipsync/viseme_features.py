from __future__ import annotations

from src.preprocessing.landmark_extractor import LandmarkPoint


def compute_mouth_openness(upper_lip: list[LandmarkPoint], lower_lip: list[LandmarkPoint]) -> float:
    if not upper_lip or not lower_lip or len(upper_lip) != len(lower_lip):
        return 0.0

    distances = [abs(lower.y - upper.y) for upper, lower in zip(upper_lip, lower_lip)]
    return sum(distances) / len(distances)