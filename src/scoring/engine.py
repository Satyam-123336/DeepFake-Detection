from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FinalScoreResult:
    confidence_score: float
    risk_level: str
    module_scores: dict[str, float]
    reasons: list[str]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _blink_suspicion(duration_seconds: float, blink_count: int, irregularity: float) -> float:
    if duration_seconds <= 0:
        return 0.0

    # Short clips under 4s: not blinking is completely natural
    if duration_seconds < 4.0:
        return 0.05 if blink_count == 0 else 0.08

    rate_per_min = blink_count / (duration_seconds / 60.0)
    # Natural human blink rate is ~12-25 blinks/min.
    # Abnormally low (< 8 blinks/min) is a classic deepfake indicator.
    # Abnormally high (> 35 blinks/min) indicates rapid fluttering artifacts.
    too_low = _clamp01((8.0 - rate_per_min) / 8.0) if rate_per_min < 8.0 else 0.0
    too_high = _clamp01((rate_per_min - 35.0) / 20.0) if rate_per_min > 35.0 else 0.0
    uniformity = _clamp01((0.25 - irregularity) / 0.25) if irregularity < 0.25 else 0.0
    return _clamp01(0.55 * max(too_low, too_high) + 0.45 * uniformity)


def _lipsync_suspicion(offset_seconds: float, correlation: float) -> float:
    # Unavailable or non-speaking signal
    if (correlation <= 0.01 and abs(offset_seconds) < 1e-6) or correlation < 0.05:
        return 0.0

    # Dynamic mouth aperture cross-correlation with audio RMS:
    # Natural speech sync: offset <= 0.18s with correlation >= 0.25
    corr_deficit = max(0.0, (0.28 - correlation) / 0.28)
    delay_deficit = _clamp01((abs(offset_seconds) - 0.15) / 0.35) if abs(offset_seconds) > 0.15 else 0.0
    combined = 0.40 * corr_deficit + 0.60 * delay_deficit
    
    # Severe desync penalty applies when there is a clear audio-visual signal (correlation >= 0.10)
    # that is substantially delayed (>= 0.30s, well beyond ITU-R human perception limit of 125ms)
    if abs(offset_seconds) >= 0.30 and correlation >= 0.10:
        lag_severity = min(1.0, 0.70 + 0.30 * min((abs(offset_seconds) - 0.30) / 0.15, 1.0))
        return _clamp01(max(combined, lag_severity))
    return _clamp01(combined)


def _visual_suspicion(
    cnn_confidence: float | None,
    lighting_asymmetry: float | None,
    sharpness_score: float | None,
    texture_score: float | None,
) -> float:
    # If no face features are present, there is no visual manipulation evidence to score.
    if cnn_confidence is None and lighting_asymmetry is None and sharpness_score is None and texture_score is None:
        return 0.0

    cnn = _clamp01(cnn_confidence) if cnn_confidence is not None else None

    # Handcrafted artifact heuristics:
    light = _clamp01((lighting_asymmetry or 0.0) / 80.0)
    sharpness_val = sharpness_score if sharpness_score is not None else 80.0
    oversmooth = _clamp01((40.0 - sharpness_val) / 35.0) if sharpness_val < 40.0 else 0.0
    low_texture = _clamp01((35.0 - (texture_score if texture_score is not None else 35.0)) / 20.0)
    heuristic = _clamp01(0.50 * light + 0.40 * oversmooth + 0.10 * low_texture)

    if cnn is None:
        return heuristic

    # When CNN is confident the face is REAL (< 0.20):
    # CNN is the primary neural artifact detector; heuristics provide faint baseline evidence
    # without forcing a false-positive escalation on natural lighting.
    if cnn < 0.20:
        return max(cnn, heuristic * 0.60)

    # When CNN is intermediate (0.20 to 0.70):
    if cnn <= 0.70:
        return 0.65 * cnn + 0.35 * heuristic

    # CNN says FAKE (> 0.70):
    return max(cnn, heuristic)


def compute_final_score(
    *,
    duration_seconds: float,
    blink_count: int,
    blink_irregularity: float,
    lipsync_offset_seconds: float,
    lipsync_correlation: float,
    cnn_confidence: float | None,
    lighting_asymmetry: float | None,
    sharpness_score: float | None,
    texture_score: float | None,
    watermark_confidence: float,
    nlp_suspicion_score: float,
) -> FinalScoreResult:
    module_scores = {
        "blink": _blink_suspicion(duration_seconds, blink_count, blink_irregularity),
        "lipsync": _lipsync_suspicion(lipsync_offset_seconds, lipsync_correlation),
        "visual": _visual_suspicion(cnn_confidence, lighting_asymmetry, sharpness_score, texture_score),
        "watermark": _clamp01(watermark_confidence),
        "nlp": _clamp01(nlp_suspicion_score),
    }

    weights = {
        "blink": 0.2,
        "lipsync": 0.25,
        "visual": 0.35,
        "watermark": 0.1,
        "nlp": 0.1,
    }

    confidence = _clamp01(sum(module_scores[name] * weights[name] for name in module_scores))

    # Escalation rules:
    # 1. Strong neural visual artifact detection (CNN >= 0.70)
    strong_visual = module_scores["visual"] >= 0.70
    # 2. Strong lip-sync desync with genuine correlation
    strong_lipsync = module_scores["lipsync"] >= 0.70 and lipsync_correlation >= 0.10
    # 3. Corroborating signals
    corroboration_signals = [
        strong_visual,
        strong_lipsync,
        module_scores["watermark"] >= 0.20,
        module_scores["nlp"] >= 0.20,
        module_scores["blink"] >= 0.50 and blink_count > 0,
    ]
    corroboration_count = sum(1 for flag in corroboration_signals if flag)

    confidence_floor = 0.0

    # 4. Multi-signal behavioral desync + physical anomaly (catches faceswaps that evade simple CNNs)
    dubbed_faceswap = (
        module_scores["lipsync"] >= 0.55
        and lipsync_correlation >= 0.10
        and ((lighting_asymmetry or 0.0) >= 55.0 or module_scores["watermark"] >= 0.14)
    )

    if strong_visual and corroboration_count >= 2:
        confidence_floor = 0.90
    elif strong_visual:
        confidence_floor = 0.75
    elif strong_lipsync and (module_scores["visual"] >= 0.50 or corroboration_count >= 2 or module_scores["lipsync"] >= 0.85):
        confidence_floor = 0.75
    elif dubbed_faceswap:
        confidence_floor = 0.75

    final_confidence = max(confidence, confidence_floor)

    # Clean risk classification
    if final_confidence >= 0.65:
        risk = "high"
    elif final_confidence >= 0.38:
        risk = "medium"
    else:
        risk = "low"

    reason_map = {
        "blink": "Abnormal blink rhythm profile",
        "lipsync": "Audio-visual synchronization mismatch",
        "visual": "Visual artifact signals are elevated",
        "watermark": "Synthetic/watermark traces detected",
        "nlp": "Speech-language timing pattern is suspicious",
    }

    reasons = [reason_map[name] for name, score in module_scores.items() if score >= 0.55]
    if not reasons:
        reasons = ["No single module exceeded suspicious threshold"]

    return FinalScoreResult(
        confidence_score=final_confidence,
        risk_level=risk,
        module_scores=module_scores,
        reasons=reasons,
    )
