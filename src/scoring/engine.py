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
        return 0.25

    # No detected blink windows with default irregularity usually means insufficient signal,
    # not necessarily suspicious behavior.
    if blink_count == 0 and irregularity >= 0.95:
        return 0.25

    rate_per_min = blink_count / (duration_seconds / 60.0)
    too_low = _clamp01((6.0 - rate_per_min) / 6.0) if rate_per_min < 6.0 else 0.0
    too_high = _clamp01((rate_per_min - 35.0) / 20.0) if rate_per_min > 35.0 else 0.0
    uniformity = _clamp01((0.25 - irregularity) / 0.25) if irregularity < 0.25 else 0.0
    return _clamp01(0.45 * max(too_low, too_high) + 0.55 * uniformity)


def _lipsync_suspicion(offset_seconds: float, correlation: float) -> float:
    # This pattern is emitted by behavioral stage when lip-sync could not be estimated reliably.
    if correlation <= 0.01 and abs(offset_seconds) < 1e-6:
        return 0.2

    corr_term = _clamp01(1.0 - correlation)
    delay_term = _clamp01(abs(offset_seconds) / 0.2)
    return _clamp01(0.7 * corr_term + 0.3 * delay_term)


def _visual_suspicion(
    cnn_confidence: float | None,
    lighting_asymmetry: float | None,
    sharpness_score: float | None,
    texture_score: float | None,
) -> float:
    cnn = _clamp01(cnn_confidence) if cnn_confidence is not None else 0.0

    # Handcrafted artifact heuristics aligned to the methodology:
    # low sharpness => oversmoothed skin, high lighting asymmetry => inconsistent illumination.
    light = _clamp01((lighting_asymmetry or 0.0) / 80.0)
    oversmooth = _clamp01((400.0 - (sharpness_score or 0.0)) / 400.0)
    low_texture = _clamp01((55.0 - (texture_score or 55.0)) / 20.0)
    heuristic = _clamp01(0.45 * light + 0.45 * oversmooth + 0.10 * low_texture)

    # If the CNN is confident the sample is fake, trust it strongly.
    # If the CNN predicts real, still preserve visual heuristics instead of suppressing them.
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

    # Escalation rules: a strong primary signal should not be averaged down to low risk.
    strong_visual = module_scores["visual"] >= 0.75
    corroboration_signals = [
        (lighting_asymmetry or 0.0) >= 40.0,
        module_scores["watermark"] >= 0.2,
        module_scores["nlp"] >= 0.25,
        (lipsync_correlation > 0.01 and module_scores["lipsync"] >= 0.55),
        (blink_count > 0 and module_scores["blink"] >= 0.55),
    ]
    strong_visual_corroborated = strong_visual and sum(1 for flag in corroboration_signals if flag) >= 2
    strong_behavioral = max(module_scores["blink"], module_scores["lipsync"]) >= 0.75
    corroborated_visual = module_scores["visual"] >= 0.65 and max(
        module_scores["blink"], module_scores["lipsync"], module_scores["watermark"], module_scores["nlp"]
    ) >= 0.3
    unresolved_lipsync = lipsync_correlation <= 0.01 and abs(lipsync_offset_seconds) < 1e-6
    suspicious_combo = (
        (lighting_asymmetry or 0.0) >= 55.0
        and module_scores["visual"] >= 0.8
        and blink_count == 0
        and unresolved_lipsync
        and (module_scores["watermark"] >= 0.15 or module_scores["nlp"] >= 0.2)
    )

    confidence_floor = 0.0
    if strong_visual and strong_behavioral:
        confidence_floor = 0.72
    elif strong_visual_corroborated:
        confidence_floor = 0.5
    elif corroborated_visual:
        confidence_floor = 0.5
    elif suspicious_combo:
        confidence_floor = 0.5

    confidence = max(confidence, confidence_floor)

    if confidence >= 0.7:
        risk = "high"
    elif confidence >= 0.45:
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
        confidence_score=confidence,
        risk_level=risk,
        module_scores=module_scores,
        reasons=reasons,
    )
