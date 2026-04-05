from src.scoring.engine import compute_final_score


def test_scoring_engine_high_risk_for_multiple_strong_signals() -> None:
    result = compute_final_score(
        duration_seconds=12.0,
        blink_count=0,
        blink_irregularity=0.02,
        lipsync_offset_seconds=0.18,
        lipsync_correlation=0.1,
        cnn_confidence=0.92,
        lighting_asymmetry=0.35,
        sharpness_score=120.0,
        texture_score=25.0,
        watermark_confidence=0.8,
        nlp_suspicion_score=0.6,
    )
    assert result.risk_level == "high"
    assert result.confidence_score >= 0.65
    assert len(result.reasons) >= 1


def test_scoring_engine_low_risk_for_clean_signals() -> None:
    result = compute_final_score(
        duration_seconds=12.0,
        blink_count=4,
        blink_irregularity=0.4,
        lipsync_offset_seconds=0.01,
        lipsync_correlation=0.95,
        cnn_confidence=0.12,
        lighting_asymmetry=0.05,
        sharpness_score=1200.0,
        texture_score=70.0,
        watermark_confidence=0.0,
        nlp_suspicion_score=0.0,
    )
    assert result.risk_level in {"low", "medium"}
    assert result.confidence_score < 0.5


def test_scoring_engine_does_not_over_penalize_missing_signals() -> None:
    result = compute_final_score(
        duration_seconds=12.0,
        blink_count=0,
        blink_irregularity=1.0,
        lipsync_offset_seconds=0.0,
        lipsync_correlation=0.0,
        cnn_confidence=None,
        lighting_asymmetry=22.0,
        sharpness_score=800.0,
        texture_score=65.0,
        watermark_confidence=0.05,
        nlp_suspicion_score=0.0,
    )
    assert result.risk_level == "low"
    assert result.confidence_score < 0.45


def test_scoring_engine_uses_cnn_fake_probability_signal() -> None:
    low_fake_prob = compute_final_score(
        duration_seconds=12.0,
        blink_count=4,
        blink_irregularity=0.4,
        lipsync_offset_seconds=0.01,
        lipsync_correlation=0.95,
        cnn_confidence=0.05,
        lighting_asymmetry=10.0,
        sharpness_score=900.0,
        texture_score=68.0,
        watermark_confidence=0.0,
        nlp_suspicion_score=0.0,
    )
    high_fake_prob = compute_final_score(
        duration_seconds=12.0,
        blink_count=4,
        blink_irregularity=0.4,
        lipsync_offset_seconds=0.01,
        lipsync_correlation=0.95,
        cnn_confidence=0.95,
        lighting_asymmetry=10.0,
        sharpness_score=900.0,
        texture_score=68.0,
        watermark_confidence=0.0,
        nlp_suspicion_score=0.0,
    )
    assert high_fake_prob.confidence_score > low_fake_prob.confidence_score


def test_scoring_engine_does_not_force_medium_for_strong_visual_without_corroboration() -> None:
    result = compute_final_score(
        duration_seconds=12.0,
        blink_count=0,
        blink_irregularity=1.0,
        lipsync_offset_seconds=0.0,
        lipsync_correlation=0.0,
        cnn_confidence=0.931,
        lighting_asymmetry=42.96,
        sharpness_score=139.0,
        texture_score=58.4,
        watermark_confidence=0.06,
        nlp_suspicion_score=0.0,
    )
    assert result.risk_level == "low"
    assert result.confidence_score < 0.5


def test_scoring_engine_escalates_strong_visual_and_behavioral_to_high() -> None:
    result = compute_final_score(
        duration_seconds=12.0,
        blink_count=0,
        blink_irregularity=0.02,
        lipsync_offset_seconds=0.14,
        lipsync_correlation=0.15,
        cnn_confidence=0.9,
        lighting_asymmetry=38.0,
        sharpness_score=130.0,
        texture_score=40.0,
        watermark_confidence=0.0,
        nlp_suspicion_score=0.0,
    )
    assert result.risk_level == "high"


def test_scoring_engine_user_reported_ai_like_case_can_remain_low_without_corroboration() -> None:
    result = compute_final_score(
        duration_seconds=12.65,
        blink_count=0,
        blink_irregularity=1.0,
        lipsync_offset_seconds=0.0,
        lipsync_correlation=0.0,
        cnn_confidence=0.0,
        lighting_asymmetry=53.74,
        sharpness_score=900.0,
        texture_score=75.0,
        watermark_confidence=0.0,
        nlp_suspicion_score=0.0,
    )
    assert result.risk_level == "low"
    assert result.confidence_score < 0.45


def test_scoring_engine_heuristics_raise_visual_signal_without_forced_medium() -> None:
    result = compute_final_score(
        duration_seconds=12.65,
        blink_count=0,
        blink_irregularity=1.0,
        lipsync_offset_seconds=0.0,
        lipsync_correlation=0.0,
        cnn_confidence=0.0,
        lighting_asymmetry=53.74,
        sharpness_score=139.09,
        texture_score=58.44,
        watermark_confidence=0.0,
        nlp_suspicion_score=0.0,
    )
    assert result.risk_level == "low"
    assert result.module_scores["visual"] >= 0.55
    assert result.confidence_score < 0.45


def test_scoring_engine_does_not_escalate_isolated_visual_spike_without_corroboration() -> None:
    result = compute_final_score(
        duration_seconds=15.84,
        blink_count=0,
        blink_irregularity=1.0,
        lipsync_offset_seconds=0.0,
        lipsync_correlation=0.0,
        cnn_confidence=0.96,
        lighting_asymmetry=19.0,
        sharpness_score=7.0,
        texture_score=18.0,
        watermark_confidence=0.05,
        nlp_suspicion_score=0.0,
    )
    assert result.risk_level == "low"
