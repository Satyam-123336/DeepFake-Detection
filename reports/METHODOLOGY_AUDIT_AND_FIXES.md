# Methodology Audit And Fixes

Date: 2026-03-15

## Summary

The previous behavior mismatch came from scoring/inference implementation issues, not only UI wording.
Core decision logic has now been corrected and validated with tests.

## Critical Issues Found

1. CNN score semantic bug
- Previous behavior: scoring used `max softmax confidence` as fake suspicion.
- Problem: high confidence for class 0 (real) was still treated as suspicious.
- Fix: scoring now uses explicit `fake_probability` from class-1 output.

2. CNN weights path mismatch
- Previous behavior: full pipeline looked for `lightweight_artifact_cnn.pt` only.
- Problem: available file was `demo_lightweight_artifact_cnn.pt`, so CNN signal was often unavailable.
- Fix: weight resolver now checks both canonical and demo paths.

3. Lip-sync timestamp misalignment
- Previous behavior: audio was sampled using first N frame timestamps, while mouth signal used only frames where landmarks existed.
- Problem: timeline mismatch degraded correlation quality.
- Fix: audio is now sampled at exact mouth-signal timestamps.

4. Overly sensitive fallback scoring
- Previous behavior: unavailable/missing evidence could strongly increase suspicion.
- Fix: missing-signal patterns are now treated as weak evidence; thresholds tuned.

## Methodology Alignment Status

Phase 1-4: operational
Phase 5: in progress (watermark + NLP proxy)
Phase 6: operational initial version (scoring + Streamlit)
Phase 7: in progress (expanded tests and calibration path)

## Tests Added/Updated

- Scoring regression for missing signals (prevents easy false fake outcomes)
- Scoring test for fake-probability semantics
- Pipeline schema test for visual fake probability field

Current status: test suite passing.

## What To Verify Next (Recommended)

1. Batch evaluation
- Run a labeled batch (real and fake) and compute confusion summary:
  - true positive rate
  - false positive rate
  - false negative rate

2. Threshold calibration
- Adjust risk thresholds using validation set metrics, not heuristic-only rules.

3. Production STT and NLP
- Replace energy-proxy transcript with Whisper or equivalent.
- Add transcript-content NLP classifier for Phase 5 completeness.

4. UI clarity checks
- Verify user interpretation with non-technical users.
- Keep technical JSON optional only.
