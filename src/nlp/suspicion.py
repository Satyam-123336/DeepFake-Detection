from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from src.nlp.transcription import TranscriptResult


@dataclass(slots=True)
class NLPSuspicionResult:
    score: float
    reasons: list[str]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_nlp_suspicion(transcript: TranscriptResult) -> NLPSuspicionResult:
    if transcript.method == "unavailable":
        return NLPSuspicionResult(0.0, ["No transcript/audio available"])

    score = 0.0
    reasons: list[str] = []

    if transcript.duration_seconds >= 8.0 and transcript.speech_segments < 2:
        score += 0.25
        reasons.append("Very low speaking-event count for clip duration")

    if transcript.voiced_ratio > 0.92:
        score += 0.2
        reasons.append("Near-continuous voiced activity")

    if transcript.segment_durations:
        mean_dur = float(np.mean(transcript.segment_durations))
        std_dur = float(np.std(transcript.segment_durations))
        cv = std_dur / mean_dur if mean_dur > 0 else 0.0
        if transcript.speech_segments >= 4 and cv < 0.2:
            score += 0.3
            reasons.append("Speech segment durations are unusually uniform")

    text = (transcript.transcript_text or "").strip()
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if words and transcript.method != "energy-proxy":
        unique_ratio = len(set(words)) / len(words)
        if len(words) >= 30 and unique_ratio < 0.38:
            score += 0.15
            reasons.append("Lexical diversity is unusually low")

        repeats = sum(1 for idx in range(1, len(words)) if words[idx] == words[idx - 1])
        if len(words) >= 20 and (repeats / len(words)) > 0.05:
            score += 0.15
            reasons.append("Consecutive word repetition is elevated")

        punctuation_count = len(re.findall(r"[.,;:!?]", text))
        if len(words) >= 80 and (punctuation_count / len(words)) < 0.01:
            score += 0.1
            reasons.append("Transcript punctuation cadence is unusually flat")

    return NLPSuspicionResult(_clamp01(score), reasons)
