from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(slots=True)
class TranscriptResult:
    transcript_text: str
    method: str
    confidence: float
    voiced_ratio: float
    speech_segments: int
    segment_durations: list[float]
    duration_seconds: float


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _transcribe_with_whisper(audio_path: Path) -> TranscriptResult | None:
    try:
        import whisper
    except ImportError:
        return None

    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(str(audio_path), fp16=False, language="en", verbose=False)
    except Exception:
        return None

    text = (result.get("text") or "").strip()
    segments = result.get("segments") or []

    if not segments:
        return TranscriptResult(
            transcript_text=text,
            method="whisper-stt",
            confidence=0.55 if text else 0.0,
            voiced_ratio=0.0,
            speech_segments=0,
            segment_durations=[],
            duration_seconds=0.0,
        )

    segment_durations = [max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))) for seg in segments]
    duration_seconds = max(float(segments[-1].get("end", 0.0)), 0.0)
    voiced_seconds = float(sum(segment_durations))
    voiced_ratio = (voiced_seconds / duration_seconds) if duration_seconds > 0 else 0.0

    log_probs = [float(seg.get("avg_logprob", -1.2)) for seg in segments if "avg_logprob" in seg]
    if log_probs:
        confidence = _clamp01(float(np.exp(np.mean(log_probs))))
    else:
        confidence = 0.6 if text else 0.0

    return TranscriptResult(
        transcript_text=text,
        method="whisper-stt",
        confidence=confidence,
        voiced_ratio=_clamp01(voiced_ratio),
        speech_segments=len(segments),
        segment_durations=segment_durations,
        duration_seconds=duration_seconds,
    )


def _transcribe_with_energy_proxy(audio_path: Path) -> TranscriptResult:
    try:
        waveform, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    except Exception:
        waveform = np.array([])
        sr = 16000

    if waveform.size == 0:
        return TranscriptResult(
            transcript_text="",
            method="energy-proxy",
            confidence=0.0,
            voiced_ratio=0.0,
            speech_segments=0,
            segment_durations=[],
            duration_seconds=0.0,
        )

    duration_seconds = float(waveform.size / sr)
    frame_length = 512
    hop_length = 256
    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]

    threshold = float(np.percentile(rms, 55))
    active = rms > threshold
    voiced_ratio = float(np.mean(active)) if active.size else 0.0

    segment_durations: list[float] = []
    if active.size:
        start = None
        for idx, is_active in enumerate(active):
            if is_active and start is None:
                start = idx
            elif not is_active and start is not None:
                frames = idx - start
                segment_durations.append((frames * hop_length) / sr)
                start = None
        if start is not None:
            frames = active.size - start
            segment_durations.append((frames * hop_length) / sr)

    speech_segments = len(segment_durations)
    confidence = _clamp01(0.2 + 0.7 * voiced_ratio)

    # Acoustic forensic analysis using librosa:
    try:
        # Pitch jitter tracking
        f0 = librosa.yin(waveform, fmin=60, fmax=400, sr=sr)
        f0_voiced = f0[f0 > 65]
        pitch_jitter = float(np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-6)) if len(f0_voiced) > 10 else 0.30
    except Exception:
        pitch_jitter = 0.30

    try:
        # Spectral Flatness (natural speech is harmonic < 0.08, vocoders/TTS have high noise > 0.12)
        flatness_arr = librosa.feature.spectral_flatness(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
        mean_flatness = float(np.mean(flatness_arr)) if flatness_arr.size else 0.035
    except Exception:
        mean_flatness = 0.035

    try:
        # Spectral Centroid
        centroid_arr = librosa.feature.spectral_centroid(y=waveform, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
        mean_centroid = float(np.mean(centroid_arr)) if centroid_arr.size else 1700.0
    except Exception:
        mean_centroid = 1700.0

    transcript_text = (
        f"[acoustic_proxy] voiced_ratio={voiced_ratio:.3f}; "
        f"segments={speech_segments}; duration={duration_seconds:.2f}s; "
        f"pitch_jitter={pitch_jitter:.3f}; flatness={mean_flatness:.4f}; "
        f"centroid={mean_centroid:.1f}Hz"
    )

    return TranscriptResult(
        transcript_text=transcript_text,
        method="energy-proxy",
        confidence=confidence,
        voiced_ratio=voiced_ratio,
        speech_segments=speech_segments,
        segment_durations=segment_durations,
        duration_seconds=duration_seconds,
    )


def transcribe_audio_proxy(audio_path: Path | None) -> TranscriptResult:
    if audio_path is None or not audio_path.exists():
        return TranscriptResult(
            transcript_text="",
            method="unavailable",
            confidence=0.0,
            voiced_ratio=0.0,
            speech_segments=0,
            segment_durations=[],
            duration_seconds=0.0,
        )

    whisper_result = _transcribe_with_whisper(audio_path)
    if whisper_result is not None:
        return whisper_result

    return _transcribe_with_energy_proxy(audio_path)
