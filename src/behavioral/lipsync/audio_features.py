from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def extract_audio_envelope(audio_path: Path, sample_rate: int = 16000, hop_length: int = 256) -> tuple[np.ndarray, np.ndarray]:
    waveform, sr = librosa.load(audio_path, sr=sample_rate)
    rms = librosa.feature.rms(y=waveform, hop_length=hop_length)[0]
    timestamps = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return timestamps, rms