from __future__ import annotations

import importlib
from pathlib import Path

from src.utils.io import ensure_dir


def _load_video_file_clip():
    try:
        return importlib.import_module("moviepy").VideoFileClip
    except (ImportError, AttributeError):
        return importlib.import_module("moviepy.editor").VideoFileClip


def extract_audio(video_path: Path, output_dir: Path) -> Path | None:
    ensure_dir(output_dir)
    output_path = output_dir / f"{video_path.stem}.wav"
    video_file_clip = _load_video_file_clip()

    with video_file_clip(str(video_path)) as clip:
        if clip.audio is None:
            return None
        clip.audio.write_audiofile(str(output_path), fps=16000, logger=None)

    return output_path