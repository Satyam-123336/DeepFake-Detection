"""Tests for VideoCacheManager and deduplication."""

from pathlib import Path
import tempfile
import pytest

from src.utils.video_cache import VideoCacheManager


def test_video_cache_store_and_deduplicate():
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = VideoCacheManager(cache_dir=tmp_dir, max_files=10, max_size_mb=100)

        # Create dummy video bytes
        dummy_data = b"RIFF" + b"\x00" * 1024 * 50  # 50KB dummy payload

        # First store: new file
        path1, hash1, is_dup1 = manager.store(dummy_data, "sample.mp4")
        assert not is_dup1
        assert path1.exists()
        assert hash1 == manager.compute_hash(dummy_data)

        # Second store: exact same content with different name
        path2, hash2, is_dup2 = manager.store(dummy_data, "sample_copy.mp4")
        assert is_dup2
        assert hash1 == hash2
        assert path1 == path2  # Must point to same canonical file

        # Only one video file exists on disk
        stored_files = [f for f in Path(tmp_dir).iterdir() if f.is_file() and f.suffix == ".mp4"]
        assert len(stored_files) == 1

        stats = manager.get_stats()
        assert stats["total_stored_videos"] == 1
        assert stats["duplicate_detections"] == 1


def test_video_cache_prune_limits():
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = VideoCacheManager(cache_dir=tmp_dir, max_files=2, max_size_mb=10)

        # Store 3 unique files
        manager.store(b"video_1_" + b"\x01" * 1024, "v1.mp4")
        manager.store(b"video_2_" + b"\x02" * 1024, "v2.mp4")
        manager.store(b"video_3_" + b"\x03" * 1024, "v3.mp4")

        # Must prune down to max_files = 2
        stats = manager.get_stats()
        assert stats["total_stored_videos"] == 2
        stored_mp4s = [f for f in Path(tmp_dir).iterdir() if f.is_file() and f.suffix == ".mp4"]
        assert len(stored_mp4s) == 2
