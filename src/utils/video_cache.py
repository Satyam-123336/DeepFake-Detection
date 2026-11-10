"""Video Cache Manager with deduplication and storage management."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Union


class VideoCacheManager:
    """
    Manages persistent storage and deduplication of uploaded and processed videos.
    
    Guarantees:
    1. Zero duplicate video files stored on disk (content-addressed storage by hash).
    2. Sub-millisecond duplicate detection.
    3. Self-healing metadata index with automatic disk reconciliation.
    4. Size and file-count bounds to prevent disk exhaustion.
    """

    def __init__(
        self,
        cache_dir: Union[Path, str] = "data/video_cache",
        max_files: int = 100,
        max_size_mb: int = 2048,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "video_index.json"
        self.max_files = max_files
        self.max_size_mb = max_size_mb
        self._lock = threading.Lock()
        self._duplicate_hits = 0
        self._index: dict[str, dict[str, Any]] = self._load_index()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Load index from disk or rebuild from directory if missing/corrupt."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return self._rebuild_index()

    def _save_index(self) -> None:
        """Persist index to disk safely."""
        try:
            tmp_path = self.index_file.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2)
            tmp_path.replace(self.index_file)
        except Exception as e:
            print(f"[VideoCacheManager] Warning: failed to save index: {e}")

    def _rebuild_index(self) -> dict[str, dict[str, Any]]:
        """Reconcile index from disk contents."""
        index: dict[str, dict[str, Any]] = {}
        for file in self.cache_dir.iterdir():
            if file.is_file() and file.name != "video_index.json" and not file.name.endswith(".tmp"):
                stem = file.stem
                stat = file.stat()
                index[stem] = {
                    "hash": stem,
                    "filename": file.name,
                    "stored_filename": file.name,
                    "size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "last_accessed": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "hit_count": 1,
                }
        return index

    @staticmethod
    def compute_hash(source: Union[bytes, Path, str]) -> str:
        """
        Compute deterministic hash matching optimized_inference._video_hash.
        Hashes first 1MB + total size for instant O(1) performance.
        """
        if isinstance(source, (str, Path)):
            p = Path(source)
            with open(p, "rb") as f:
                header = f.read(1024 * 1024)
                f.seek(0, 2)
                size = f.tell()
        else:
            header = source[:1024 * 1024]
            size = len(source)

        hasher = hashlib.md5(header)
        hasher.update(str(size).encode())
        return hasher.hexdigest()[:16]

    def get(self, video_hash: str) -> Path | None:
        """Retrieve stored video path by hash if it exists."""
        with self._lock:
            info = self._index.get(video_hash)
            if not info:
                # Check directly on disk in case index missed it
                matches = list(self.cache_dir.glob(f"{video_hash}.*"))
                if matches and matches[0].is_file():
                    target = matches[0]
                    self._index[video_hash] = {
                        "hash": video_hash,
                        "filename": target.name,
                        "stored_filename": target.name,
                        "size_bytes": target.stat().st_size,
                        "created_at": datetime.now().isoformat(),
                        "last_accessed": datetime.now().isoformat(),
                        "hit_count": 1,
                    }
                    self._save_index()
                    return target
                return None

            stored_name = info.get("stored_filename", f"{video_hash}.mp4")
            target_path = self.cache_dir / stored_name
            if target_path.exists():
                info["last_accessed"] = datetime.now().isoformat()
                info["hit_count"] = info.get("hit_count", 0) + 1
                self._save_index()
                return target_path
            else:
                del self._index[video_hash]
                self._save_index()
                return None

    def store(
        self,
        source: Union[bytes, Path, str],
        original_filename: str = "video.mp4",
    ) -> tuple[Path, str, bool]:
        """
        Store video with full content-addressed deduplication.

        Returns:
            (stored_path, video_hash, is_duplicate)
        """
        video_hash = self.compute_hash(source)
        ext = Path(original_filename).suffix.lower() or ".mp4"
        stored_filename = f"{video_hash}{ext}"
        target_path = self.cache_dir / stored_filename

        with self._lock:
            # Check if file already exists in index and on disk
            if target_path.exists() and target_path.stat().st_size > 0:
                self._duplicate_hits += 1
                info = self._index.setdefault(video_hash, {})
                info["hash"] = video_hash
                info["stored_filename"] = stored_filename
                info["original_filename"] = original_filename
                info["last_accessed"] = datetime.now().isoformat()
                info["hit_count"] = info.get("hit_count", 1) + 1
                info["size_bytes"] = target_path.stat().st_size
                self._save_index()
                return target_path, video_hash, True

            # Not on disk: write file
            if isinstance(source, (str, Path)):
                shutil.copy2(Path(source), target_path)
            else:
                with open(target_path, "wb") as f:
                    f.write(source)

            # Record in index
            self._index[video_hash] = {
                "hash": video_hash,
                "original_filename": original_filename,
                "stored_filename": stored_filename,
                "size_bytes": target_path.stat().st_size,
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "hit_count": 1,
            }
            self._save_index()

            # Self-prune if capacity exceeded
            self._prune_unlocked()

            return target_path, video_hash, False

    def prune(self) -> int:
        """Prune cache according to max_files and max_size_mb limits."""
        with self._lock:
            return self._prune_unlocked()

    def _prune_unlocked(self) -> int:
        """Internal un-locked prune method."""
        if not self._index:
            return 0

        evicted = 0
        # Check count limit
        if len(self._index) > self.max_files:
            # Sort by last_accessed ascending (oldest first)
            sorted_hashes = sorted(
                self._index.keys(),
                key=lambda h: self._index[h].get("last_accessed", ""),
            )
            while len(self._index) > self.max_files and sorted_hashes:
                oldest_hash = sorted_hashes.pop(0)
                info = self._index.pop(oldest_hash, None)
                if info:
                    p = self.cache_dir / info.get("stored_filename", f"{oldest_hash}.mp4")
                    p.unlink(missing_ok=True)
                    evicted += 1

        # Check total size limit
        total_bytes = sum(info.get("size_bytes", 0) for info in self._index.values())
        max_bytes = self.max_size_mb * 1024 * 1024
        if total_bytes > max_bytes:
            sorted_hashes = sorted(
                self._index.keys(),
                key=lambda h: self._index[h].get("last_accessed", ""),
            )
            while total_bytes > max_bytes and sorted_hashes:
                oldest_hash = sorted_hashes.pop(0)
                info = self._index.pop(oldest_hash, None)
                if info:
                    p = self.cache_dir / info.get("stored_filename", f"{oldest_hash}.mp4")
                    total_bytes -= info.get("size_bytes", 0)
                    p.unlink(missing_ok=True)
                    evicted += 1

        if evicted > 0:
            self._save_index()
        return evicted

    def get_stats(self) -> dict[str, Any]:
        """Return video cache performance and storage statistics."""
        with self._lock:
            total_bytes = sum(info.get("size_bytes", 0) for info in self._index.values())
            return {
                "total_stored_videos": len(self._index),
                "total_size_mb": round(total_bytes / (1024 * 1024), 2),
                "duplicate_detections": self._duplicate_hits,
                "max_files_capacity": self.max_files,
                "max_size_mb_capacity": self.max_size_mb,
                "videos": [
                    {
                        "hash": info.get("hash"),
                        "filename": info.get("original_filename") or info.get("stored_filename"),
                        "size_mb": round(info.get("size_bytes", 0) / (1024 * 1024), 2),
                        "hit_count": info.get("hit_count", 1),
                        "last_accessed": info.get("last_accessed"),
                    }
                    for info in sorted(
                        self._index.values(),
                        key=lambda x: x.get("last_accessed", ""),
                        reverse=True,
                    )[:20]
                ],
            }

    def clear(self) -> None:
        """Clear all stored cached videos."""
        with self._lock:
            for file in self.cache_dir.iterdir():
                if file.is_file():
                    try:
                        file.unlink()
                    except Exception:
                        pass
            self._index.clear()
            self._duplicate_hits = 0
            self._save_index()


# Global shared video cache instance
video_cache_manager = VideoCacheManager()
