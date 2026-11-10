"""Optimized inference wrapper with caching and memoization."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.pipeline.run_full_pipeline import run_phase_four_pipeline
from src.utils.cache_manager import cached, PersistentCache
from src.utils.video_cache import video_cache_manager

# BUG 7 FIX: Increment this whenever the pipeline logic changes to bust stale
# cached results stored on disk (24h TTL cache bypassed otherwise).
PIPELINE_VERSION = "9"


class OptimizedInferencePipeline:
    """Wraps the full pipeline with intelligent caching and optimization."""

    def __init__(self, cache_dir: Path | str = "data/.cache"):
        """Initialize with persistent and runtime caching."""
        self.persistent_cache = PersistentCache(Path(cache_dir))
        self._inference_count = 0
        self._cache_hits = 0

    def _video_hash(self, video_path: Path) -> str:
        """Generate deterministic hash of video content for caching."""
        with open(video_path, "rb") as f:
            # Only hash first 1MB + size for speed, not entire file
            header = f.read(1024 * 1024)
            f.seek(0, 2)
            size = f.tell()
        hasher = hashlib.md5(header)
        hasher.update(str(size).encode())
        return hasher.hexdigest()[:16]

    def analyze_video(self, video_path: Path, processed_dir: Path, force_refresh: bool = False) -> dict[str, Any]:
        """
        Run full pipeline with intelligent caching and deduplication.

        Args:
            video_path: Path to input video
            processed_dir: Directory for intermediate artifacts
            force_refresh: Skip cache and reprocess

        Returns:
            Analysis result dictionary
        """
        self._inference_count += 1
        video_path = Path(video_path)
        processed_dir = Path(processed_dir)

        # Store / index in video cache for deduplication
        stored_path, video_hash, is_dup = video_cache_manager.store(video_path, video_path.name)
        # Use stored canonical path if caller passed a transient upload path
        active_video_path = stored_path if stored_path.exists() else video_path

        # Generate cache key based on video content
        # BUG 7 FIX: Include pipeline version in cache key so cached results from
        # older code are automatically invalidated after logic changes.
        cache_key = f"inference_{video_hash}_v{PIPELINE_VERSION}"

        # Check persistent cache first
        if not force_refresh:
            cached_result = self.persistent_cache.get(cache_key)
            if cached_result is not None:
                self._cache_hits += 1
                result_copy = dict(cached_result)
                result_copy["from_cache"] = True
                result_copy["video_hash"] = video_hash
                result_copy["is_duplicate"] = is_dup
                return result_copy

        # Run pipeline if not cached
        result = run_phase_four_pipeline(active_video_path, processed_dir)
        payload = result.to_dict()
        payload["from_cache"] = False
        payload["video_hash"] = video_hash
        payload["is_duplicate"] = is_dup

        # Store in persistent cache (24-hour TTL for full inference results)
        self.persistent_cache.set(cache_key, payload, ttl_seconds=86400)

        return payload

    def get_stats(self) -> dict[str, Any]:
        """Return timing and cache statistics."""
        hit_rate = self._cache_hits / max(1, self._inference_count)
        return {
            "total_inferences": self._inference_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": round(hit_rate, 3),
            "inferences_from_disk": self._cache_hits,
            "inferences_recomputed": self._inference_count - self._cache_hits,
        }


# Global optimized pipeline instance
_optimized_pipeline = OptimizedInferencePipeline()


def analyze_video_optimized(
    video_path: Path,
    processed_dir: Path = Path("data/processed"),
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Convenience function for optimized analysis."""
    return _optimized_pipeline.analyze_video(video_path, processed_dir, force_refresh=force_refresh)


def get_optimization_stats() -> dict[str, Any]:
    """Get current optimization statistics."""
    return _optimized_pipeline.get_stats()
