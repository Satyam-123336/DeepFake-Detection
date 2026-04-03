"""Cache manager for runtime optimization with TTL and memory-efficient storage."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Container for cached data with timestamp and metadata."""

    value: T
    timestamp: float
    ttl_seconds: float
    hit_count: int = 0
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.timestamp > self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for inspection."""
        return {
            "timestamp": self.timestamp,
            "ttl_seconds": self.ttl_seconds,
            "hit_count": self.hit_count,
            "size_bytes": self.size_bytes,
            "age_seconds": time.time() - self.timestamp,
        }


class RuntimeCache:
    """In-memory cache with TTL and automatic expiration."""

    def __init__(self, max_size_mb: int = 256):
        """Initialize cache with size limit."""
        self._cache: dict[str, CacheEntry[Any]] = {}
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._current_size_bytes = 0
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _make_key(self, name: str, *args, **kwargs) -> str:
        """Generate deterministic cache key from function name and arguments."""
        key_data = json.dumps({"name": name, "args": str(args), "kwargs": str(kwargs)}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()

    def set(self, key: str, value: Any, ttl_seconds: float = 3600) -> None:
        """Store value in cache with TTL."""
        self._cleanup_expired()

        if key in self._cache:
            old_size = self._cache[key].size_bytes
            self._current_size_bytes -= old_size

        try:
            size = len(pickle.dumps(value))
        except Exception:
            size = 0

        entry = CacheEntry(value=value, timestamp=time.time(), ttl_seconds=ttl_seconds, size_bytes=size)
        self._cache[key] = entry
        self._current_size_bytes += size

        self._evict_if_necessary()

    def get(self, key: str) -> Any | None:
        """Retrieve value if not expired."""
        if key not in self._cache:
            self._stats["misses"] += 1
            return None

        entry = self._cache[key]
        if entry.is_expired():
            del self._cache[key]
            self._current_size_bytes -= entry.size_bytes
            self._stats["misses"] += 1
            return None

        entry.hit_count += 1
        self._stats["hits"] += 1
        return entry.value

    def _cleanup_expired(self) -> None:
        """Remove all expired entries."""
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for key in expired_keys:
            size = self._cache[key].size_bytes
            del self._cache[key]
            self._current_size_bytes -= size

    def _evict_if_necessary(self) -> None:
        """Evict LRU entries if cache exceeds size limit."""
        while self._current_size_bytes > self._max_size_bytes and self._cache:
            lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].hit_count)
            size = self._cache[lru_key].size_bytes
            del self._cache[lru_key]
            self._current_size_bytes -= size
            self._stats["evictions"] += 1

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._current_size_bytes = 0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        self._cleanup_expired()
        hit_rate = self._stats["hits"] / (self._stats["hits"] + self._stats["misses"] + 1e-9)
        return {
            "entries": len(self._cache),
            "size_mb": self._current_size_bytes / (1024 * 1024),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": round(hit_rate, 3),
            "evictions": self._stats["evictions"],
        }


class PersistentCache:
    """Disk-backed cache with JSON serialization."""

    def __init__(self, cache_dir: Path):
        """Initialize persistent cache directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.cache_dir / ".cache_metadata.json"

    def _metadata_path(self, key: str) -> Path:
        """Get path for cache metadata file."""
        return self.cache_dir / f".{key}.json"

    def _data_path(self, key: str) -> Path:
        """Get path for cache data file."""
        return self.cache_dir / f"{key}.pkl"

    def set(self, key: str, value: Any, ttl_seconds: float = 86400) -> None:
        """Store value to disk with TTL metadata."""
        data_path = self._data_path(key)
        metadata_path = self._metadata_path(key)

        try:
            with open(data_path, "wb") as f:
                pickle.dump(value, f)

            metadata = {
                "timestamp": time.time(),
                "ttl_seconds": ttl_seconds,
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
        except Exception as e:
            print(f"Warning: Failed to cache to disk: {e}")

    def get(self, key: str) -> Any | None:
        """Retrieve value if not expired."""
        data_path = self._data_path(key)
        metadata_path = self._metadata_path(key)

        if not data_path.exists() or not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            age = time.time() - metadata["timestamp"]
            if age > metadata["ttl_seconds"]:
                data_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                return None

            with open(data_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to retrieve cache from disk: {e}")
            return None

    def clear(self) -> None:
        """Clear cache directory."""
        import shutil

        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# Global runtime cache instance
_runtime_cache = RuntimeCache(max_size_mb=256)


def cached(ttl_seconds: float = 3600):
    """Decorator for caching function results in runtime memory."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            key = _runtime_cache._make_key(func.__name__, *args, **kwargs)
            cached_value = _runtime_cache.get(key)
            if cached_value is not None:
                return cached_value

            result = func(*args, **kwargs)
            _runtime_cache.set(key, result, ttl_seconds=ttl_seconds)
            return result

        return wrapper

    return decorator


def get_cache_stats() -> dict[str, Any]:
    """Get current cache statistics."""
    return _runtime_cache.stats()


def clear_cache() -> None:
    """Clear all runtime cache."""
    _runtime_cache.clear()
