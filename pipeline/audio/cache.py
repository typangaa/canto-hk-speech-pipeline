"""
pipeline/audio/cache.py
───────────────────────
On-disk LRU cache for decoded audio arrays produced by ``pipeline/audio/bus.py``.

Keying rationale
~~~~~~~~~~~~~~~~
Cache keys are ``(id, sample_rate)`` rather than ``(path, sample_rate)`` because:

  • Segment paths live on a warm-tier spinning-disk mount that may move between
    runs (``/mnt/Drive4/canto/segments/`` is a mount point, not a stable location).
  • The 12-hex-char catalog id (e.g. ``"a1b2c3d4e5f6"``) is the segment's stable
    primary key in DuckDB and will not change even if the underlying WAV is moved
    or re-exported.

Relationship to bus.py
~~~~~~~~~~~~~~~~~~~~~~
This module is intentionally **decoupled** from ``bus.py``:

  • ``bus.py`` knows nothing about caching.
  • ``cache.py`` knows nothing about audio decoding.
  • The calling node (label-suite or a supervisor) is responsible for the
    check-before-decode pattern::

        arrays = cache.get(seg_id, sr)
        if arrays is None:
            arrays = bus.decode(path, sr)
            if arrays is not None:
                cache.put(seg_id, sr, arrays)

Eviction
~~~~~~~~
``evict_lru()`` uses mtime as the recency signal.  ``get()`` updates mtime on
every cache hit via ``os.utime(path, None)`` (sets to current time, no data
read/write).  Eviction is **not** triggered automatically on ``put()`` -- the
orchestrator supervisor should call ``evict_lru()`` periodically (e.g. once per
N batches) to keep the cache below ``DECODE_CACHE["max_gb"]``.  Running a full
directory walk on every write over ~1.3 M cache files would be far too slow.

On-disk layout
~~~~~~~~~~~~~~
``{root}/{sr}/{id[:2]}/{id}.npy``

The 2-character shard prefix keeps individual directory entry counts well inside
ext4 comfort levels (~65 k entries per directory at worst) even at 455 k segments
x 3 sample rates.

Dependencies (CPU-only, no torch):
    numpy, pathlib, os, tempfile
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from config.storage_layout import DECODE_CACHE

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Derived config
# ---------------------------------------------------------------------------

_ROOT = Path(DECODE_CACHE["root"])
_MAX_GB: float = float(DECODE_CACHE["max_gb"])
_BYTES_PER_GB: int = 1 << 30  # 1 073 741 824


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _cache_path(id: str, sample_rate: int) -> Path:
    """Return the canonical on-disk path for a given ``(id, sample_rate)`` key.

    Layout: ``{root}/{sample_rate}/{id[:2]}/{id}.npy``
    """
    shard = id[:2]
    return _ROOT / str(sample_rate) / shard / f"{id}.npy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get(id: str, sample_rate: int) -> Optional[np.ndarray]:
    """Return the cached float32 array for ``(id, sample_rate)``, or ``None``."""
    path = _cache_path(id, sample_rate)
    if not path.exists():
        return None

    try:
        arr = np.load(path)
        os.utime(path, None)  # bump mtime -> LRU recency signal
        return arr
    except Exception as e:
        log.warning(f"cache read fail {path}: {e}")
        return None


def put(id: str, sample_rate: int, array: np.ndarray) -> None:
    """Write *array* to the cache for ``(id, sample_rate)`` atomically."""
    path = _cache_path(id, sample_rate)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npy.tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.save(fh, array)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        log.warning(f"cache write fail {path}: {e}")


def evict_lru(target_gb: Optional[float] = None) -> dict:
    """Delete oldest-mtime files until total size is below *target_gb*."""
    threshold_bytes = int((target_gb if target_gb is not None else _MAX_GB) * _BYTES_PER_GB)

    entries: list[tuple[float, int, Path]] = []
    total_bytes = 0

    if _ROOT.exists():
        for dirpath, _dirs, filenames in os.walk(_ROOT):
            for fname in filenames:
                if not fname.endswith(".npy"):
                    continue
                fpath = Path(dirpath) / fname
                try:
                    st = fpath.stat()
                    entries.append((st.st_mtime, st.st_size, fpath))
                    total_bytes += st.st_size
                except OSError:
                    pass

    total_before = total_bytes
    freed_bytes = 0
    files_deleted = 0

    if total_bytes > threshold_bytes:
        entries.sort(key=lambda t: t[0])

        for mtime, size, fpath in entries:
            if total_bytes <= threshold_bytes:
                break
            try:
                fpath.unlink()
                total_bytes -= size
                freed_bytes += size
                files_deleted += 1
            except OSError as e:
                log.warning(f"evict_lru: could not delete {fpath}: {e}")

        log.info(
            f"evict_lru: deleted {files_deleted} files, "
            f"freed {freed_bytes / _BYTES_PER_GB:.2f} GB"
        )

    return {
        "freed_bytes": freed_bytes,
        "files_deleted": files_deleted,
        "total_bytes_before": total_before,
        "total_bytes_after": total_bytes,
    }


def stats() -> dict:
    """Return ``{total_bytes, n_files, root}`` for observability."""
    total_bytes = 0
    n_files = 0

    if _ROOT.exists():
        for dirpath, _dirs, filenames in os.walk(_ROOT):
            for fname in filenames:
                if not fname.endswith(".npy"):
                    continue
                fpath = Path(dirpath) / fname
                try:
                    total_bytes += fpath.stat().st_size
                    n_files += 1
                except OSError:
                    pass

    return {
        "total_bytes": total_bytes,
        "n_files": n_files,
        "root": str(_ROOT),
    }
