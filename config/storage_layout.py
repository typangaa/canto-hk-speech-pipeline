"""
Read-only accessor for storage_layout.yaml.
Usage:
    from config.storage_layout import STORAGE, path
    raw = path("raw_youtube")          # → /mnt/Drive2/canto-corpus/data/raw/youtube
    seg = path("segments_root")        # → /mnt/Drive4/canto/segments

P5-C sharding (2026-07-06): shard_index()/shard_root() are the single source of
truth for the segments hash-sharding scheme — both pipeline/nodes/segment.py
(new writes) and pipeline/nodes/rebalance.py (one-time backlog migration) call
these, never re-implement the hash inline, so the two never disagree on where a
given key belongs.
"""
import hashlib
from pathlib import Path
import yaml

_HERE = Path(__file__).parent
_LAYOUT = yaml.safe_load((_HERE / "storage_layout.yaml").read_text())

STORAGE: dict = _LAYOUT["storage"]
SHARDING: dict = _LAYOUT["sharding"]
DECODE_CACHE: dict = _LAYOUT["decode_cache"]


def path(key: str) -> Path:
    """Return Path for a storage key; raises KeyError if unknown."""
    return Path(STORAGE[key])


def shard_index(key: str) -> int:
    """Deterministic shard index for *key* in [0, n_shards).

    Uses md5 rather than Python's builtin hash() — the latter is randomized
    per-process (PYTHONHASHSEED) unless explicitly disabled, which would make
    the same key hash to a different shard on every run.
    """
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % SHARDING["n_shards"]


def shard_root(key: str) -> Path:
    """Return the shard root directory *key* hashes to."""
    return Path(SHARDING["shard_roots"][shard_index(key)])


def raw_path(source: str) -> Path:
    """Return raw dir for a named source (youtube / rthk / podcast)."""
    key = f"raw_{source}"
    if key not in STORAGE:
        raise KeyError(f"Unknown raw source '{source}' — add raw_{source} to storage_layout.yaml")
    return Path(STORAGE[key])
