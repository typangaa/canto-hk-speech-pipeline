"""
Read-only accessor for storage_layout.yaml.
Usage:
    from config.storage_layout import STORAGE, path
    raw = path("raw_youtube")          # → /mnt/Drive2/canto-corpus/data/raw/youtube
    seg = path("segments_root")        # → /mnt/Drive4/canto/segments
"""
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


def raw_path(source: str) -> Path:
    """Return raw dir for a named source (youtube / rthk / podcast)."""
    key = f"raw_{source}"
    if key not in STORAGE:
        raise KeyError(f"Unknown raw source '{source}' — add raw_{source} to storage_layout.yaml")
    return Path(STORAGE[key])
