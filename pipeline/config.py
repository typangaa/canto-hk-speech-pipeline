"""
pipeline/config.py
Read-only accessor for config/pipeline.yaml, mirroring config/storage_layout.py's pattern.
"""
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # lets us import config.storage_layout regardless of cwd

_CFG = yaml.safe_load((REPO_ROOT / "config" / "pipeline.yaml").read_text())

CATALOG_PATH = REPO_ROOT / _CFG["catalog"]["path"]
GOLDEN_MANIFEST = REPO_ROOT / _CFG["golden"]["manifest"]
GOLDEN_LEGACY_SNAPSHOT = REPO_ROOT / _CFG["golden"]["legacy_snapshot"]
GOLDEN_SAMPLE_SIZE = int(_CFG["golden"]["sample_size"])
LOGS_DIR = REPO_ROOT / _CFG["logs_dir"]
