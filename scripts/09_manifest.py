#!/usr/bin/env python3
"""
scripts/09_manifest.py
Assemble final manifest.jsonl from all per-segment metadata files.
Usage: python scripts/09_manifest.py [--dry-run]

Reads from data/filtered/: *.filter.json, *.jyutping.json, *.speaker.json, *.transcript.json
Output: metadata/manifest.jsonl, metadata/train.jsonl, metadata/val.jsonl

Schema: see docs/MANIFEST_SCHEMA.md
Split: 95/5 train/val, stratified by source, no speaker_id overlap across splits.

Data quality tiers (stored in manifest "tier" field):
  gold   — text_verified=True in filter.json (human-confirmed via Stage 5, or ASR agreement >= 0.80)
  silver — ASR agreement >= SILVER_AGREE_MIN (0.65): high audio quality, model disagreement
            likely due to script/encoding differences (Traditional vs Simplified). Safe for
            initial TTS training; should be prioritised for Stage 5 human calibration.
  (excluded) — ASR agreement < 0.65: too uncertain; excluded until Stage 5 review.
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "metadata" / "logs" / "09_manifest.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

FILTERED_DIR = ROOT / "data" / "filtered"
META_DIR = ROOT / "metadata"
MANIFEST_PATH = META_DIR / "manifest.jsonl"
TRAIN_PATH = META_DIR / "train.jsonl"
VAL_PATH = META_DIR / "val.jsonl"

# Minimum ASR agreement to include as "silver" tier without human verification.
# Segments below this threshold require Stage 5 human calibration before inclusion.
SILVER_AGREE_MIN = 0.65

REQUIRED = [
    "id", "audio_path", "source", "source_url", "program", "domain",
    "text", "text_verified", "asr_candidates", "asr_agreement", "jyutping",
    "duration_sec", "sample_rate", "speaker_id",
    "gender", "style", "snr_db", "dnsmos", "english_ratio", "created_at", "tier",
]


def load_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def stable_id(wav_path: Path) -> str:
    return hashlib.md5(str(wav_path.resolve()).encode()).hexdigest()[:12]


def build_entry(wav_path: Path, seg_meta: Optional[dict]) -> Optional[dict]:
    filter_data = load_json(wav_path.with_suffix(".filter.json"))
    jp_data = load_json(wav_path.with_suffix(".jyutping.json"))
    speaker_data = load_json(wav_path.with_suffix(".speaker.json"))
    transcript_data = load_json(
        (FILTERED_DIR.parent / "segments" /
         wav_path.relative_to(FILTERED_DIR)).with_suffix(".transcript.json")
    )

    if not filter_data:
        return None
    if not jp_data:
        return None  # no Jyutping = can't include

    asr_agreement_val = float(filter_data.get("asr_agreement", 0.0))
    human_verified = bool(filter_data.get("text_verified"))

    # Tier assignment:
    #   gold   = human-verified (Stage 5) or auto-verified (ASR agreement >= 0.80)
    #   silver = ASR agreement >= SILVER_AGREE_MIN (0.65); good audio, script divergence
    #   excluded = below SILVER_AGREE_MIN; needs Stage 5 before inclusion
    if human_verified:
        tier = "gold"
    elif asr_agreement_val >= SILVER_AGREE_MIN:
        tier = "silver"
    else:
        return None  # below silver threshold — exclude until Stage 5 review

    # Source metadata: try segment metadata first, then derive from path
    seg = seg_meta or {}
    source = seg.get("source") or filter_data.get("source") or wav_path.relative_to(FILTERED_DIR).parts[0]
    program = seg.get("program") or ""
    domain = seg.get("domain") or "other"
    style = seg.get("style") or "formal"
    source_url = seg.get("source_url") or ""

    # Speaker
    speaker_id = "unknown_000"
    gender = "unknown"
    if speaker_data:
        speaker_id = speaker_data.get("speaker_id", f"{source}_unk")
        gender = speaker_data.get("gender", "unknown")

    # ASR
    asr_candidates = []
    if transcript_data:
        asr_candidates = transcript_data.get("asr_candidates", [])

    entry = {
        "id": stable_id(wav_path),
        "audio_path": str(wav_path.resolve()),
        "source": source,
        "source_url": source_url,
        "program": program,
        "domain": domain,
        "text": filter_data["text"],
        "text_verified": human_verified,
        "asr_candidates": asr_candidates,
        "asr_agreement": round(asr_agreement_val, 3),
        "jyutping": jp_data["jyutping"],
        "duration_sec": round(float(filter_data.get("duration_sec", 0)), 3),
        "sample_rate": int(filter_data.get("sample_rate", 48000)),
        "speaker_id": speaker_id,
        "gender": gender,
        "style": style,
        "snr_db": round(float(filter_data.get("snr_db", 0)), 1),
        "dnsmos": round(float(filter_data.get("dnsmos", 0)), 2),
        "english_ratio": round(float(filter_data.get("english_ratio", 0)), 3),
        "created_at": str(date.today()),
        "tier": tier,
    }

    # Validate required fields
    for field in REQUIRED:
        if field not in entry:
            log.warning(f"Missing field {field} in {wav_path.name}")
            return None

    # Hard gate: paths must be under an expected location (project root, old /mnt/Drive3/,
    # or /mnt/Drive1/ where filtered data now lives after relocation)
    if not (entry["audio_path"].startswith("/mnt/Drive3/")
            or entry["audio_path"].startswith("/mnt/Drive1/")
            or entry["audio_path"].startswith(str(ROOT))):
        log.error(f"Bad audio_path: {entry['audio_path']}")
        return None

    return entry


def load_segment_meta() -> dict[str, dict]:
    """Load metadata from 03_segment.py *_segments.jsonl files."""
    seg_dir = ROOT / "data" / "segments"
    meta: dict[str, dict] = {}
    for jsonl in seg_dir.rglob("*_segments.jsonl"):
        try:
            with open(jsonl) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    meta[rec["seg_path"]] = rec
        except Exception:
            pass
    return meta


def train_val_split(entries: list[dict], val_frac: float = 0.05) -> tuple[list, list]:
    """Stratified by source, no speaker_id in both splits."""
    by_source = defaultdict(list)
    for e in entries:
        by_source[e["source"]].append(e)

    train, val = [], []
    for src, src_entries in by_source.items():
        # Group by speaker_id
        by_speaker: dict[str, list] = defaultdict(list)
        for e in src_entries:
            by_speaker[e["speaker_id"]].append(e)

        speakers = list(by_speaker.keys())
        n_val_spk = max(1, int(len(speakers) * val_frac))
        val_speakers = set(speakers[-n_val_spk:])

        for spk, segs in by_speaker.items():
            if spk in val_speakers:
                val.extend(segs)
            else:
                train.extend(segs)

    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    wavs = sorted(FILTERED_DIR.rglob("*.wav"))
    log.info(f"Found {len(wavs)} WAVs in data/filtered/")

    seg_meta = load_segment_meta()
    log.info(f"Loaded {len(seg_meta)} segment metadata records")

    entries = []
    errors = 0
    for wav_path in wavs:
        # seg_meta is keyed by segments/ path, not filtered/ path
        seg_wav_path = ROOT / "data" / "segments" / wav_path.relative_to(FILTERED_DIR)
        sm = seg_meta.get(str(seg_wav_path))
        entry = build_entry(wav_path, sm)
        if entry:
            entries.append(entry)
        else:
            errors += 1

    log.info(f"Built {len(entries)} manifest entries ({errors} skipped)")

    # Check for duplicate IDs
    ids = [e["id"] for e in entries]
    from collections import Counter
    dups = [k for k, v in Counter(ids).items() if v > 1]
    if dups:
        log.error(f"Duplicate IDs found: {dups[:5]}")

    # Check for unexpected paths (outside project root and old /mnt/Drive3/ location)
    win_paths = [e["audio_path"] for e in entries if not (e["audio_path"].startswith("/mnt/Drive3/") or e["audio_path"].startswith("/mnt/Drive1/") or e["audio_path"].startswith(str(ROOT)))]
    if win_paths:
        log.error(f"Non-Linux paths: {win_paths[:3]}")

    if args.dry_run:
        dur_total = sum(e["duration_sec"] for e in entries) / 3600
        print(f"\n[DRY-RUN] Would write {len(entries)} entries (~{dur_total:.1f}h)")
        print(f"  Duplicate IDs: {len(dups)}")
        print(f"  Bad paths:     {len(win_paths)}")
        return

    META_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        for e in sorted(entries, key=lambda x: x["id"]):
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    train_entries, val_entries = train_val_split(entries)
    with open(TRAIN_PATH, "w") as f:
        for e in train_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(VAL_PATH, "w") as f:
        for e in val_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    total_hours = sum(e["duration_sec"] for e in entries) / 3600
    n_speakers = len(set(e["speaker_id"] for e in entries))

    print(f"\nManifest: {len(entries)} entries ({total_hours:.1f}h), {n_speakers} speakers")
    print(f"  Train: {len(train_entries)}, Val: {len(val_entries)}")
    print(f"  {MANIFEST_PATH}")
    print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
