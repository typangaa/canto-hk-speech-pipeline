#!/usr/bin/env python3
"""
scripts/07_g2p.py
Convert verified Cantonese text to Jyutping romanisation (G2P).
Usage: python scripts/07_g2p.py --source [rthk|youtube|podcast|all] [--dry-run]

Tool: canto-hk-g2p (Rust-core, PyPipeline.convert_detailed)
Input: data/filtered/*.filter.json — text field (text_verified must be True)
Output: *.jyutping.json alongside each WAV

Output format: pure Jyutping syllables only (e.g. "nei5 hou2 ge3").
English tokens and punctuation are silently excluded — no bracket placeholders.
Validation: every token must match ^[a-z]+[1-6]$.
Segments with < 80% valid tokens are rejected; 80–95% emit a warning.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import canto_hk_g2p
from canto_hk_g2p._canto_hk_g2p import PyPipeline as _PyPipeline

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "metadata" / "logs" / "07_g2p.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

FILTERED_DIR = ROOT / "data" / "filtered"
G2P_REPORT_PATH = ROOT / "metadata" / "g2p_report.json"

JYUTPING_TOKEN = re.compile(r"^[a-z]+[1-6]$")

# Load dicts from absolute path so this script works from any CWD.
_G2P_DATA_DIR = str(Path(canto_hk_g2p.__file__).resolve().parent.parent.parent / "data")
_G2P = _PyPipeline.from_dir(_G2P_DATA_DIR)


def _convert_for_moss(text: str) -> str:
    """Return space-separated Jyutping for Cantonese tokens only (no English/punct)."""
    tokens = _G2P.convert_detailed(text)
    parts = [jp for _, jp, lang in tokens if lang == "yue"]
    return " ".join(parts)


def text_to_jyutping(text: str) -> Optional[str]:
    """Convert Cantonese text to space-separated Jyutping string.

    English tokens and punctuation are excluded from the output.
    Returns None if no Cantonese tokens were found.
    """
    try:
        jyutping = _convert_for_moss(text)
    except Exception as exc:
        log.error(f"canto-g2p failed: {exc}")
        return None
    return jyutping if jyutping else None


def validate_jyutping(jyutping: str) -> tuple[bool, float, list[str]]:
    """Returns (accept, valid_fraction, bad_tokens)."""
    tokens = jyutping.strip().split()
    if not tokens:
        return True, 1.0, []
    valid = [t for t in tokens if JYUTPING_TOKEN.match(t)]
    bad = [t for t in tokens if not JYUTPING_TOKEN.match(t)]
    frac = len(valid) / len(tokens)
    return frac >= 0.80, round(frac, 3), bad


def process_segment(wav_path: Path, dry_run: bool) -> Optional[dict]:
    out_path = wav_path.with_suffix(".jyutping.json")
    if out_path.exists():
        return None  # already done

    filter_path = wav_path.with_suffix(".filter.json")
    if not filter_path.exists():
        log.debug(f"No filter.json for {wav_path.name}")
        return None

    with open(filter_path) as f:
        fdata = json.load(f)

    # text_verified: soft flag only — pipeline runs on ASR text; Stage 9 filters on True for training.
    is_verified = bool(fdata.get("text_verified"))

    text = fdata.get("text", "").strip()
    if not text:
        return None

    jyutping = text_to_jyutping(text)
    if not jyutping:
        log.warning(f"G2P returned empty for: {wav_path.name}")
        return None

    accept, frac, bad = validate_jyutping(jyutping)

    if not accept:
        log.warning(f"REJECT low Jyutping validity {frac:.2f} {bad[:5]}: {wav_path.name}")
        return None

    if frac < 0.95:
        log.info(f"  Warning: Jyutping validity {frac:.2f} {bad[:3]}: {wav_path.name}")

    record = {
        "wav_path": str(wav_path),
        "text": text,
        "jyutping": jyutping,
        "valid_fraction": frac,
        "text_verified": is_verified,
    }

    if not dry_run:
        with open(out_path, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all",
                        choices=["rthk", "youtube", "podcast", "hktv", "all"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.source == "all":
        wavs = sorted(FILTERED_DIR.rglob("*.wav"))
    else:
        wavs = sorted((FILTERED_DIR / args.source).rglob("*.wav"))

    todo = [w for w in wavs if not w.with_suffix(".jyutping.json").exists()]
    log.info(f"Found {len(wavs)} WAVs, {len(todo)} need G2P")

    processed = skipped = rejected = 0
    frac_sum = 0.0

    for wav_path in todo:
        try:
            rec = process_segment(wav_path, args.dry_run)
            if rec:
                processed += 1
                frac_sum += rec["valid_fraction"]
            else:
                skipped += 1
        except Exception as exc:
            log.error(f"Failed {wav_path.name}: {exc}", exc_info=True)
            rejected += 1

    avg_frac = frac_sum / max(processed, 1)
    print(f"\nDone: {processed} G2P generated, {skipped} skipped, {rejected} failed")
    print(f"Average Jyutping validity: {avg_frac:.3f}")

    report = {
        "processed": processed,
        "skipped": skipped,
        "rejected": rejected,
        "avg_valid_fraction": round(avg_frac, 3),
    }
    G2P_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(G2P_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
