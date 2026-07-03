#!/usr/bin/env python3
"""
scripts/fix_stale_asr_model_manifest.py
One-time hotfix: asr_candidates[].model for the local canto_ft model in the JSONL
metadata files carries 2 stale historical absolute-path prefixes — the repo moved
twice (/mnt/Drive3/Development/AI-ML/canto-corpus ->
/home/typangaa/Documents/canto-corpus -> /home/typangaa/Documents/canto-hk-speech-pipeline)
and each row recorded whatever REPO_ROOT was live at transcription time. Same root
cause as pipeline/catalog/fix_stale_asr_model.py's DuckDB remap (found during P3
session 1 golden-set review, 2026-07-03); this script fixes the JSONL side, which
that DuckDB fix does not touch. A follow-up broader-sample parity test (20 non-golden
segments, 2026-07-03) surfaced the same bug independently: char_agreement() against
a stale-model row silently reads as "" (empty legacy text) instead of a real
comparison.

Verified counts (2026-07-03, grep over metadata/manifest.jsonl): 400,215 rows already
at the current path, 17,351 + 37,733 = 55,084 stale, summing to exactly 455,299 — the
full segment count, matching pipeline/catalog/fix_stale_asr_model.py's DuckDB counts
exactly (same underlying rows, imported from this same manifest).

Usage: python scripts/fix_stale_asr_model_manifest.py [--dry-run]
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = REPO_ROOT / "metadata"
LOG_DIR = METADATA_DIR / "logs"
LOG_FILE = LOG_DIR / "fix_stale_asr_model_manifest.log"

_LOCAL_CT2_SUFFIX = "/data/ct2_models/whisper-large-v2-cantonese+zh"
_STALE_ROOTS = [
    "/mnt/Drive3/Development/AI-ML/canto-corpus",
    "/home/typangaa/Documents/canto-corpus",
]
_CURRENT_MODEL = str(REPO_ROOT) + _LOCAL_CT2_SUFFIX
_STALE_MODELS = {root + _LOCAL_CT2_SUFFIX for root in _STALE_ROOTS}

TARGET_FILES = [
    METADATA_DIR / "manifest.jsonl",
    METADATA_DIR / "train.jsonl",
    METADATA_DIR / "val.jsonl",
]


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fix_stale_asr_model_manifest")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def process_file(file_path: Path, dry_run: bool, logger: logging.Logger) -> dict:
    stats = {
        "file": str(file_path.relative_to(REPO_ROOT)),
        "total_rows": 0,
        "remapped_candidates": 0,
        "rows_touched": 0,
        "skipped_malformed": 0,
    }

    if not file_path.exists():
        logger.warning("File not found, skipping: %s", file_path)
        return stats

    logger.info("Processing: %s (dry_run=%s)", file_path, dry_run)

    output_lines: list[str] = []
    with file_path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            stripped = raw_line.rstrip("\n")
            if not stripped:
                output_lines.append(stripped)
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d Malformed JSON, skipping line: %s",
                                file_path.name, lineno, exc)
                stats["skipped_malformed"] += 1
                output_lines.append(stripped)
                continue

            stats["total_rows"] += 1
            candidates = record.get("asr_candidates", [])
            row_touched = False
            for cand in candidates:
                model = cand.get("model", "")
                if model in _STALE_MODELS:
                    cand["model"] = _CURRENT_MODEL
                    stats["remapped_candidates"] += 1
                    row_touched = True
            if row_touched:
                stats["rows_touched"] += 1

            output_lines.append(json.dumps(record, ensure_ascii=False))

    if dry_run:
        logger.info("[DRY-RUN] Would write %d lines to %s (no changes written).",
                     len(output_lines), file_path)
        return stats

    bak_path = file_path.with_suffix(file_path.suffix + ".pre-asr-remap.bak")
    if bak_path.exists():
        logger.info("Backup already exists, skipping backup step: %s", bak_path)
    else:
        shutil.copy2(file_path, bak_path)
        logger.info("Backup created: %s", bak_path)

    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            for line in output_lines:
                fh.write(line + "\n")
        tmp_path.replace(file_path)
        logger.info("Wrote updated file: %s", file_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return stats


def print_summary(all_stats: list[dict], dry_run: bool) -> None:
    prefix = "[DRY-RUN] " if dry_run else ""
    header = (
        f"\n{prefix}{'-' * 90}\n"
        f"{'FILE':<30} {'TOTAL ROWS':>12} {'ROWS TOUCHED':>14} {'CANDIDATES REMAPPED':>22}\n"
        f"{'-' * 90}"
    )
    print(header)
    for s in all_stats:
        print(f"{s['file']:<30} {s['total_rows']:>12} {s['rows_touched']:>14} "
              f"{s['remapped_candidates']:>22}")
    print("-" * 90)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remap stale canto_ft absolute-path model strings in metadata JSONL files."
    )
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would be changed without modifying any files.")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=== fix_stale_asr_model_manifest.py started (dry_run=%s) ===", args.dry_run)
    logger.info("current model: %s", _CURRENT_MODEL)
    logger.info("stale models: %s", sorted(_STALE_MODELS))

    all_stats = [process_file(fp, args.dry_run, logger) for fp in TARGET_FILES]
    print_summary(all_stats, dry_run=args.dry_run)

    total_remapped = sum(s["remapped_candidates"] for s in all_stats)
    print(f"\nDone: {total_remapped} asr_candidates entr{'y' if total_remapped == 1 else 'ies'} "
          f"{'would be ' if args.dry_run else ''}remapped to {_CURRENT_MODEL}")
    logger.info("=== fix_stale_asr_model_manifest.py finished ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
