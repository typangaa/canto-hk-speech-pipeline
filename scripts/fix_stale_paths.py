#!/usr/bin/env python3
"""
scripts/fix_stale_paths.py
One-time hotfix: remap stale pre-migration absolute paths in metadata JSONL files.
Usage: python scripts/fix_stale_paths.py [--dry-run]
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = REPO_ROOT / "metadata"
LOG_DIR = METADATA_DIR / "logs"
LOG_FILE = LOG_DIR / "fix_stale_paths.log"

# Each entry: (file_path, field_name, old_prefix, new_prefix)
REMAP_RULES = [
    (
        METADATA_DIR / "manifest.jsonl",
        "audio_path",
        "/mnt/Drive1/canto/",
        "/mnt/Drive4/canto/",
    ),
    (
        METADATA_DIR / "train.jsonl",
        "audio_path",
        "/mnt/Drive1/canto/",
        "/mnt/Drive4/canto/",
    ),
    (
        METADATA_DIR / "val.jsonl",
        "audio_path",
        "/mnt/Drive1/canto/",
        "/mnt/Drive4/canto/",
    ),
    (
        METADATA_DIR / "downloaded.jsonl",
        "wav_path",
        "/mnt/Drive3/Development/AI-ML/canto-corpus/",
        "/mnt/Drive2/canto-corpus/",
    ),
]

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fix_stale_paths")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_file(
    file_path: Path,
    field: str,
    old_prefix: str,
    new_prefix: str,
    dry_run: bool,
    logger: logging.Logger,
) -> dict:
    """
    Process a single JSONL file according to the remap rule.

    Returns a stats dict:
        {
            "file": str,
            "total_rows": int,
            "remapped": int,
            "unchanged": int,
            "missing_after_remap": int,
            "missing_examples": list[str],   # up to 5 offending paths
            "skipped_malformed": int,
        }
    """
    stats = {
        "file": str(file_path.relative_to(REPO_ROOT)),
        "total_rows": 0,
        "remapped": 0,
        "unchanged": 0,
        "missing_after_remap": 0,
        "missing_examples": [],
        "skipped_malformed": 0,
    }

    if not file_path.exists():
        logger.warning("File not found, skipping: %s", file_path)
        return stats

    logger.info("Processing: %s  (field=%r, dry_run=%s)", file_path, field, dry_run)

    # --- Read and process lines ---
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
                logger.warning(
                    "%s:%d  Malformed JSON, skipping line: %s",
                    file_path.name,
                    lineno,
                    exc,
                )
                stats["skipped_malformed"] += 1
                output_lines.append(stripped)
                continue

            stats["total_rows"] += 1
            field_value: str = record.get(field, "")
            remapped = False

            if isinstance(field_value, str) and field_value.startswith(old_prefix):
                new_value = new_prefix + field_value[len(old_prefix):]
                record[field] = new_value
                field_value = new_value
                remapped = True
                stats["remapped"] += 1
                logger.info(
                    "%s:%d  Remapped %r -> %r",
                    file_path.name,
                    lineno,
                    old_prefix + "...",
                    new_prefix + "...",
                )
            else:
                stats["unchanged"] += 1

            # Verify existence of the (possibly remapped) path
            if field_value and not os.path.exists(field_value):
                stats["missing_after_remap"] += 1
                if len(stats["missing_examples"]) < 5:
                    stats["missing_examples"].append(field_value)
                if remapped:
                    logger.warning(
                        "%s:%d  Path missing after remap: %s",
                        file_path.name,
                        lineno,
                        field_value,
                    )
                # Also flag unchanged paths that are missing
                else:
                    logger.debug(
                        "%s:%d  Path missing (unchanged): %s",
                        file_path.name,
                        lineno,
                        field_value,
                    )

            output_lines.append(json.dumps(record, ensure_ascii=False))

    if dry_run:
        logger.info(
            "[DRY-RUN] Would write %d lines to %s (no changes written).",
            len(output_lines),
            file_path,
        )
        return stats

    # --- Backup ---
    bak_path = file_path.with_suffix(file_path.suffix + ".pre-remap.bak")
    if bak_path.exists():
        logger.info(
            "Backup already exists, skipping backup step: %s", bak_path
        )
    else:
        shutil.copy2(file_path, bak_path)
        logger.info("Backup created: %s", bak_path)

    # --- Atomic write ---
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            for line in output_lines:
                fh.write(line + "\n")
        os.replace(tmp_path, file_path)
        logger.info("Wrote updated file: %s", file_path)
    except Exception:
        # Clean up tmp on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return stats


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(all_stats: list[dict], dry_run: bool) -> None:
    prefix = "[DRY-RUN] " if dry_run else ""
    header = (
        f"\n{prefix}{'─' * 90}\n"
        f"{'FILE':<35} {'TOTAL':>8} {'REMAPPED':>10} {'UNCHANGED':>10} {'MISSING':>10}\n"
        f"{'─' * 90}"
    )
    print(header)
    for s in all_stats:
        print(
            f"{s['file']:<35} {s['total_rows']:>8} {s['remapped']:>10} "
            f"{s['unchanged']:>10} {s['missing_after_remap']:>10}"
        )
    print("─" * 90)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remap stale pre-migration absolute paths in metadata JSONL files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be changed without modifying any files.",
    )
    args = parser.parse_args()

    logger = setup_logging()
    logger.info(
        "=== fix_stale_paths.py started (dry_run=%s) ===", args.dry_run
    )

    all_stats: list[dict] = []

    for file_path, field, old_prefix, new_prefix in REMAP_RULES:
        stats = process_file(
            file_path=file_path,
            field=field,
            old_prefix=old_prefix,
            new_prefix=new_prefix,
            dry_run=args.dry_run,
            logger=logger,
        )
        all_stats.append(stats)

    print_summary(all_stats, dry_run=args.dry_run)

    # --- Exit code logic ---
    exit_code = 0
    for s in all_stats:
        if s["missing_after_remap"] > 0:
            exit_code = 1
            mode = "[DRY-RUN] " if args.dry_run else ""
            print(
                f"\n{mode}ERROR: {s['file']} has {s['missing_after_remap']} "
                f"path(s) missing after remap. First up to 5 offending path(s):"
            )
            for path in s["missing_examples"]:
                print(f"  - {path}")

    if exit_code == 0:
        logger.info("All paths verified. Exiting with code 0.")
    else:
        logger.error(
            "One or more files have missing paths after remap. Exiting with code 1."
        )

    logger.info("=== fix_stale_paths.py finished ===")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
