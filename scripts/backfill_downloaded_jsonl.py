#!/usr/bin/env python3
"""
scripts/backfill_downloaded_jsonl.py
One-off repair: metadata/downloaded.jsonl is missing entries for ~4,648 raw
audio files that physically exist under data/raw/{podcast,rthk,youtube}/ —
confirmed 2026-07-04 while investigating why `raw_files` (6,272 rows, a
faithful reflection of downloaded.jsonl's 6,272 unique ids) undercounts the
10,920 real files on disk (youtube: 511/4,454 logged, podcast: 3,596/4,303,
rthk: 2,012/2,181). Root cause of the gap itself is unknown (predates this
investigation and the responsible download run is no longer identifiable),
but the fix is mechanical: every one of these files already carries a stable
id inside its filename (the same id 02_download.py would have logged), so we
can reconstruct a downloaded.jsonl-compatible row for each missing file by
parsing the filename + probing the audio itself — no new information is
invented, only recovered from what's already encoded in the filename/file.

Filename id extraction is a two-pattern heuristic, validated against all
6,631 EXISTING downloaded.jsonl rows (6,630/6,631 exact match — the one
miss is a pre-existing malformed '...orig' entry unrelated to this backfill):
  - hash-style id:    8 lowercase hex chars,      e.g. ..._61b0896c.wav
  - youtube-style id: 11-char base64url video id, e.g. ..._a959sncUEpA.wav
    (YouTube ids can themselves contain '_' — a naive rsplit('_', 1) on the
    filename is WRONG for these; this is why a fixed-width suffix check is
    used instead of splitting on the delimiter).

This script only APPENDS new lines to downloaded.jsonl (a backup is written
first) — it never rewrites or removes any existing line. After appending, run
`python -m pipeline.catalog.ingest` (or `pipe catalog build`) separately to
rebuild `raw_files` from the now-complete jsonl (import_raw_files() already
TRUNCATEs + re-inserts idempotently — this is its documented normal rebuild
path, not a special/risky operation).

Usage: python scripts/backfill_downloaded_jsonl.py [--dry-run]
"""

import argparse
import datetime
import json
import logging
import re
import sys
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = REPO_ROOT / "metadata"
DOWNLOADED_JSONL = METADATA_DIR / "downloaded.jsonl"
RAW_ROOT = Path("/mnt/Drive2/canto-corpus/data/raw")
SOURCES = ["podcast", "rthk", "youtube", "hktv"]

log_path = Path("metadata/logs") / f"{Path(__file__).stem}.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

_HASH_RE = re.compile(r"[0-9a-f]{8}")
_YT_RE = re.compile(r"[A-Za-z0-9_-]{11}")


def parse_id(stem: str) -> str | None:
    """Reconstruct the raw_id 02_download.py would have logged, from the
    on-disk filename stem (no extension). Returns None if neither pattern
    matches (never guesses)."""
    if len(stem) >= 9 and stem[-9] == "_" and _HASH_RE.fullmatch(stem[-8:]):
        return stem[-8:]
    if len(stem) >= 12 and stem[-12] == "_" and _YT_RE.fullmatch(stem[-11:]):
        return stem[-11:]
    return None


def parse_program_and_pubdate(stem: str, raw_id: str) -> tuple[str, str | None]:
    """Best-effort program (middle filename segment) + pub_date (leading
    YYYYMMDD prefix, if present) — NOT guaranteed accurate metadata, just a
    reasonable placeholder so the row isn't entirely empty. Never invents a
    source_url or domain/style/language (left NULL — see module docstring)."""
    body = stem[: -(len(raw_id) + 1)]  # strip "_<raw_id>"
    m = re.match(r"^(\d{8})_(.*)$", body)
    if m:
        return m.group(2), m.group(1)
    return body, None


def build_row(path: Path, source: str, existing_ids: set[str]) -> dict | None:
    stem = path.stem
    raw_id = parse_id(stem)
    if raw_id is None:
        log.warning(f"Could not parse id from filename, skipping: {path.name}")
        return None
    if raw_id in existing_ids:
        return None  # already logged — not part of the gap

    try:
        info = sf.info(str(path))
        duration_sec = round(info.frames / info.samplerate, 1)
        sample_rate = info.samplerate
    except Exception as e:
        log.warning(f"Could not probe audio, skipping: {path} ({e})")
        return None

    program, pub_date = parse_program_and_pubdate(stem, raw_id)
    source_url = f"https://www.youtube.com/watch?v={raw_id}" if (
        source == "youtube" and _YT_RE.fullmatch(raw_id)
    ) else None
    downloaded_at = datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()

    return {
        "id": raw_id,
        "wav_path": str(path),
        "source": source,
        "source_url": source_url,
        "title": stem,
        "pub_date": pub_date,
        "program": program,
        "domain": None,
        "style": None,
        "language": None,
        "duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "downloaded_at": downloaded_at,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    existing_ids: set[str] = set()
    if DOWNLOADED_JSONL.exists():
        with open(DOWNLOADED_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_ids.add(json.loads(line).get("id"))
                except json.JSONDecodeError:
                    continue
    log.info(f"{len(existing_ids)} unique ids already logged in {DOWNLOADED_JSONL.name}")

    new_rows: list[dict] = []
    per_source_new: dict[str, int] = {}
    per_source_skip_parse: dict[str, int] = {}

    for source in SOURCES:
        src_dir = RAW_ROOT / source
        if not src_dir.exists():
            continue
        for path in sorted(src_dir.glob("*.wav")):
            row = build_row(path, source, existing_ids)
            if row is None:
                if parse_id(path.stem) is None:
                    per_source_skip_parse[source] = per_source_skip_parse.get(source, 0) + 1
                continue
            new_rows.append(row)
            per_source_new[source] = per_source_new.get(source, 0) + 1

    log.info(f"New rows to backfill: {len(new_rows)} — by source: {per_source_new}")
    if per_source_skip_parse:
        log.warning(f"Unparseable filenames (skipped, not backfilled): {per_source_skip_parse}")

    if args.dry_run:
        log.info("--dry-run: not writing anything")
        print(f"\nDone (dry-run): {len(new_rows)} would be appended, "
              f"{sum(per_source_skip_parse.values())} unparseable")
        return 0

    if new_rows:
        backup_path = DOWNLOADED_JSONL.with_suffix(
            f".jsonl.bak-{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}"
        )
        if DOWNLOADED_JSONL.exists():
            backup_path.write_bytes(DOWNLOADED_JSONL.read_bytes())
            log.info(f"Backed up existing jsonl to {backup_path}")

        with open(DOWNLOADED_JSONL, "a", encoding="utf-8") as f:
            for row in new_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info(f"Appended {len(new_rows)} new lines to {DOWNLOADED_JSONL}")

    print(f"\nDone: {len(new_rows)} appended, "
          f"{sum(per_source_skip_parse.values())} unparseable/skipped")
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
