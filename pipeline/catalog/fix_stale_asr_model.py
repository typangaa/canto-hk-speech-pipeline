"""
pipeline/catalog/fix_stale_asr_model.py
One-time hotfix: asr_results.model for the local canto_ft model was imported from
metadata/manifest.jsonl carrying 3 different historical absolute-path prefixes — the
repo moved twice (/mnt/Drive3/Development/AI-ML/canto-corpus ->
/home/typangaa/Documents/canto-corpus -> /home/typangaa/Documents/canto-hk-speech-pipeline)
and manifest.jsonl's asr_candidates[].model field recorded whatever REPO_ROOT was live
at transcription time. pipeline/nodes/asr.py's model_field('canto_ft') always computes
the CURRENT REPO_ROOT-based path, so without this remap discover_transcribe() would
treat every row still keyed under a stale prefix as "not yet transcribed" and silently
re-run ~55k/455k segments (12%) for no reason — found during P3 session 1's golden-set
review, 2026-07-03 (tests/golden/legacy_snapshot.jsonl surfaced a stale
/mnt/Drive3/... model string). Mirrors scripts/fix_stale_paths.py's B9 audio_path
remap, applied to the DuckDB catalog instead of jsonl files since the catalog is now
the live source of truth.

Verified counts (2026-07-03, grep over manifest.jsonl): 400,215 rows already at the
current path, 17,351 + 37,733 = 55,084 stale, summing to exactly 455,299 — the full
segment count, with no id appearing under more than one prefix. The collision-guard
DELETE below is defensive only; it should be a no-op in practice.

Usage: python -m pipeline.catalog.fix_stale_asr_model [--dry-run]
"""

import argparse
import logging

from pipeline.catalog.catalog import connect
from pipeline.nodes.asr import model_field

log = logging.getLogger(__name__)

_STALE_ROOTS = [
    "/mnt/Drive3/Development/AI-ML/canto-corpus",
    "/home/typangaa/Documents/canto-corpus",
]
_LOCAL_CT2_SUFFIX = "/data/ct2_models/whisper-large-v2-cantonese+zh"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    current = model_field("canto_ft")
    conn = connect()

    total_stale = 0
    for stale_root in _STALE_ROOTS:
        stale_model = stale_root + _LOCAL_CT2_SUFFIX
        count = conn.execute(
            "SELECT COUNT(*) FROM asr_results WHERE model = ?", [stale_model]
        ).fetchone()[0]
        log.info(f"{stale_model}: {count} rows")
        total_stale += count
        if not count or args.dry_run:
            continue

        # asr_results PK is (id, model) — a blind UPDATE would collide if a row
        # already exists for the same id under the current model string. Drop
        # any such stale duplicate first (keeps the current-path row).
        collided = conn.execute(
            "SELECT COUNT(*) FROM asr_results s WHERE s.model = ? AND EXISTS "
            "(SELECT 1 FROM asr_results c WHERE c.id = s.id AND c.model = ?)",
            [stale_model, current],
        ).fetchone()[0]
        if collided:
            log.warning(f"  {collided} id(s) already have a current-path row — "
                        f"dropping the stale duplicate for those")
            conn.execute(
                "DELETE FROM asr_results WHERE model = ? AND id IN "
                "(SELECT id FROM asr_results WHERE model = ?)",
                [stale_model, current],
            )

        conn.execute("UPDATE asr_results SET model = ? WHERE model = ?", [current, stale_model])
        remaining = conn.execute(
            "SELECT COUNT(*) FROM asr_results WHERE model = ?", [stale_model]
        ).fetchone()[0]
        log.info(f"  remapped -> {current} (remaining stale rows: {remaining})")

    final_count = conn.execute(
        "SELECT COUNT(*) FROM asr_results WHERE model = ?", [current]
    ).fetchone()[0]
    log.info(f"Total stale rows found: {total_stale}. Rows at current path now: {final_count}")
    print(f"\nDone: {total_stale} stale row(s) {'would be ' if args.dry_run else ''}remapped to {current}")
    print(f"Current-path row count: {final_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
