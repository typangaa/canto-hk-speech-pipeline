#!/usr/bin/env python3
"""
pipeline/catalog/verify.py
P0 gate checks (docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md §12):
  1. Row counts only ever grow (floor check against a dated baseline, plus a
     <= segments ceiling for tables that are 1:1-with-segments in principle).
     Originally (P0) this was an exact-match check against a frozen legacy-jsonl
     import count, but the corpus has grown continuously since (orphan recovery,
     backlog processing, ongoing ingestion — now 618,695+ segments and counting),
     so exact equality now always fails. Same fix pattern as
     tests/test_catalog.py's *_monotonic_growth tests — see that file's docstring
     for the full history.
  2. Path-existence sample check == 100% (already run during build; re-verified here).
  3. Discovery-SQL arithmetic: segments minus labels_music == the SQL anti-join
     result exactly (validates the "music not-yet-tagged" query the P1 pilot
     will actually run against the catalog).
  4. Zero stale pre-migration paths remain in the on-disk legacy jsonl files.

Usage: python -m pipeline.catalog.verify
"""

import os
import subprocess
import sys

from pipeline.catalog.catalog import connect_ro
from pipeline.config import REPO_ROOT

METADATA_DIR = REPO_ROOT / "metadata"

# Row-count floor baselines, updated 2026-07-11 when this exact-match check was
# replaced with a floor: row counts should only ever grow from here, never
# shrink. Values below are the live catalog counts queried on 2026-07-11 (see
# tests/test_catalog.py for the equivalent, previously-established pattern —
# SEGMENTS_P0_BASELINE / RAW_FILES_BASELINE / ONE_TO_ONE_TABLE_BASELINES /
# G2P_BASELINE / LABELS_MUSIC_P0_BASELINE / LABELS_LANG_OVERLAP_P0_BASELINE).
#
# - segments / raw_files: top of the growth chain, floor only, no ceiling.
# - asr_agreement / filters / tiers: 1:1-with-segments in principle (become
#   eligible once >= 2 ASR models have landed for a segment — see
#   asr.agreement's docstring), so floor + a <= segments ceiling.
# - g2p: runs only on human-verified/agreement-passing text, a subset of
#   asr_agreement — floor only.
# - labels_lang / labels_overlap / labels_music: floor only (labels_music has
#   already caught up to the full segments count as of 2026-07-11; the other
#   two still trail slightly — label.suite fills the gap opportunistically).
# - asr_results: not 1:1 with segments (multiple ASR models write multiple
#   rows per segment id, so its natural ceiling is much higher than segments
#   count) — floor only, no ceiling.
ROW_COUNT_FLOORS = {
    "segments": 618695,
    "raw_files": 10990,
    "asr_results": 2508851,
    "asr_agreement": 618695,
    "filters": 618695,
    "g2p": 484832,
    "tiers": 618695,
    "labels_lang": 455642,
    "labels_overlap": 455642,
    "labels_music": 618695,
}

# Tables that are 1:1-with-segments in principle and therefore also get a
# <= total_segments ceiling check, matching tests/test_catalog.py's
# ONE_TO_ONE_TABLE_BASELINES convention.
ONE_TO_ONE_WITH_SEGMENTS = {"asr_agreement", "filters", "tiers"}


def check_row_counts(conn) -> list[tuple[str, bool, str]]:
    results = []
    total_segments = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    for table, baseline in ROW_COUNT_FLOORS.items():
        actual = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        ok = actual >= baseline
        detail = f"baseline={baseline} actual={actual} (floor, must be >=)"
        if table in ONE_TO_ONE_WITH_SEGMENTS:
            ceiling_ok = actual <= total_segments
            ok = ok and ceiling_ok
            detail += f" segments={total_segments} (ceiling, must be <=)"
        results.append((f"row_count[{table}]", ok, detail))
    return results


def check_path_exists_sample(conn, sample_size: int = 2000) -> list[tuple[str, bool, str]]:
    results = []
    for table, col in (("segments", "audio_path"), ("raw_files", "wav_path")):
        rows = conn.execute(
            f"SELECT {col} FROM {table} USING SAMPLE {sample_size} ROWS"
        ).fetchall()
        missing = [r[0] for r in rows if not os.path.exists(r[0])]
        ok = len(missing) == 0
        detail = f"sampled={len(rows)} missing={len(missing)}"
        if missing:
            detail += f" examples={missing[:3]}"
        results.append((f"path_exists[{table}]", ok, detail))
    return results


def check_discovery_sql(conn) -> list[tuple[str, bool, str]]:
    total_segments = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    total_music = conn.execute("SELECT COUNT(*) FROM labels_music").fetchone()[0]
    expected_todo = total_segments - total_music

    actual_todo = conn.execute("""
        SELECT COUNT(*) FROM segments s
        LEFT JOIN labels_music m ON s.id = m.id
        WHERE m.id IS NULL
    """).fetchone()[0]

    ok = actual_todo == expected_todo
    detail = (f"segments={total_segments} labels_music={total_music} "
              f"expected_todo={expected_todo} actual_anti_join={actual_todo}")
    return [("discovery_sql[labels_music]", ok, detail)]


def check_stale_paths() -> list[tuple[str, bool, str]]:
    results = []
    checks = [
        (METADATA_DIR / "manifest.jsonl", "/mnt/Drive1/"),
        (METADATA_DIR / "train.jsonl", "/mnt/Drive1/"),
        (METADATA_DIR / "val.jsonl", "/mnt/Drive1/"),
        (METADATA_DIR / "downloaded.jsonl", "/mnt/Drive3/Development"),
    ]
    for path, pattern in checks:
        if not path.exists():
            results.append((f"stale_path[{path.name}]", False, f"file not found: {path}"))
            continue
        count = int(
            subprocess.run(
                ["grep", "-c", pattern, str(path)],
                capture_output=True, text=True, check=False,
            ).stdout.strip() or "0"
        )
        ok = count == 0
        results.append((f"stale_path[{path.name}]", ok, f"pattern={pattern!r} count={count}"))
    return results


def main() -> int:
    conn = connect_ro()
    try:
        all_results = (
            check_row_counts(conn)
            + check_path_exists_sample(conn)
            + check_discovery_sql(conn)
            + check_stale_paths()
        )
    finally:
        conn.close()

    print(f"\n{'CHECK':<32} {'RESULT':<8} DETAIL")
    print("─" * 100)
    n_fail = 0
    for name, ok, detail in all_results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(f"{name:<32} {status:<8} {detail}")
    print("─" * 100)

    if n_fail:
        print(f"\n{n_fail}/{len(all_results)} checks FAILED — P0 gate not met.")
        return 1

    print(f"\nAll {len(all_results)} checks PASSED — P0 gate met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
