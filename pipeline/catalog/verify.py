#!/usr/bin/env python3
"""
pipeline/catalog/verify.py
P0 gate checks (docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md §12):
  1. Row counts match the legacy jsonl source-of-truth exactly.
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

# Expected counts, derived directly from the legacy jsonl files (see
# docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md P0 section + session notes).
# raw_files=6272 (not 6631): downloaded.jsonl has 359 duplicate ids from
# 00_reingest.py double-logging some files — see ingest.py import_raw_files().
EXPECTED = {
    "segments": 455299,
    "raw_files": 6272,
    "asr_results": 910598,
    "asr_agreement": 455299,
    "filters": 455299,
    "g2p": 455299,
    "tiers": 455299,
    "labels_lang": 455275,
    "labels_overlap": 455275,
    "labels_music": 105668,
}


def check_row_counts(conn) -> list[tuple[str, bool, str]]:
    results = []
    for table, expected in EXPECTED.items():
        actual = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        ok = actual == expected
        results.append((f"row_count[{table}]", ok, f"expected={expected} actual={actual}"))
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
