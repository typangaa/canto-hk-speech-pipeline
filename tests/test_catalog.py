"""P0 gate tests — verify the DuckDB catalog matches the legacy jsonl source-of-truth exactly. Run: pytest tests/test_catalog.py -v

labels_music / labels_lang / labels_overlap (and task_runs) are excluded from the
frozen EXPECTED counts below: since P1's label.music and P2's label.suite
orchestrator nodes went live, they are live-growing tables that new pilot/
production runs append to — see test_labels_music_monotonic_growth,
test_labels_music_provenance_breakdown, and test_labels_lang_overlap_monotonic_growth
for their invariants instead.
"""

import os

import pytest

from pipeline.catalog.catalog import connect_ro
from pipeline.config import CATALOG_PATH

# P0 legacy-import baseline, frozen — these tables are never written to again
# after the one-time P0 import (their source jsonl files are static).
EXPECTED = {
    "segments": 455299,
    "raw_files": 6272,
    "asr_results": 910598,
    "asr_agreement": 455299,
    "filters": 455299,
    "g2p": 455299,
    "tiers": 455299,
}

# P0 import baseline for labels_music, before P1's orchestrator started
# appending new rows — a floor, not an exact count.
LABELS_MUSIC_P0_BASELINE = 105668

# P0 import baseline for labels_lang / labels_overlap, before P2's label.suite
# node started filling the ~24-row legacy-import gap in each — a floor, not an
# exact count. Full corpus size (455299) is the ceiling both approach.
LABELS_LANG_OVERLAP_P0_BASELINE = 455275


@pytest.fixture(scope="module")
def catalog_conn():
    if not os.path.exists(CATALOG_PATH):
        pytest.skip("catalog not built — run: python -m pipeline.cli catalog build")
    conn = connect_ro()
    yield conn
    conn.close()


def test_row_counts(catalog_conn):
    failures = []
    for table, expected in EXPECTED.items():
        actual = catalog_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if actual != expected:
            failures.append(
                f"  table={table!r}: expected {expected}, got {actual}"
            )
    assert not failures, "Row count mismatches:\n" + "\n".join(failures)


def test_segments_primary_key_unique(catalog_conn):
    total = catalog_conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    distinct = catalog_conn.execute("SELECT COUNT(DISTINCT id) FROM segments").fetchone()[0]
    assert total == distinct, (
        f"segments.id is not unique: total rows={total}, distinct ids={distinct}"
    )


def test_path_exists_sample(catalog_conn):
    rows = catalog_conn.execute(
        "SELECT audio_path FROM segments USING SAMPLE 500 ROWS"
    ).fetchall()
    missing = [row[0] for row in rows if not os.path.exists(row[0])]
    if missing:
        preview = missing[:5]
        pytest.fail(
            f"{len(missing)} sampled audio_path(s) do not exist on disk. "
            f"First up to 5 missing:\n" + "\n".join(f"  {p}" for p in preview)
        )


def test_no_stale_drive1_paths(catalog_conn):
    count = catalog_conn.execute(
        "SELECT COUNT(*) FROM segments WHERE audio_path LIKE ?", ["/mnt/Drive1/%"]
    ).fetchone()[0]
    assert count == 0, (
        f"Found {count} segment(s) with stale /mnt/Drive1/ paths in segments.audio_path"
    )


def test_discovery_sql_matches_arithmetic(catalog_conn):
    total_segments = catalog_conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    total_music = catalog_conn.execute("SELECT COUNT(*) FROM labels_music").fetchone()[0]
    anti_join_count = catalog_conn.execute(
        "SELECT COUNT(*) FROM segments s LEFT JOIN labels_music m ON s.id = m.id WHERE m.id IS NULL"
    ).fetchone()[0]
    expected_anti_join = total_segments - total_music
    assert anti_join_count == expected_anti_join, (
        f"Anti-join result ({anti_join_count}) != total_segments - total_music "
        f"({total_segments} - {total_music} = {expected_anti_join})"
    )


def test_labels_music_monotonic_growth(catalog_conn):
    total = catalog_conn.execute("SELECT COUNT(*) FROM labels_music").fetchone()[0]
    assert total >= LABELS_MUSIC_P0_BASELINE, (
        f"labels_music has {total} rows, below the P0 import baseline of "
        f"{LABELS_MUSIC_P0_BASELINE} — rows should only ever be added, never lost"
    )
    dupes = catalog_conn.execute(
        "SELECT id, COUNT(*) FROM labels_music GROUP BY id HAVING COUNT(*) != 1"
    ).fetchall()
    assert not dupes, f"duplicate ids in labels_music: {dupes[:5]}"


def test_labels_music_provenance_breakdown(catalog_conn):
    """s0/s1/tag_calib are the frozen P0 import — never written to again.
    p1_pilot/p2_suite/read_failed are orchestrator output and only ever grow
    (p1_pilot from label.music, p2_suite from label.suite's decode-once fan-out).
    """
    rows = catalog_conn.execute(
        "SELECT provenance, COUNT(*) FROM labels_music GROUP BY provenance"
    ).fetchall()
    actual = {row[0]: row[1] for row in rows}
    frozen = {"s0": 52795, "s1": 52404, "tag_calib": 469}
    for provenance, expected_count in frozen.items():
        assert actual.get(provenance) == expected_count, (
            f"frozen P0 provenance {provenance!r} changed: "
            f"expected {expected_count}, got {actual.get(provenance)}"
        )
    live_provenances = set(actual) - set(frozen)
    allowed_live = {"p1_pilot", "p2_suite", "read_failed"}
    assert live_provenances <= allowed_live, (
        f"unexpected provenance value(s) in labels_music: "
        f"{live_provenances - allowed_live}"
    )


def test_labels_lang_overlap_monotonic_growth(catalog_conn):
    """P0 imported 455275/455299 rows for both tables (24-segment gap — an
    isolated batch of zero-byte podcast files, see docs/REARCHITECTURE_
    IMPLEMENTATION_PLAN.md). P2's label.suite node fills this gap opportunistically
    alongside its music pass; rows should only ever be added, never lost, and
    never exceed the full corpus size.
    """
    total_segments = catalog_conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    for table in ("labels_lang", "labels_overlap"):
        total = catalog_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert total >= LABELS_LANG_OVERLAP_P0_BASELINE, (
            f"{table} has {total} rows, below the P0 import baseline of "
            f"{LABELS_LANG_OVERLAP_P0_BASELINE} — rows should only ever be added, never lost"
        )
        assert total <= total_segments, (
            f"{table} has {total} rows, more than segments ({total_segments})"
        )
        dupes = catalog_conn.execute(
            f"SELECT id, COUNT(*) FROM {table} GROUP BY id HAVING COUNT(*) != 1"
        ).fetchall()
        assert not dupes, f"duplicate ids in {table}: {dupes[:5]}"


def test_asr_results_two_models_per_segment(catalog_conn):
    offenders = catalog_conn.execute(
        "SELECT id, COUNT(*) FROM asr_results GROUP BY id HAVING COUNT(*) != 2"
    ).fetchall()
    if offenders:
        preview = offenders[:5]
        preview_str = "\n".join(
            f"  id={row[0]!r}, count={row[1]}" for row in preview
        )
        pytest.fail(
            f"{len(offenders)} segment id(s) in asr_results do not have exactly 2 rows. "
            f"First up to 5 offenders:\n{preview_str}"
        )
