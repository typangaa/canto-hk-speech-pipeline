"""Catalog gate tests — verify the DuckDB catalog's growing tables only ever grow
(never shrink/duplicate) and that their cross-table invariants hold. Run: pytest tests/test_catalog.py -v

Originally (P0) these compared several tables against a frozen exact-count
EXPECTED dict, on the assumption they were written once by the legacy jsonl
import and never touched again. That assumption broke as the corpus grew
past the original 455,299-row P0 import via orphan recovery (recover.orphans),
backlog processing, and — as of 2026-07-09 — the Phase A repair chain
(speaker.embed --verify-existing repair of 454,775 embedding_ref rows orphaned
by the §7.3 filtered/ tree retirement, followed by a full speaker.cluster
recompute, a tier.assign resume, and a fresh manifest.export). asr_agreement /
filters / g2p / tiers are now live-growing tables just like segments /
raw_files / labels_music / labels_lang / labels_overlap — see the
*_monotonic_growth tests below for all of them.
"""

import os

import pytest

from pipeline.catalog.catalog import connect_ro
from pipeline.config import CATALOG_PATH

# P0 import baseline for segments, before P3 S4's segment.vad_cut node started
# appending newly pipeline-cut segments (raw_id IS NOT NULL) — a floor, not an
# exact count.
SEGMENTS_P0_BASELINE = 455299

# Baseline for raw_files as of the 2026-07-04 downloaded.jsonl backfill — a
# floor, not an exact count. Was 6,272 before the backfill recovered ~4,648
# previously-unlogged files; future 02_download.py runs / further backfills
# should only ever grow this, never shrink it.
RAW_FILES_BASELINE = 10910

# P0 import baseline for labels_music, before P1's orchestrator started
# appending new rows — a floor, not an exact count.
LABELS_MUSIC_P0_BASELINE = 105668

# P0 import baseline for labels_lang / labels_overlap, before P2's label.suite
# node started filling the ~24-row legacy-import gap in each — a floor, not an
# exact count. Full corpus size (455299) is the ceiling both approach.
LABELS_LANG_OVERLAP_P0_BASELINE = 455275

# 2026-07-09 baseline: current row counts for the tables that are 1:1 with
# segments (asr_agreement/filters/tiers all become eligible once >= 2 ASR
# models have landed for a segment — see asr.agreement's docstring). Floors,
# not exact counts — new raw ingestion grows `segments` first, and these
# tables catch up a batch behind it, so `<= segments` is the ceiling to check
# instead of an exact match.
ONE_TO_ONE_TABLE_BASELINES = {
    "asr_agreement": 618695,
    "filters": 618695,
    "tiers": 618695,
}

# 2026-07-09 baseline for g2p: runs only on human-verified/agreement-passing
# text (a subset of asr_agreement, not all of segments) — a floor, not an
# exact count.
G2P_BASELINE = 484832


@pytest.fixture(scope="module")
def catalog_conn():
    if not os.path.exists(CATALOG_PATH):
        pytest.skip("catalog not built — run: python -m pipeline.cli catalog build")
    conn = connect_ro()
    yield conn
    conn.close()


def test_one_to_one_tables_monotonic_growth(catalog_conn):
    total_segments = catalog_conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    for table, baseline in ONE_TO_ONE_TABLE_BASELINES.items():
        total = catalog_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert total >= baseline, (
            f"{table} has {total} rows, below the 2026-07-09 baseline of "
            f"{baseline} — rows should only ever be added, never lost"
        )
        assert total <= total_segments, (
            f"{table} has {total} rows, more than segments ({total_segments})"
        )
        dupes = catalog_conn.execute(
            f"SELECT id, COUNT(*) FROM {table} GROUP BY id HAVING COUNT(*) != 1"
        ).fetchall()
        assert not dupes, f"duplicate ids in {table}: {dupes[:5]}"


def test_g2p_monotonic_growth(catalog_conn):
    total = catalog_conn.execute("SELECT COUNT(*) FROM g2p").fetchone()[0]
    assert total >= G2P_BASELINE, (
        f"g2p has {total} rows, below the 2026-07-09 baseline of "
        f"{G2P_BASELINE} — rows should only ever be added, never lost"
    )
    orphans = catalog_conn.execute(
        "SELECT COUNT(*) FROM g2p g LEFT JOIN asr_agreement a ON a.id = g.id WHERE a.id IS NULL"
    ).fetchone()[0]
    assert orphans == 0, f"{orphans} g2p row(s) have no matching asr_agreement row"
    dupes = catalog_conn.execute(
        "SELECT id, COUNT(*) FROM g2p GROUP BY id HAVING COUNT(*) != 1"
    ).fetchall()
    assert not dupes, f"duplicate ids in g2p: {dupes[:5]}"


def test_segments_monotonic_growth(catalog_conn):
    total = catalog_conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    assert total >= SEGMENTS_P0_BASELINE, (
        f"segments has {total} rows, below the P0 import baseline of "
        f"{SEGMENTS_P0_BASELINE} — rows should only ever be added, never lost"
    )
    dupes = catalog_conn.execute(
        "SELECT id, COUNT(*) FROM segments GROUP BY id HAVING COUNT(*) != 1"
    ).fetchall()
    assert not dupes, f"duplicate ids in segments: {dupes[:5]}"


def test_raw_files_monotonic_growth(catalog_conn):
    total = catalog_conn.execute("SELECT COUNT(*) FROM raw_files").fetchone()[0]
    assert total >= RAW_FILES_BASELINE, (
        f"raw_files has {total} rows, below the 2026-07-04 backfill baseline of "
        f"{RAW_FILES_BASELINE} — rows should only ever be added, never lost"
    )
    dupes = catalog_conn.execute(
        "SELECT raw_id, COUNT(*) FROM raw_files GROUP BY raw_id HAVING COUNT(*) != 1"
    ).fetchall()
    assert not dupes, f"duplicate raw_ids in raw_files: {dupes[:5]}"


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


def test_asr_results_at_least_two_architectures_per_segment(catalog_conn):
    """Originally exactly 2 rows/id (canto_ft + whisper_v3 at P0 import time).
    Since then two more independent ASR backends (qwen3_asr, sense_voice) went
    live, and canto_ft's `model` string has changed across sessions/machines
    (an absolute ct2_models path baked into the string) — so raw row count per
    id is no longer a fixed number (currently 4 or 5 in the live catalog). What
    must still hold is asr.agreement's own eligibility rule ("an id becomes
    eligible once >= 2 models have landed") — check the path-normalized
    distinct-architecture count per id, not the raw row count.
    """
    offenders = catalog_conn.execute(
        """
        SELECT id, COUNT(DISTINCT
            CASE WHEN model LIKE '%whisper-large-v2-cantonese%' THEN 'canto_ft' ELSE model END
        ) AS n_arch
        FROM asr_results GROUP BY id HAVING n_arch < 2
        """
    ).fetchall()
    if offenders:
        preview = offenders[:5]
        preview_str = "\n".join(
            f"  id={row[0]!r}, n_arch={row[1]}" for row in preview
        )
        pytest.fail(
            f"{len(offenders)} segment id(s) in asr_results have fewer than 2 "
            f"distinct ASR architectures. First up to 5 offenders:\n{preview_str}"
        )


def test_tiers_valid_values(catalog_conn):
    """tiers row count is covered by test_one_to_one_tables_monotonic_growth
    above (it grows with the corpus, no longer a fixed count as of the
    2026-07-09 tier.assign resume that took it from 245,500 to all 618,695
    segments) -- this test just checks every row's tier value is one of the
    three valid verification-confidence tiers (gold/silver/excluded; note
    'gold' currently has zero rows in the live catalog since no human-
    calibration node is wired into the DAG yet — see CLAUDE.md's ASR
    strategy note — this is expected, not a bug)."""
    bad_tiers = catalog_conn.execute(
        "SELECT DISTINCT tier FROM tiers WHERE tier NOT IN ('gold', 'silver', 'excluded')"
    ).fetchall()
    assert not bad_tiers, f"unexpected tier value(s): {bad_tiers}"


def test_manifest_build_matches_expected_corpus_totals(catalog_conn):
    """P4 gate: manifest.build()'s catalog join must reproduce the exact totals the
    current on-disk metadata/manifest.jsonl was built with. Baseline updated
    2026-07-09 after the Phase A repair chain (speaker.embed --verify-existing
    repair of 454,775 orphaned embeddings -> speaker.cluster full recompute ->
    tier.assign resume -> manifest.export): 369,700 entries / 8,160 speakers /
    gold=0 / silver=369,700 -- fewer entries than the pre-repair chain's
    455,299 because manifest.build additionally gates on filters.pass and
    g2p.valid_fraction (independent eligibility signals), and gold=0 because
    no human-calibration node is wired into the DAG (see test_tiers_valid_values
    above). A mismatch here means the eligibility join (filters.pass,
    g2p.valid_fraction, tier IN (gold,silver)) regressed -- update this baseline
    only after an intentional, verified manifest.export re-run."""
    from pipeline.nodes.manifest import run_manifest_build

    result = run_manifest_build()
    assert result["count"] == 369700
    assert result["n_speakers"] == 8160
    assert result["tier_counts"].get("gold", 0) == 0
    assert result["tier_counts"].get("silver") == 369700
