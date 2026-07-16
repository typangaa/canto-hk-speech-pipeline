import asyncio

import duckdb
import pytest

from pipeline.catalog.catalog import init_schema, upsert_rows
from pipeline.nodes.quality_tier import (
    B_DNSMOS_MIN,
    B_MUSIC_MAX,
    B_OVERLAP_MAX,
    assign_quality_tier,
    discover,
    run_quality_tier_assign,
)


@pytest.fixture
def scratch_conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "scratch.duckdb"))
    init_schema(conn)
    yield conn
    conn.close()


def _seed_segment(
    conn,
    seg_id,
    *,
    passes_filter=True,
    tier="auto_gold",
    dnsmos=4.0,
    music_prob=None,
    overlap_ratio=None,
):
    conn.execute(
        "INSERT INTO segments (id, audio_path, source, duration_sec, program) "
        "VALUES (?, ?, 'podcast', 6.0, 'test-program')",
        [seg_id, f"/tmp/{seg_id}.flac"],
    )
    conn.execute(
        "INSERT INTO filters (id, pass, dnsmos) VALUES (?, ?, ?)",
        [seg_id, passes_filter, dnsmos],
    )
    if tier is not None:
        conn.execute(
            "INSERT INTO tiers (id, tier, provenance) VALUES (?, ?, 'tier_assign')",
            [seg_id, tier],
        )
    if music_prob is not None:
        conn.execute(
            "INSERT INTO labels_music (id, music_prob) VALUES (?, ?)", [seg_id, music_prob]
        )
    if overlap_ratio is not None:
        conn.execute(
            "INSERT INTO labels_overlap (id, overlap_ratio) VALUES (?, ?)",
            [seg_id, overlap_ratio],
        )


# ---------------------------------------------------------------------------
# assign_quality_tier() -- pure logic
# ---------------------------------------------------------------------------

def test_assign_quality_tier_all_gates_pass_is_b():
    assert assign_quality_tier(B_DNSMOS_MIN, B_MUSIC_MAX - 0.01, B_OVERLAP_MAX - 0.01) == "B"


def test_assign_quality_tier_boundary_dnsmos_exact_threshold_inclusive():
    assert assign_quality_tier(B_DNSMOS_MIN, 0.0, 0.0) == "B"
    assert assign_quality_tier(B_DNSMOS_MIN - 0.01, 0.0, 0.0) == "A"


def test_assign_quality_tier_boundary_music_prob_exclusive():
    """music_prob < B_MUSIC_MAX is strict -- exactly at the max fails."""
    assert assign_quality_tier(5.0, B_MUSIC_MAX, 0.0) == "A"
    assert assign_quality_tier(5.0, B_MUSIC_MAX - 0.001, 0.0) == "B"


def test_assign_quality_tier_boundary_overlap_ratio_exclusive():
    assert assign_quality_tier(5.0, 0.0, B_OVERLAP_MAX) == "A"
    assert assign_quality_tier(5.0, 0.0, B_OVERLAP_MAX - 0.001) == "B"


def test_assign_quality_tier_missing_dnsmos_fails_closed_to_a():
    assert assign_quality_tier(None, 0.0, 0.0) == "A"


def test_assign_quality_tier_missing_music_prob_fails_closed_to_a():
    assert assign_quality_tier(5.0, None, 0.0) == "A"


def test_assign_quality_tier_missing_overlap_ratio_fails_closed_to_a():
    assert assign_quality_tier(5.0, 0.0, None) == "A"


def test_assign_quality_tier_high_music_prob_is_a():
    assert assign_quality_tier(5.0, 0.5, 0.0) == "A"


def test_assign_quality_tier_high_overlap_ratio_is_a():
    assert assign_quality_tier(5.0, 0.0, 0.5) == "A"


# ---------------------------------------------------------------------------
# discover() -- scope: gold/auto_gold + filters.pass=TRUE only
# ---------------------------------------------------------------------------

def test_discover_excludes_silver_and_bronze(scratch_conn):
    _seed_segment(scratch_conn, "ag1", tier="auto_gold")
    _seed_segment(scratch_conn, "sv1", tier="silver")
    _seed_segment(scratch_conn, "br1", tier="bronze")

    ids = {row[0] for row in discover(scratch_conn)}
    assert ids == {"ag1"}


def test_discover_includes_gold_and_auto_gold(scratch_conn):
    _seed_segment(scratch_conn, "g1", tier="gold")
    _seed_segment(scratch_conn, "ag1", tier="auto_gold")

    ids = {row[0] for row in discover(scratch_conn)}
    assert ids == {"g1", "ag1"}


def test_discover_excludes_filter_failing_segments(scratch_conn):
    _seed_segment(scratch_conn, "pass1", passes_filter=True)
    _seed_segment(scratch_conn, "fail1", passes_filter=False)

    ids = {row[0] for row in discover(scratch_conn)}
    assert ids == {"pass1"}


def test_discover_excludes_already_quality_tiered_segments(scratch_conn):
    _seed_segment(scratch_conn, "s1")
    upsert_rows(
        scratch_conn, "quality_tiers",
        [{"id": "s1", "quality_tier": "A", "provenance": "quality_tier_assign"}],
        ["id"],
    )

    assert discover(scratch_conn) == []


def test_discover_left_joins_missing_labels_as_none(scratch_conn):
    _seed_segment(scratch_conn, "nolab", dnsmos=4.0)  # no music/overlap rows

    rows = discover(scratch_conn)
    assert len(rows) == 1
    seg_id, dnsmos, music_prob, overlap_ratio = rows[0]
    assert seg_id == "nolab"
    assert dnsmos == 4.0
    assert music_prob is None
    assert overlap_ratio is None


# ---------------------------------------------------------------------------
# run_quality_tier_assign() -- end to end
# ---------------------------------------------------------------------------

def test_run_quality_tier_assign_writes_a_and_b(scratch_conn):
    _seed_segment(scratch_conn, "clean1", tier="gold", dnsmos=4.0, music_prob=0.02, overlap_ratio=0.0)
    _seed_segment(scratch_conn, "noisy1", tier="auto_gold", dnsmos=3.6, music_prob=0.3, overlap_ratio=0.0)

    result = asyncio.run(run_quality_tier_assign(conn=scratch_conn))

    assert result["processed"] == 2
    assert result["tier_a"] == 1
    assert result["tier_b"] == 1
    rows = dict(scratch_conn.execute("SELECT id, quality_tier FROM quality_tiers").fetchall())
    assert rows == {"clean1": "B", "noisy1": "A"}


def test_run_quality_tier_assign_is_idempotent(scratch_conn):
    _seed_segment(scratch_conn, "s1")
    asyncio.run(run_quality_tier_assign(conn=scratch_conn))
    result = asyncio.run(run_quality_tier_assign(conn=scratch_conn))
    assert result["processed"] == 0


def test_run_quality_tier_assign_respects_limit(scratch_conn):
    for i in range(5):
        _seed_segment(scratch_conn, f"s{i}")
    result = asyncio.run(run_quality_tier_assign(conn=scratch_conn, limit=2))
    assert result["processed"] == 2


def test_run_quality_tier_assign_empty_pool_noop(scratch_conn):
    result = asyncio.run(run_quality_tier_assign(conn=scratch_conn))
    assert result == {"processed": 0, "tier_a": 0, "tier_b": 0, "errors": 0}


def test_run_quality_tier_assign_never_calls_connect_when_conn_passed(scratch_conn, monkeypatch):
    import pipeline.catalog.catalog as catalog_module

    def _boom(*args, **kwargs):
        raise AssertionError("connect() must not be called when conn is passed")

    monkeypatch.setattr(catalog_module, "connect", _boom)
    _seed_segment(scratch_conn, "s1")
    asyncio.run(run_quality_tier_assign(conn=scratch_conn))
