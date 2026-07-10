import asyncio

import duckdb
import pytest

from pipeline.catalog.catalog import init_schema
from pipeline.nodes.calibrate import (
    discover,
    next_pending,
    queue_stats,
    record_decision,
    run_calibrate_sample,
)


@pytest.fixture
def scratch_conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "scratch.duckdb"))
    init_schema(conn)
    yield conn
    conn.close()


def _seed_segment(conn, seg_id, *, passes_filter=True, agreement=0.7, source="podcast"):
    conn.execute(
        "INSERT INTO segments (id, audio_path, source, duration_sec, program) "
        "VALUES (?, ?, ?, 6.0, 'test-program')",
        [seg_id, f"/tmp/{seg_id}.flac", source],
    )
    conn.execute(
        "INSERT INTO asr_agreement (id, agreement, best_text, text_verified) VALUES (?, ?, ?, FALSE)",
        [seg_id, agreement, "呢個係測試文字"],
    )
    conn.execute(
        "INSERT INTO filters (id, pass) VALUES (?, ?)",
        [seg_id, passes_filter],
    )
    conn.execute(
        "INSERT INTO asr_results (id, model, text, confidence) VALUES (?, 'canto_ft', '呢個係測試文字', 0.9)",
        [seg_id],
    )


# ---------------------------------------------------------------------------
# discover() / run_calibrate_sample() — sample selection + queuing
# ---------------------------------------------------------------------------

def test_discover_excludes_filter_failing_segments(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "pass1", passes_filter=True)
    _seed_segment(conn, "fail1", passes_filter=False)

    ids = discover(conn, 10)
    assert ids == ["pass1"]


def test_discover_excludes_already_queued_segments(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "s1")
    conn.execute(
        "INSERT INTO calibration_review (id, decision) VALUES ('s1', 'pending')"
    )
    assert discover(conn, 10) == []


def test_run_calibrate_sample_queues_pending_rows(scratch_conn):
    conn = scratch_conn
    for i in range(5):
        _seed_segment(conn, f"s{i}")

    result = asyncio.run(run_calibrate_sample(conn=conn, n=3))

    assert result["queued"] == 3
    rows = conn.execute(
        "SELECT decision, sample_batch FROM calibration_review"
    ).fetchall()
    assert len(rows) == 3
    assert all(decision == "pending" for decision, _ in rows)
    assert all(batch == result["run_id"] for _, batch in rows)


def test_run_calibrate_sample_empty_when_nothing_eligible(scratch_conn):
    result = asyncio.run(run_calibrate_sample(conn=scratch_conn, n=10))
    assert result == {"queued": 0, "run_id": None}


# ---------------------------------------------------------------------------
# record_decision() — human review write-back
# ---------------------------------------------------------------------------

def test_record_decision_verified_flips_text_verified_and_gold_tier(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "s1")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('s1', 'pending')")
    conn.execute("INSERT INTO tiers (id, tier, provenance) VALUES ('s1', 'silver', 'tier_assign')")

    record_decision(conn, "s1", "verified", "校正之後嘅文字")

    review = conn.execute(
        "SELECT decision, reviewed_text FROM calibration_review WHERE id='s1'"
    ).fetchone()
    assert review == ("verified", "校正之後嘅文字")

    agreement_row = conn.execute(
        "SELECT text_verified, best_text FROM asr_agreement WHERE id='s1'"
    ).fetchone()
    assert agreement_row == (True, "校正之後嘅文字")

    tier_row = conn.execute("SELECT tier, provenance FROM tiers WHERE id='s1'").fetchone()
    assert tier_row == ("gold", "calibrate_verify")


def test_record_decision_skipped_does_not_touch_agreement_or_tier(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "s1")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('s1', 'pending')")
    conn.execute("INSERT INTO tiers (id, tier, provenance) VALUES ('s1', 'silver', 'tier_assign')")

    record_decision(conn, "s1", "skipped", None)

    agreement_row = conn.execute(
        "SELECT text_verified FROM asr_agreement WHERE id='s1'"
    ).fetchone()
    assert agreement_row == (False,)
    tier_row = conn.execute("SELECT tier FROM tiers WHERE id='s1'").fetchone()
    assert tier_row == ("silver",)


def test_record_decision_rejects_invalid_decision(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "s1")
    with pytest.raises(ValueError):
        record_decision(conn, "s1", "maybe", None)


# ---------------------------------------------------------------------------
# next_pending() / queue_stats() — browser UI read path
# ---------------------------------------------------------------------------

def test_next_pending_returns_segment_with_candidates(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "s1", agreement=0.72)
    conn.execute("INSERT INTO calibration_review (id, decision, queued_at) VALUES ('s1', 'pending', '2026-07-10 00:00:00')")

    item = next_pending(conn)

    assert item["id"] == "s1"
    assert item["agreement"] == 0.72
    assert item["candidates"] == [{"model": "canto_ft", "text": "呢個係測試文字", "confidence": 0.9}]


def test_next_pending_skips_non_pending(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "s1")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('s1', 'verified')")
    assert next_pending(conn) is None


def test_next_pending_scoped_to_sample_batch(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "a")
    _seed_segment(conn, "b")
    conn.execute("INSERT INTO calibration_review (id, decision, sample_batch, queued_at) VALUES ('a', 'pending', 'batch1', '2026-07-10 00:00:00')")
    conn.execute("INSERT INTO calibration_review (id, decision, sample_batch, queued_at) VALUES ('b', 'pending', 'batch2', '2026-07-10 00:00:01')")

    item = next_pending(conn, sample_batch="batch2")
    assert item["id"] == "b"


def test_queue_stats_counts_by_decision(scratch_conn):
    conn = scratch_conn
    for seg_id, decision in [("a", "pending"), ("b", "verified"), ("c", "verified"), ("d", "skipped")]:
        _seed_segment(conn, seg_id)
        conn.execute(
            "INSERT INTO calibration_review (id, decision) VALUES (?, ?)", [seg_id, decision]
        )

    stats = queue_stats(conn)
    assert stats == {"pending": 1, "verified": 2, "skipped": 1, "rejected": 0, "total": 4}
