import asyncio

import duckdb
import pytest

from pipeline.catalog.catalog import init_schema
from pipeline.nodes.calibrate import (
    _levenshtein,
    discover,
    get_item,
    jyutping_preview,
    list_batches,
    list_history,
    list_sources,
    next_pending,
    queue_stats,
    record_decision,
    run_calibrate_sample,
    summary_stats,
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

    picked = discover(conn, 10)
    assert [seg_id for seg_id, _ in picked] == ["pass1"]


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


def test_run_calibrate_sample_snapshots_original_best_text(scratch_conn):
    """Regression guard: 'verified' decisions overwrite asr_agreement.best_text
    in place, so summary_stats' edit-distance metric needs its own snapshot
    taken at queue time, before any correction happens."""
    conn = scratch_conn
    _seed_segment(conn, "s0")

    asyncio.run(run_calibrate_sample(conn=conn, n=1))

    row = conn.execute(
        "SELECT original_best_text FROM calibration_review WHERE id='s0'"
    ).fetchone()
    assert row == ("呢個係測試文字",)


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
    assert stats == {"pending": 1, "verified": 2, "skipped": 1, "rejected": 0, "flagged": 0, "total": 4}


def test_queue_stats_scoped_by_source(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "a", source="rthk")
    _seed_segment(conn, "b", source="podcast")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('a', 'verified')")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('b', 'skipped')")

    assert queue_stats(conn, source="rthk") == {
        "pending": 0, "verified": 1, "skipped": 0, "rejected": 0, "flagged": 0, "total": 1,
    }
    assert queue_stats(conn, source="podcast")["skipped"] == 1


# ---------------------------------------------------------------------------
# next_pending() ordering / source filter
# ---------------------------------------------------------------------------

def test_next_pending_order_agreement_asc(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "high", agreement=0.9)
    _seed_segment(conn, "low", agreement=0.3)
    conn.execute("INSERT INTO calibration_review (id, decision, queued_at) VALUES ('high', 'pending', '2026-07-10 00:00:00')")
    conn.execute("INSERT INTO calibration_review (id, decision, queued_at) VALUES ('low', 'pending', '2026-07-10 00:00:01')")

    item = next_pending(conn, order="agreement_asc")
    assert item["id"] == "low"

    item = next_pending(conn, order="agreement_desc")
    assert item["id"] == "high"


def test_next_pending_filters_by_source(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "a", source="rthk")
    _seed_segment(conn, "b", source="podcast")
    conn.execute("INSERT INTO calibration_review (id, decision, queued_at) VALUES ('a', 'pending', '2026-07-10 00:00:00')")
    conn.execute("INSERT INTO calibration_review (id, decision, queued_at) VALUES ('b', 'pending', '2026-07-10 00:00:01')")

    item = next_pending(conn, source="podcast")
    assert item["id"] == "b"


# ---------------------------------------------------------------------------
# get_item() — reopen any item (pending or decided) by id
# ---------------------------------------------------------------------------

def test_get_item_returns_decided_item_with_reviewed_text(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "s1")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('s1', 'pending')")
    record_decision(conn, "s1", "verified", "已經校正嘅文字")

    item = get_item(conn, "s1")
    assert item["decision"] == "verified"
    assert item["reviewed_text"] == "已經校正嘅文字"
    assert item["candidates"]


def test_get_item_unknown_id_returns_none(scratch_conn):
    assert get_item(scratch_conn, "nope") is None


# ---------------------------------------------------------------------------
# list_history() / list_batches() / list_sources()
# ---------------------------------------------------------------------------

def test_list_history_excludes_pending_orders_newest_first(scratch_conn):
    conn = scratch_conn
    for seg_id in ("a", "b", "c"):
        _seed_segment(conn, seg_id)
    conn.execute("INSERT INTO calibration_review (id, decision, reviewed_at) VALUES ('a', 'verified', '2026-07-10 00:00:01')")
    conn.execute("INSERT INTO calibration_review (id, decision, reviewed_at) VALUES ('b', 'pending', NULL)")
    conn.execute("INSERT INTO calibration_review (id, decision, reviewed_at) VALUES ('c', 'skipped', '2026-07-10 00:00:02')")

    items = list_history(conn)
    assert [i["id"] for i in items] == ["c", "a"]


def test_list_batches_reports_per_batch_stats(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "a")
    _seed_segment(conn, "b")
    conn.execute("INSERT INTO calibration_review (id, decision, sample_batch) VALUES ('a', 'pending', 'batch1')")
    conn.execute("INSERT INTO calibration_review (id, decision, sample_batch) VALUES ('b', 'verified', 'batch2')")

    batches = {b["sample_batch"]: b for b in list_batches(conn)}
    assert batches["batch1"]["pending"] == 1
    assert batches["batch2"]["verified"] == 1


def test_list_sources_returns_distinct_sources(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "a", source="rthk")
    _seed_segment(conn, "b", source="podcast")
    _seed_segment(conn, "c", source="rthk")

    assert list_sources(conn) == ["podcast", "rthk"]


# ---------------------------------------------------------------------------
# record_decision('flagged', ...) -- pipeline-bug reports, no gold side-effect
# ---------------------------------------------------------------------------

def test_record_decision_flagged_stores_reason_without_touching_gold(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "s1")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('s1', 'pending')")
    conn.execute("INSERT INTO tiers (id, tier, provenance) VALUES ('s1', 'silver', 'tier_assign')")

    record_decision(conn, "s1", "flagged", None, flag_reason="segment 切錯咗,含兩個speaker")

    review = conn.execute(
        "SELECT decision, flag_reason FROM calibration_review WHERE id='s1'"
    ).fetchone()
    assert review == ("flagged", "segment 切錯咗,含兩個speaker")

    # 'flagged' must NOT touch text_verified or tiers (unlike 'verified')
    agreement_row = conn.execute("SELECT text_verified FROM asr_agreement WHERE id='s1'").fetchone()
    assert agreement_row == (False,)
    tier_row = conn.execute("SELECT tier FROM tiers WHERE id='s1'").fetchone()
    assert tier_row == ("silver",)


# ---------------------------------------------------------------------------
# jyutping_preview() -- reuses g2p.py's real conversion/validation
# ---------------------------------------------------------------------------

def test_jyutping_preview_valid_cantonese_text():
    result = jyutping_preview("心臟病中風")
    assert result["accept"] is True
    assert result["valid_fraction"] == 1.0
    assert result["bad_tokens"] == []
    assert "sam1" in result["jyutping"]


def test_jyutping_preview_empty_text():
    result = jyutping_preview("")
    assert result == {"jyutping": "", "valid_fraction": 1.0, "accept": True, "bad_tokens": []}


# ---------------------------------------------------------------------------
# _levenshtein() / summary_stats()
# ---------------------------------------------------------------------------

def test_levenshtein_basic_cases():
    assert _levenshtein("", "") == 0
    assert _levenshtein("abc", "abc") == 0
    assert _levenshtein("", "abc") == 3
    assert _levenshtein("abc", "") == 3
    assert _levenshtein("心臟病", "心病") == 1


def test_summary_stats_aggregates_decisions_and_edit_distance(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "a", source="rthk", agreement=0.9)
    _seed_segment(conn, "b", source="podcast", agreement=0.5)
    conn.execute("INSERT INTO calibration_review (id, decision, original_best_text) VALUES ('a', 'pending', '呢個係測試文字')")
    conn.execute("INSERT INTO calibration_review (id, decision, original_best_text) VALUES ('b', 'pending', '呢個係測試文字')")
    record_decision(conn, "a", "verified", "呢個係校正咗嘅文字")   # differs from seeded best_text
    record_decision(conn, "b", "flagged", None, flag_reason="audio corrupt")

    stats = summary_stats(conn)
    assert stats["decision_counts"]["verified"] == 1
    assert stats["decision_counts"]["flagged"] == 1
    assert stats["by_source"]["rthk"]["verified"] == 1
    assert stats["by_source"]["podcast"]["flagged"] == 1
    assert stats["verified_edit_sample_size"] == 1
    assert stats["avg_edit_distance_verified"] > 0
    assert stats["top_flag_reasons"] == [{"reason": "audio corrupt", "count": 1}]


def test_summary_stats_scoped_by_batch(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "a")
    _seed_segment(conn, "b")
    conn.execute("INSERT INTO calibration_review (id, decision, sample_batch) VALUES ('a', 'verified', 'batch1')")
    conn.execute("INSERT INTO calibration_review (id, decision, sample_batch) VALUES ('b', 'verified', 'batch2')")

    stats = summary_stats(conn, sample_batch="batch1")
    assert stats["decision_counts"]["total"] == 1
