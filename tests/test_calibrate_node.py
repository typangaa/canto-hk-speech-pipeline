import asyncio
import json

import duckdb
import pytest

from pipeline.catalog.catalog import init_schema
from pipeline.nodes.calibrate import (
    CODE_SWITCH_QA_MULTIPLIER,
    MANDARIN_FLAG_REASON,
    NOT_SINGLE_SPEAKER_FLAG_REASON,
    QA_SAMPLE_RATE_BY_TIER,
    WRONG_SPEAKER_ID_FLAG_REASON,
    _levenshtein,
    append_pending_decision,
    discover,
    get_item,
    jyutping_preview,
    list_batches,
    list_history,
    list_sources,
    load_pending_decisions,
    next_pending,
    pending_queue_rows,
    progress_report,
    queue_stats,
    record_decision,
    recommended_sample_n,
    run_calibrate_export_snapshot,
    run_calibrate_flush_pending,
    run_calibrate_sample,
    summary_stats,
)


@pytest.fixture
def scratch_conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "scratch.duckdb"))
    init_schema(conn)
    yield conn
    conn.close()


def _seed_segment(conn, seg_id, *, passes_filter=True, agreement=0.7, source="podcast", tier=None, english_ratio=0.0):
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
        "INSERT INTO filters (id, pass, english_ratio) VALUES (?, ?, ?)",
        [seg_id, passes_filter, english_ratio],
    )
    conn.execute(
        "INSERT INTO asr_results (id, model, text, confidence) VALUES (?, 'canto_ft', '呢個係測試文字', 0.9)",
        [seg_id],
    )
    if tier is not None:
        conn.execute(
            "INSERT INTO tiers (id, tier, provenance) VALUES (?, ?, 'tier_assign')",
            [seg_id, tier],
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


def test_discover_scoped_to_tier(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "ag1", tier="auto_gold")
    _seed_segment(conn, "sv1", tier="silver")
    _seed_segment(conn, "br1", tier="bronze")

    picked = discover(conn, 10, tier="auto_gold")
    assert [seg_id for seg_id, _ in picked] == ["ag1"]

    picked = discover(conn, 10, tier="bronze")
    assert [seg_id for seg_id, _ in picked] == ["br1"]


def test_discover_scoped_to_min_agreement(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "low", agreement=0.72)
    _seed_segment(conn, "high", agreement=0.97)

    picked = discover(conn, 10, min_agreement=0.95)
    assert [seg_id for seg_id, _ in picked] == ["high"]


def test_discover_tier_and_min_agreement_combine(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "br_low", agreement=0.72, tier="bronze")
    _seed_segment(conn, "sv_high", agreement=0.90, tier="silver")

    picked = discover(conn, 10, tier="silver", min_agreement=0.85)
    assert [seg_id for seg_id, _ in picked] == ["sv_high"]


def test_recommended_sample_n_scales_with_population_and_rate(scratch_conn):
    conn = scratch_conn
    for i in range(1000):
        _seed_segment(conn, f"br{i}", tier="bronze")

    n = recommended_sample_n(conn, "bronze")
    assert n == round(1000 * QA_SAMPLE_RATE_BY_TIER["bronze"])


def test_recommended_sample_n_floors_at_min_n_for_small_populations(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "ag1", tier="auto_gold")

    assert recommended_sample_n(conn, "auto_gold", min_n=50) == 50


def test_recommended_sample_n_rejects_unknown_tier(scratch_conn):
    with pytest.raises(ValueError):
        recommended_sample_n(scratch_conn, "excluded")


# ---------------------------------------------------------------------------
# code_switch oversampling (added 2026-07-15, T18)
# ---------------------------------------------------------------------------

def test_discover_code_switch_only_scopes_to_english_ratio_positive(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "mono1", english_ratio=0.0)
    _seed_segment(conn, "cs1", english_ratio=0.2)

    picked = discover(conn, 10, code_switch="only")
    assert [seg_id for seg_id, _ in picked] == ["cs1"]


def test_discover_code_switch_exclude_scopes_to_english_ratio_zero(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "mono1", english_ratio=0.0)
    _seed_segment(conn, "cs1", english_ratio=0.2)

    picked = discover(conn, 10, code_switch="exclude")
    assert [seg_id for seg_id, _ in picked] == ["mono1"]


def test_discover_code_switch_none_returns_both(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "mono1", english_ratio=0.0)
    _seed_segment(conn, "cs1", english_ratio=0.2)

    picked = {seg_id for seg_id, _ in discover(conn, 10)}
    assert picked == {"mono1", "cs1"}


def test_discover_code_switch_combines_with_tier(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "ag_cs", tier="auto_gold", english_ratio=0.2)
    _seed_segment(conn, "sv_cs", tier="silver", english_ratio=0.2)
    _seed_segment(conn, "ag_mono", tier="auto_gold", english_ratio=0.0)

    picked = discover(conn, 10, tier="auto_gold", code_switch="only")
    assert [seg_id for seg_id, _ in picked] == ["ag_cs"]


def test_discover_rejects_invalid_code_switch(scratch_conn):
    with pytest.raises(ValueError):
        discover(scratch_conn, 10, code_switch="bogus")


def test_recommended_sample_n_code_switch_applies_multiplier(scratch_conn):
    conn = scratch_conn
    for i in range(1000):
        _seed_segment(conn, f"br_cs{i}", tier="bronze", english_ratio=0.2)

    n = recommended_sample_n(conn, "bronze", code_switch=True)
    expected_rate = min(QA_SAMPLE_RATE_BY_TIER["bronze"] * CODE_SWITCH_QA_MULTIPLIER, 1.0)
    assert n == round(1000 * expected_rate)
    assert n > round(1000 * QA_SAMPLE_RATE_BY_TIER["bronze"])


def test_recommended_sample_n_code_switch_scopes_population_to_code_switch_only(scratch_conn):
    conn = scratch_conn
    for i in range(100):
        _seed_segment(conn, f"ag_mono{i}", tier="auto_gold", english_ratio=0.0)
    for i in range(10):
        _seed_segment(conn, f"ag_cs{i}", tier="auto_gold", english_ratio=0.2)

    n = recommended_sample_n(conn, "auto_gold", min_n=1, code_switch=True)
    expected_rate = min(QA_SAMPLE_RATE_BY_TIER["auto_gold"] * CODE_SWITCH_QA_MULTIPLIER, 1.0)
    assert n == max(1, round(10 * expected_rate))


def test_recommended_sample_n_code_switch_rate_capped_at_one(scratch_conn):
    """auto_gold base rate (0.015) * 10x = 0.15, well under 1.0 -- but bronze's base
    rate (0.10) * 10x = 1.0 exactly, the boundary the cap must not exceed."""
    conn = scratch_conn
    for i in range(20):
        _seed_segment(conn, f"br_cs{i}", tier="bronze", english_ratio=0.2)

    n = recommended_sample_n(conn, "bronze", min_n=1, code_switch=True)
    assert n == 20  # 100% of the 20-row population, not more


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


def test_record_decision_rejected_excludes_from_tier(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "s1")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('s1', 'pending')")
    conn.execute("INSERT INTO tiers (id, tier, provenance) VALUES ('s1', 'silver', 'tier_assign')")

    record_decision(conn, "s1", "rejected", None)

    review = conn.execute("SELECT decision FROM calibration_review WHERE id='s1'").fetchone()
    assert review == ("rejected",)
    tier_row = conn.execute("SELECT tier, provenance FROM tiers WHERE id='s1'").fetchone()
    assert tier_row == ("excluded", "calibrate_reject")
    # rejection must never masquerade as human text verification
    agreement_row = conn.execute("SELECT text_verified FROM asr_agreement WHERE id='s1'").fetchone()
    assert agreement_row == (False,)


def test_record_decision_rejected_with_mandarin_reason_excludes_and_stores_reason(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "s1")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('s1', 'pending')")

    record_decision(conn, "s1", "rejected", None, flag_reason=MANDARIN_FLAG_REASON)

    review = conn.execute(
        "SELECT decision, flag_reason FROM calibration_review WHERE id='s1'"
    ).fetchone()
    assert review == ("rejected", "mandarin")
    tier_row = conn.execute("SELECT tier FROM tiers WHERE id='s1'").fetchone()
    assert tier_row == ("excluded",)


# ---------------------------------------------------------------------------
# T9 speaker-purity buttons (2026-07-17) — two different failure modes,
# two different consequences, both riding record_decision's existing
# 'rejected'/'flagged' mechanics (no new decision type needed).
# ---------------------------------------------------------------------------

def test_record_decision_not_single_speaker_excludes_and_stores_reason(scratch_conn):
    """The 'Multi-speaker' button -- a real audio defect (Hard Constraint #5
    violation) -- must exclude the segment, same mechanism as Mandarin."""
    conn = scratch_conn
    _seed_segment(conn, "s1")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('s1', 'pending')")

    record_decision(conn, "s1", "rejected", None, flag_reason=NOT_SINGLE_SPEAKER_FLAG_REASON)

    review = conn.execute(
        "SELECT decision, flag_reason FROM calibration_review WHERE id='s1'"
    ).fetchone()
    assert review == ("rejected", "not_single_speaker")
    tier_row = conn.execute("SELECT tier FROM tiers WHERE id='s1'").fetchone()
    assert tier_row == ("excluded",)


def test_record_decision_wrong_speaker_id_does_not_exclude(scratch_conn):
    """The 'Wrong speaker ID' button -- a harmless metadata mislabel, audio is
    fine -- must NOT exclude the segment and must NOT touch tiers/asr_agreement,
    same as the generic 'flagged' decision."""
    conn = scratch_conn
    _seed_segment(conn, "s1")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('s1', 'pending')")
    conn.execute("INSERT INTO tiers (id, tier, provenance) VALUES ('s1', 'silver', 'tier_assign')")

    record_decision(conn, "s1", "flagged", None, flag_reason=WRONG_SPEAKER_ID_FLAG_REASON)

    review = conn.execute(
        "SELECT decision, flag_reason FROM calibration_review WHERE id='s1'"
    ).fetchone()
    assert review == ("flagged", "wrong_speaker_id")
    tier_row = conn.execute("SELECT tier, provenance FROM tiers WHERE id='s1'").fetchone()
    assert tier_row == ("silver", "tier_assign")  # untouched
    agreement_row = conn.execute("SELECT text_verified FROM asr_agreement WHERE id='s1'").fetchone()
    assert agreement_row == (False,)  # untouched


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


def test_summary_stats_includes_mandarin_rejections_in_flag_leaderboard(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "a", source="youtube")
    _seed_segment(conn, "b", source="youtube")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('a', 'pending')")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('b', 'pending')")
    record_decision(conn, "a", "rejected", None, flag_reason=MANDARIN_FLAG_REASON)
    record_decision(conn, "b", "rejected", None)  # plain rejection, no reason

    stats = summary_stats(conn)
    assert stats["decision_counts"]["rejected"] == 2
    assert stats["top_flag_reasons"] == [{"reason": "mandarin", "count": 1}]


def test_summary_stats_scoped_by_batch(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "a")
    _seed_segment(conn, "b")
    conn.execute("INSERT INTO calibration_review (id, decision, sample_batch) VALUES ('a', 'verified', 'batch1')")
    conn.execute("INSERT INTO calibration_review (id, decision, sample_batch) VALUES ('b', 'verified', 'batch2')")

    stats = summary_stats(conn, sample_batch="batch1")
    assert stats["decision_counts"]["total"] == 1


# ---------------------------------------------------------------------------
# Offline-review support (2026-07-13): next_pending exclude_ids,
# pending_queue_rows / export_snapshot, append_pending_decision /
# load_pending_decisions / flush_pending
# ---------------------------------------------------------------------------

def test_next_pending_exclude_ids_skips_locally_decided(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "a")
    _seed_segment(conn, "b")
    conn.execute("INSERT INTO calibration_review (id, decision, queued_at) VALUES ('a', 'pending', '2026-07-13 00:00:00')")
    conn.execute("INSERT INTO calibration_review (id, decision, queued_at) VALUES ('b', 'pending', '2026-07-13 00:00:01')")

    item = next_pending(conn, exclude_ids={"a"})
    assert item["id"] == "b"


def test_next_pending_exclude_ids_empty_queue_returns_none(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "a")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('a', 'pending')")
    assert next_pending(conn, exclude_ids={"a"}) is None


def test_pending_queue_rows_returns_all_pending_with_batch(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "a")
    _seed_segment(conn, "b")
    conn.execute("INSERT INTO calibration_review (id, decision, sample_batch, queued_at) VALUES ('a', 'pending', 'batch1', '2026-07-13 00:00:00')")
    conn.execute("INSERT INTO calibration_review (id, decision, sample_batch, queued_at) VALUES ('b', 'verified', 'batch1', '2026-07-13 00:00:01')")

    rows = pending_queue_rows(conn)
    assert [r["id"] for r in rows] == ["a"]
    assert rows[0]["sample_batch"] == "batch1"
    assert rows[0]["candidates"] == [{"model": "canto_ft", "text": "呢個係測試文字", "confidence": 0.9}]


def test_run_calibrate_export_snapshot_writes_json(scratch_conn, tmp_path):
    conn = scratch_conn
    _seed_segment(conn, "a")
    conn.execute("INSERT INTO calibration_review (id, decision, sample_batch, queued_at) VALUES ('a', 'pending', 'batch1', '2026-07-13 00:00:00')")
    out_path = tmp_path / "snapshot.json"

    result = asyncio.run(run_calibrate_export_snapshot(conn=conn, out_path=out_path))

    assert result["exported"] == 1
    snapshot = json.loads(out_path.read_text())
    assert "snapshot_at" in snapshot
    assert snapshot["items"][0]["id"] == "a"


def test_append_pending_decision_rejects_invalid_decision(tmp_path):
    with pytest.raises(ValueError):
        append_pending_decision("s1", "bogus", "text", None, path=tmp_path / "pending.jsonl")


def test_append_and_load_pending_decisions_roundtrip(tmp_path):
    path = tmp_path / "pending.jsonl"
    append_pending_decision("s1", "verified", "校正文字", None, sample_batch="batch1", source="rthk", path=path)
    append_pending_decision("s2", "rejected", None, None, path=path)

    loaded = load_pending_decisions(path)
    assert set(loaded) == {"s1", "s2"}
    assert loaded["s1"]["decision"] == "verified"
    assert loaded["s1"]["text"] == "校正文字"
    assert loaded["s1"]["sample_batch"] == "batch1"


def test_append_pending_decision_resubmit_keeps_only_latest(tmp_path):
    path = tmp_path / "pending.jsonl"
    append_pending_decision("s1", "skipped", None, None, path=path)
    append_pending_decision("s1", "verified", "改咗主意", None, path=path)

    loaded = load_pending_decisions(path)
    assert len(loaded) == 1
    assert loaded["s1"]["decision"] == "verified"
    assert loaded["s1"]["text"] == "改咗主意"


def test_load_pending_decisions_missing_file_returns_empty(tmp_path):
    assert load_pending_decisions(tmp_path / "does_not_exist.jsonl") == {}


def test_run_calibrate_flush_pending_replays_and_archives(scratch_conn, tmp_path):
    conn = scratch_conn
    _seed_segment(conn, "a")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('a', 'pending')")
    path = tmp_path / "pending.jsonl"
    append_pending_decision("a", "verified", "人手校正嘅文字", None, path=path)

    result = asyncio.run(run_calibrate_flush_pending(conn=conn, in_path=path))

    assert result == {"flushed": 1, "errors": 0, "archived_to": result["archived_to"]}
    assert not path.exists()  # renamed away, not left for re-flush
    row = conn.execute("SELECT decision, reviewed_text FROM calibration_review WHERE id='a'").fetchone()
    assert row == ("verified", "人手校正嘅文字")
    tier_row = conn.execute("SELECT tier FROM tiers WHERE id='a'").fetchone()
    assert tier_row == ("gold",)


def test_run_calibrate_flush_pending_empty_buffer_is_noop(scratch_conn, tmp_path):
    result = asyncio.run(run_calibrate_flush_pending(conn=scratch_conn, in_path=tmp_path / "missing.jsonl"))
    assert result == {"flushed": 0, "errors": 0, "archived_to": None}


def test_run_calibrate_flush_pending_leaves_failed_entries_for_retry(scratch_conn, tmp_path):
    conn = scratch_conn
    # 'a' exists in calibration_review, 'ghost' does not -- record_decision's
    # UPDATE affects 0 rows for it (not an exception), so simulate a real
    # failure via an invalid decision value smuggled into the buffer file
    # directly (append_pending_decision itself validates and would refuse).
    _seed_segment(conn, "a")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('a', 'pending')")
    path = tmp_path / "pending.jsonl"
    append_pending_decision("a", "verified", "good", None, path=path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "bad1", "decision": "bogus", "text": None, "flag_reason": None}) + "\n")

    result = asyncio.run(run_calibrate_flush_pending(conn=conn, in_path=path))

    assert result["flushed"] == 1
    assert result["errors"] == 1
    assert result["archived_to"] is None
    assert path.exists()  # left in place for retry, not archived
    remaining = load_pending_decisions(path)
    assert set(remaining) == {"bad1"}


# ---------------------------------------------------------------------------
# progress_report() — T1 QA-backlog tracker (2026-07-17)
# ---------------------------------------------------------------------------

def test_progress_report_splits_pure_and_code_switch(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "a", tier="auto_gold", english_ratio=0.0)
    _seed_segment(conn, "b", tier="auto_gold", english_ratio=0.2)
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('a', 'pending')")
    conn.execute("INSERT INTO calibration_review (id, decision) VALUES ('b', 'pending')")

    report = progress_report(conn)

    assert report["totals"] == {"total": 2, "pending": 2, "reviewed": 0}
    assert report["by_code_switch"]["pure"] == {"total": 1, "pending": 1}
    assert report["by_code_switch"]["code_switch"] == {"total": 1, "pending": 1}
    assert report["breakdown"]["auto_gold"]["pure"]["pending"] == 1
    assert report["breakdown"]["auto_gold"]["code_switch"]["pending"] == 1


def test_progress_report_counts_reviewed_separately_from_pending(scratch_conn):
    conn = scratch_conn
    _seed_segment(conn, "a", tier="silver", english_ratio=0.0)
    conn.execute(
        "INSERT INTO calibration_review (id, decision) VALUES ('a', 'verified')"
    )

    report = progress_report(conn)

    assert report["totals"] == {"total": 1, "pending": 0, "reviewed": 1}
    assert report["by_code_switch"]["pure"] == {"total": 1, "pending": 0}
    assert report["breakdown"]["silver"]["pure"]["verified"] == 1


def test_progress_report_empty_queue(scratch_conn):
    report = progress_report(scratch_conn)
    assert report["totals"] == {"total": 0, "pending": 0, "reviewed": 0}
    assert report["breakdown"] == {}
