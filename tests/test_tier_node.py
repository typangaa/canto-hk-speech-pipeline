import duckdb
import pytest

from pipeline.catalog.catalog import init_schema
from pipeline.nodes.tier import (
    AUTO_GOLD_AGREE_MIN,
    AUTO_GOLD_DNSMOS_MIN,
    BRONZE_AGREE_MIN,
    SILVER_AGREE_MIN,
    assign_tier,
    discover,
)


@pytest.fixture
def scratch_conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "scratch.duckdb"))
    init_schema(conn)
    yield conn
    conn.close()


def _seed_agreement(conn, seg_id, *, agreement=0.9, text_verified=False, model_count=2, dnsmos=4.0):
    conn.execute(
        "INSERT INTO segments (id, audio_path, source, duration_sec, program) "
        "VALUES (?, ?, 'podcast', 6.0, 'test-program')",
        [seg_id, f"/tmp/{seg_id}.flac"],
    )
    conn.execute(
        "INSERT INTO asr_agreement (id, agreement, best_text, text_verified, model_count) "
        "VALUES (?, ?, '呢個係測試', ?, ?)",
        [seg_id, agreement, text_verified, model_count],
    )
    conn.execute("INSERT INTO filters (id, pass, dnsmos) VALUES (?, TRUE, ?)", [seg_id, dnsmos])


def test_assign_tier_text_verified_wins_gold():
    assert assign_tier(True, 0.0) == "gold"
    assert assign_tier(True, 1.0) == "gold"


def test_assign_tier_text_verified_wins_gold_even_over_auto_gold_criteria():
    """text_verified must win even when agreement/dnsmos would also
    qualify for auto_gold -- human verification is always the higher tier."""
    assert assign_tier(True, 1.0, 4.5) == "gold"


def test_assign_tier_high_agreement_without_dnsmos_is_silver():
    """agreement alone (no dnsmos argument) never reaches auto_gold --
    the acoustic-quality gate must be satisfied too."""
    assert assign_tier(False, SILVER_AGREE_MIN) == "silver"
    assert assign_tier(False, 0.90) == "silver"


def test_assign_tier_low_agreement_is_bronze():
    assert assign_tier(False, BRONZE_AGREE_MIN) == "bronze"
    assert assign_tier(False, SILVER_AGREE_MIN - 0.01) == "bronze"


def test_assign_tier_very_low_agreement_is_excluded():
    assert assign_tier(False, BRONZE_AGREE_MIN - 0.01) == "excluded"
    assert assign_tier(False, 0.0) == "excluded"


def test_assign_tier_silver_boundary_exact_threshold():
    assert assign_tier(False, 0.85) == "silver"


def test_assign_tier_bronze_boundary_exact_threshold():
    assert assign_tier(False, 0.70) == "bronze"


def test_assign_tier_auto_gold_both_conditions_met():
    assert assign_tier(False, AUTO_GOLD_AGREE_MIN, AUTO_GOLD_DNSMOS_MIN + 0.1) == "auto_gold"
    assert assign_tier(False, 1.0, 5.0) == "auto_gold"


def test_assign_tier_auto_gold_boundary_agreement_exact_threshold():
    """agreement >= 0.92 is inclusive."""
    assert assign_tier(False, AUTO_GOLD_AGREE_MIN, 5.0) == "auto_gold"


def test_assign_tier_auto_gold_boundary_dnsmos_exact_threshold_is_inclusive():
    """dnsmos >= 3.5 is INCLUSIVE -- exactly 3.5 must qualify."""
    assert assign_tier(False, 0.99, AUTO_GOLD_DNSMOS_MIN) == "auto_gold"
    assert assign_tier(False, 0.99, AUTO_GOLD_DNSMOS_MIN - 0.01) == "silver"


def test_assign_tier_auto_gold_high_agreement_low_dnsmos_falls_to_silver():
    assert assign_tier(False, 0.99, 2.0) == "silver"


def test_assign_tier_auto_gold_high_dnsmos_low_agreement_falls_to_silver_bronze_or_excluded():
    """dnsmos alone is never enough -- agreement must also clear the auto_gold bar."""
    assert assign_tier(False, 0.90, 5.0) == "silver"
    assert assign_tier(False, 0.70, 5.0) == "bronze"
    assert assign_tier(False, 0.50, 5.0) == "excluded"


def test_assign_tier_auto_gold_none_dnsmos_treated_as_failing():
    """dnsmos=None (e.g. filters row not yet written for this id) must never
    qualify for auto_gold, same as any dnsmos < 3.5."""
    assert assign_tier(False, 0.99, None) == "silver"


# ---------------------------------------------------------------------------
# T5 (2026-07-17): discover() re-evaluation on stale model_count, and
# permanent exclusion of human-decided (calibrate_verify/calibrate_reject) rows.
# ---------------------------------------------------------------------------

def test_discover_picks_up_never_tiered_segment(scratch_conn):
    conn = scratch_conn
    _seed_agreement(conn, "a")
    rows = discover(conn)
    assert [r[0] for r in rows] == ["a"]


def test_discover_excludes_already_current_tier(scratch_conn):
    conn = scratch_conn
    _seed_agreement(conn, "a", model_count=2)
    conn.execute(
        "INSERT INTO tiers (id, tier, provenance, asr_model_count) "
        "VALUES ('a', 'silver', 'tier_assign', 2)"
    )
    assert discover(conn) == []


def test_discover_reevaluates_when_model_count_advances(scratch_conn):
    """A later ASR model landing bumps asr_agreement.model_count -- tier.assign
    must re-tier the id even though it was already tiered once."""
    conn = scratch_conn
    _seed_agreement(conn, "a", model_count=3)
    conn.execute(
        "INSERT INTO tiers (id, tier, provenance, asr_model_count) "
        "VALUES ('a', 'silver', 'tier_assign', 2)"
    )
    rows = discover(conn)
    assert [r[0] for r in rows] == ["a"]


def test_discover_legacy_null_model_count_reevaluates(scratch_conn):
    """Legacy P0-imported tiers rows (provenance IS NULL, asr_model_count IS
    NULL) must still be picked up."""
    conn = scratch_conn
    _seed_agreement(conn, "a")
    conn.execute("INSERT INTO tiers (id, tier) VALUES ('a', 'silver')")
    rows = discover(conn)
    assert [r[0] for r in rows] == ["a"]


def test_discover_never_revisits_human_verified_row_even_if_model_count_advances(scratch_conn):
    """A human 'verified' decision (calibrate.py's record_decision) writes
    tiers.tier='gold' with provenance='calibrate_verify'. Even if a later ASR
    model then bumps asr_agreement.model_count, tier.assign discovery must NOT
    re-pick this id up -- a human decision is terminal, never silently
    revisited by a statistical recompute."""
    conn = scratch_conn
    _seed_agreement(conn, "a", text_verified=True, model_count=3)
    conn.execute(
        "INSERT INTO tiers (id, tier, provenance) VALUES ('a', 'gold', 'calibrate_verify')"
    )
    assert discover(conn) == []


def test_discover_never_revisits_human_rejected_row_even_if_model_count_advances(scratch_conn):
    """Same protection for a 'rejected' decision (tiers.tier='excluded',
    provenance='calibrate_reject') -- this is the actual bug T5 found and
    fixed: without this exclusion, the next tier.assign run would recompute
    from agreement/dnsmos alone and could silently un-reject the segment."""
    conn = scratch_conn
    _seed_agreement(conn, "a", agreement=0.99, dnsmos=5.0, model_count=3)
    conn.execute(
        "INSERT INTO tiers (id, tier, provenance) VALUES ('a', 'excluded', 'calibrate_reject')"
    )
    assert discover(conn) == []
