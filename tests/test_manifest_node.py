import json

import duckdb
import pytest

from pipeline.catalog.catalog import init_schema
from pipeline.nodes.manifest import (
    INCLUDED_TIERS,
    QUALITY_TIER_PRECEDENCE,
    TIER_PRECEDENCE,
    _export_tag,
    _quality_tiers_at_or_above,
    _stratified_split_new_entries,
    _tiers_at_or_above,
    build_entry,
    build_text_pause,
    discover,
    train_val_split,
)


def _row(**overrides):
    base = (
        "seg001", "/mnt/Drive4/canto/filtered/youtube/x.wav", "youtube", "", "", "other",
        6.97, 48000, "youtube_001", "male", "formal", "2026-06-09",
        "hello world", False, 0.91,
        "nei5 hou2", "nei5 hou2",
        35.2, 3.8, 0.02,
        "silver",
        None, None, None, None, None,
    )
    if not overrides:
        return base
    fields = [
        "seg_id", "audio_path", "source", "source_url", "program", "domain",
        "duration_sec", "sample_rate", "speaker_id", "gender", "style", "created_at",
        "best_text", "text_verified", "agreement",
        "jyutping", "jyutping_cs",
        "snr_db", "dnsmos", "english_ratio",
        "tier",
        "pause_plan_raw", "n_punct", "n_no_pause", "n_short", "n_long",
    ]
    d = dict(zip(fields, base))
    d.update(overrides)
    return tuple(d[f] for f in fields)


def test_build_entry_basic_fields():
    entry = build_entry(_row(), asr_candidates=[{"model": "m", "text": "hello world", "confidence": 0.9}])
    assert entry["id"] == "seg001"
    assert entry["source"] == "youtube"
    assert entry["text"] == "hello world"
    assert entry["text_verified"] is False
    assert entry["tier"] == "silver"
    assert entry["jyutping"] == "nei5 hou2"
    assert entry["jyutping_cs"] == "nei5 hou2"


def test_build_entry_rounding_matches_manifest_schema():
    entry = build_entry(_row(duration_sec=6.9666, snr_db=35.249, dnsmos=3.801, agreement=0.9051), [])
    assert entry["duration_sec"] == 6.967
    assert entry["snr_db"] == 35.2
    assert entry["dnsmos"] == 3.8
    assert entry["asr_agreement"] == 0.905


def test_build_entry_missing_speaker_id_falls_back_to_source_unk():
    entry = build_entry(_row(speaker_id=None), [])
    assert entry["speaker_id"] == "youtube_unk"


def test_build_entry_none_gender_style_default_unknown_formal():
    entry = build_entry(_row(gender=None, style=None), [])
    assert entry["gender"] == "unknown"
    assert entry["style"] == "formal"


def test_build_entry_missing_created_at_falls_back_to_today():
    entry = build_entry(_row(created_at=None), [])
    assert entry["created_at"]  # non-empty ISO-ish string


# ---------------------------------------------------------------------------
# P3 pause-token fields -- build_text_pause() / build_entry() additive fields
# ---------------------------------------------------------------------------

def test_build_text_pause_no_plan_returns_text_unchanged():
    assert build_text_pause("你好，世界。", None) == "你好，世界。"
    assert build_text_pause("你好，世界。", []) == "你好，世界。"


def test_build_text_pause_short_verdict_inserts_token_after_mark():
    text = "你好，世界"
    plan = [{"offset": 2, "mark": "，", "kind": "normal", "delta_t": 0.2, "verdict": "short"}]
    assert build_text_pause(text, plan) == "你好，<pause-short>世界"


def test_build_text_pause_long_verdict_inserts_token_after_mark():
    text = "你好，世界"
    plan = [{"offset": 2, "mark": "，", "kind": "normal", "delta_t": 0.5, "verdict": "long"}]
    assert build_text_pause(text, plan) == "你好，<pause-long>世界"


def test_build_text_pause_no_pause_verdict_strips_mark():
    text = "你好，世界"
    plan = [{"offset": 2, "mark": "，", "kind": "normal", "delta_t": 0.01, "verdict": "no_pause"}]
    assert build_text_pause(text, plan) == "你好世界"


def test_build_text_pause_trailing_tail_left_untouched():
    text = "你好世界。"
    plan = [{"offset": 4, "mark": "。", "kind": "trailing_tail", "delta_t": 0.1}]
    assert build_text_pause(text, plan) == "你好世界。"


def test_build_text_pause_multiple_marks_mixed_verdicts():
    text = "甲，乙，丙。"
    plan = [
        {"offset": 1, "mark": "，", "kind": "normal", "delta_t": 0.01, "verdict": "no_pause"},
        {"offset": 3, "mark": "，", "kind": "normal", "delta_t": 0.5, "verdict": "long"},
        {"offset": 5, "mark": "。", "kind": "trailing_tail", "delta_t": 0.1},
    ]
    assert build_text_pause(text, plan) == "甲乙，<pause-long>丙。"


def test_build_text_pause_drift_guard_skips_stale_offset():
    # `plan` was computed against different text -- offset 2 is '世' here, not '，'.
    text = "你好世界"
    plan = [{"offset": 2, "mark": "，", "kind": "normal", "delta_t": 0.2, "verdict": "short"}]
    assert build_text_pause(text, plan) == "你好世界"


def test_build_entry_omits_pause_fields_when_no_pause_plan_row():
    entry = build_entry(_row(), [])
    assert "text_pause" not in entry
    assert "punct_audit" not in entry


def test_build_entry_includes_pause_fields_when_pause_plan_row_present():
    plan_json = json.dumps([
        {"offset": 2, "mark": "，", "kind": "normal", "delta_t": 0.01, "verdict": "no_pause"},
    ])
    entry = build_entry(
        _row(
            best_text="你好，世界", pause_plan_raw=plan_json,
            n_punct=1, n_no_pause=1, n_short=0, n_long=0,
        ),
        [],
    )
    assert entry["text_pause"] == "你好世界"
    assert entry["punct_audit"] == {"n_punct": 1, "n_no_pause": 1, "n_short": 0, "n_long": 0}


def test_build_entry_pause_plan_unalignable_all_zero_counts_not_omitted():
    entry = build_entry(
        _row(
            best_text="你好世界", pause_plan_raw="[]",
            n_punct=0, n_no_pause=0, n_short=0, n_long=0,
        ),
        [],
    )
    assert entry["text_pause"] == "你好世界"
    assert entry["punct_audit"] == {"n_punct": 0, "n_no_pause": 0, "n_short": 0, "n_long": 0}


def test_included_tiers_excludes_the_excluded_sentinel():
    assert "excluded" not in INCLUDED_TIERS
    assert set(INCLUDED_TIERS) == {"gold", "auto_gold", "silver", "bronze"}


# ---------------------------------------------------------------------------
# min_tier cut -- _tiers_at_or_above() / _export_tag() / discover()
# ---------------------------------------------------------------------------

def test_tiers_at_or_above_auto_gold_includes_gold():
    assert _tiers_at_or_above("auto_gold") == ("gold", "auto_gold")


def test_tiers_at_or_above_bronze_includes_everything():
    assert _tiers_at_or_above("bronze") == TIER_PRECEDENCE


def test_tiers_at_or_above_gold_is_gold_only():
    assert _tiers_at_or_above("gold") == ("gold",)


def test_tiers_at_or_above_rejects_excluded():
    with pytest.raises(ValueError):
        _tiers_at_or_above("excluded")


def test_export_tag_none_when_unfiltered():
    assert _export_tag(None, None) is None


def test_export_tag_min_tier_only():
    assert _export_tag(None, "auto_gold") == "tier_auto_gold"


def test_export_tag_min_agreement_only():
    assert _export_tag(0.95, None) == "agree095"


def test_export_tag_combines_both():
    assert _export_tag(0.90, "silver") == "tier_silver_agree090"


def test_export_tag_code_switch_only():
    assert _export_tag(None, None, "only") == "codeswitch_only"


def test_export_tag_code_switch_exclude():
    assert _export_tag(None, None, "exclude") == "codeswitch_exclude"


def test_export_tag_combines_all_three():
    assert _export_tag(0.90, "silver", "only") == "tier_silver_agree090_codeswitch_only"


# ---------------------------------------------------------------------------
# min_quality_tier cut (added 2026-07-16, T13) -- SEPARATE axis from min_tier
# ---------------------------------------------------------------------------

def test_quality_tiers_at_or_above_b_is_b_only():
    assert _quality_tiers_at_or_above("B") == ("B",)


def test_quality_tiers_at_or_above_a_includes_everything():
    assert _quality_tiers_at_or_above("A") == QUALITY_TIER_PRECEDENCE == ("B", "A")


def test_quality_tiers_at_or_above_rejects_invalid():
    with pytest.raises(ValueError):
        _quality_tiers_at_or_above("C")


def test_export_tag_min_quality_tier_only():
    assert _export_tag(None, None, None, "B") == "qualityB"


def test_export_tag_combines_min_tier_and_min_quality_tier():
    assert _export_tag(None, "auto_gold", None, "B") == "tier_auto_gold_qualityB"


@pytest.fixture
def scratch_conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "scratch.duckdb"))
    init_schema(conn)
    yield conn
    conn.close()


def _seed_manifest_row(conn, seg_id, *, tier, agreement=0.9, english_ratio=0.0, quality_tier=None):
    conn.execute(
        "INSERT INTO segments (id, audio_path, source, duration_sec, sample_rate) "
        "VALUES (?, '/tmp/x.flac', 'podcast', 6.0, 48000)",
        [seg_id],
    )
    conn.execute(
        "INSERT INTO asr_agreement (id, agreement, best_text, text_verified) VALUES (?, ?, 'hello', FALSE)",
        [seg_id, agreement],
    )
    conn.execute("INSERT INTO g2p (id, jyutping, valid_fraction, provenance) VALUES (?, 'hello', 1.0, 'g2p_node')", [seg_id])
    conn.execute(
        "INSERT INTO filters (id, pass, english_ratio, provenance) VALUES (?, TRUE, ?, 'filter_decide')",
        [seg_id, english_ratio],
    )
    conn.execute("INSERT INTO tiers (id, tier, provenance) VALUES (?, ?, 'tier_assign')", [seg_id, tier])
    if quality_tier is not None:
        conn.execute(
            "INSERT INTO quality_tiers (id, quality_tier, provenance) VALUES (?, ?, 'quality_tier_assign')",
            [seg_id, quality_tier],
        )


def test_discover_joins_pause_plan_when_present(scratch_conn):
    conn = scratch_conn
    _seed_manifest_row(conn, "ag1", tier="auto_gold")
    _seed_manifest_row(conn, "sv1", tier="silver")
    conn.execute(
        "INSERT INTO pause_plan (id, plan, n_punct, n_no_pause, n_short, n_long, "
        "unalignable, calibration_version, provenance) "
        "VALUES ('ag1', '[]', 0, 0, 0, 0, FALSE, 'v1', 'pause_plan')"
    )

    ag1 = next(row for row in discover(conn) if row[0] == "ag1")
    sv1 = next(row for row in discover(conn) if row[0] == "sv1")
    assert ag1[-5] == "[]"  # pp.plan
    assert sv1[-5] is None  # no pause_plan row for silver -> LEFT JOIN NULL


def test_discover_min_tier_auto_gold_includes_gold_excludes_silver(scratch_conn):
    conn = scratch_conn
    _seed_manifest_row(conn, "g1", tier="gold")
    _seed_manifest_row(conn, "ag1", tier="auto_gold")
    _seed_manifest_row(conn, "sv1", tier="silver")
    _seed_manifest_row(conn, "br1", tier="bronze")

    picked_ids = {row[0] for row in discover(conn, min_tier="auto_gold")}
    assert picked_ids == {"g1", "ag1"}


def test_discover_min_tier_none_returns_all_included_tiers(scratch_conn):
    conn = scratch_conn
    _seed_manifest_row(conn, "g1", tier="gold")
    _seed_manifest_row(conn, "br1", tier="bronze")
    _seed_manifest_row(conn, "ex1", tier="excluded")

    picked_ids = {row[0] for row in discover(conn)}
    assert picked_ids == {"g1", "br1"}


def test_discover_min_tier_and_min_agreement_combine(scratch_conn):
    conn = scratch_conn
    _seed_manifest_row(conn, "sv_low", tier="silver", agreement=0.86)
    _seed_manifest_row(conn, "sv_high", tier="silver", agreement=0.97)

    picked_ids = {row[0] for row in discover(conn, min_agreement=0.95, min_tier="silver")}
    assert picked_ids == {"sv_high"}


# ---------------------------------------------------------------------------
# code_switch cut (added 2026-07-15, T18) -- filters.english_ratio > 0 / = 0
# ---------------------------------------------------------------------------

def test_discover_code_switch_none_returns_all(scratch_conn):
    conn = scratch_conn
    _seed_manifest_row(conn, "mono1", tier="silver", english_ratio=0.0)
    _seed_manifest_row(conn, "cs1", tier="silver", english_ratio=0.15)

    picked_ids = {row[0] for row in discover(conn)}
    assert picked_ids == {"mono1", "cs1"}


def test_discover_code_switch_only_excludes_pure_cantonese(scratch_conn):
    conn = scratch_conn
    _seed_manifest_row(conn, "mono1", tier="silver", english_ratio=0.0)
    _seed_manifest_row(conn, "cs1", tier="silver", english_ratio=0.15)

    picked_ids = {row[0] for row in discover(conn, code_switch="only")}
    assert picked_ids == {"cs1"}


def test_discover_code_switch_exclude_excludes_code_switch(scratch_conn):
    conn = scratch_conn
    _seed_manifest_row(conn, "mono1", tier="silver", english_ratio=0.0)
    _seed_manifest_row(conn, "cs1", tier="silver", english_ratio=0.15)

    picked_ids = {row[0] for row in discover(conn, code_switch="exclude")}
    assert picked_ids == {"mono1"}


def test_discover_code_switch_combines_with_min_tier(scratch_conn):
    conn = scratch_conn
    _seed_manifest_row(conn, "ag_cs", tier="auto_gold", english_ratio=0.2)
    _seed_manifest_row(conn, "sv_cs", tier="silver", english_ratio=0.2)
    _seed_manifest_row(conn, "ag_mono", tier="auto_gold", english_ratio=0.0)

    picked_ids = {row[0] for row in discover(conn, min_tier="auto_gold", code_switch="only")}
    assert picked_ids == {"ag_cs"}


def test_discover_rejects_invalid_code_switch(scratch_conn):
    with pytest.raises(ValueError):
        discover(scratch_conn, code_switch="bogus")


# ---------------------------------------------------------------------------
# min_quality_tier cut, integration -- discover() with a real quality_tiers join
# ---------------------------------------------------------------------------

def test_discover_min_quality_tier_none_ignores_axis_entirely(scratch_conn):
    """Segments with no quality_tiers row at all (e.g. silver/bronze, which the
    quality_tier.assign node never scopes) must stay included when the filter
    is unused -- this is a LEFT JOIN, not an INNER JOIN."""
    conn = scratch_conn
    _seed_manifest_row(conn, "sv_no_qt", tier="silver")  # no quality_tier row
    _seed_manifest_row(conn, "ag_a", tier="auto_gold", quality_tier="A")
    _seed_manifest_row(conn, "ag_b", tier="auto_gold", quality_tier="B")

    picked_ids = {row[0] for row in discover(conn)}
    assert picked_ids == {"sv_no_qt", "ag_a", "ag_b"}


def test_discover_min_quality_tier_b_excludes_a_and_unscored(scratch_conn):
    conn = scratch_conn
    _seed_manifest_row(conn, "sv_no_qt", tier="silver")
    _seed_manifest_row(conn, "ag_a", tier="auto_gold", quality_tier="A")
    _seed_manifest_row(conn, "ag_b", tier="auto_gold", quality_tier="B")

    picked_ids = {row[0] for row in discover(conn, min_quality_tier="B")}
    assert picked_ids == {"ag_b"}


def test_discover_min_quality_tier_a_includes_a_and_b_but_not_unscored(scratch_conn):
    conn = scratch_conn
    _seed_manifest_row(conn, "sv_no_qt", tier="silver")
    _seed_manifest_row(conn, "ag_a", tier="auto_gold", quality_tier="A")
    _seed_manifest_row(conn, "ag_b", tier="auto_gold", quality_tier="B")

    picked_ids = {row[0] for row in discover(conn, min_quality_tier="A")}
    assert picked_ids == {"ag_a", "ag_b"}


def test_discover_min_quality_tier_combines_with_min_tier(scratch_conn):
    conn = scratch_conn
    _seed_manifest_row(conn, "gold_b", tier="gold", quality_tier="B")
    _seed_manifest_row(conn, "ag_b", tier="auto_gold", quality_tier="B")

    picked_ids = {row[0] for row in discover(conn, min_tier="gold", min_quality_tier="B")}
    assert picked_ids == {"gold_b"}


def test_discover_rejects_invalid_min_quality_tier(scratch_conn):
    with pytest.raises(ValueError):
        discover(scratch_conn, min_quality_tier="C")


# ---------------------------------------------------------------------------
# train/val split -- preserve existing membership, extend for new ids only
# ---------------------------------------------------------------------------

def _entry(id_, source, speaker_id):
    return {"id": id_, "source": source, "speaker_id": speaker_id}


def test_stratified_split_new_entries_no_speaker_leakage():
    entries = [_entry(f"id{i}", "youtube", f"spk{i % 10}") for i in range(100)]
    train, val = _stratified_split_new_entries(entries, val_frac=0.2)
    train_speakers = {e["speaker_id"] for e in train}
    val_speakers = {e["speaker_id"] for e in val}
    assert not (train_speakers & val_speakers)
    assert len(train) + len(val) == len(entries)


def test_train_val_split_preserves_existing_membership(tmp_path):
    existing_train = tmp_path / "train.jsonl"
    existing_val = tmp_path / "val.jsonl"
    existing_train.write_text('{"id": "old1"}\n{"id": "old2"}\n')
    existing_val.write_text('{"id": "old3"}\n')

    entries = [
        _entry("old1", "youtube", "spkA"),
        _entry("old2", "youtube", "spkA"),
        _entry("old3", "youtube", "spkB"),
        _entry("new1", "youtube", "spkC"),
    ]
    train, val = train_val_split(entries, train_path=existing_train, val_path=existing_val, val_frac=0.05)

    train_ids = {e["id"] for e in train}
    val_ids = {e["id"] for e in val}
    assert {"old1", "old2"} <= train_ids
    assert "old3" in val_ids
    # the new id must land in exactly one split, not both or neither
    assert ("new1" in train_ids) != ("new1" in val_ids)


def test_train_val_split_no_existing_files_falls_back_to_full_stratified_split(tmp_path):
    entries = [_entry(f"id{i}", "podcast", f"spk{i % 5}") for i in range(50)]
    train, val = train_val_split(
        entries, train_path=tmp_path / "missing_train.jsonl", val_path=tmp_path / "missing_val.jsonl",
        val_frac=0.2,
    )
    assert len(train) + len(val) == 50
