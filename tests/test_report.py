"""Smoke + unit tests for report.build (pipeline/nodes/report.py).

The pure-function pieces (percentile, _check, _jyutping_valid_rate,
generate_report on hand-built entries) are tested without touching the
catalog at all. The one integration test exercises run_report_build()
against the REAL catalog, following tests/test_catalog.py's
skip-if-catalog-missing pattern (run_report_build() opens its own
connect_ro() internally, same as run_manifest_build(), so no fixture is
needed here beyond the existence check).
"""

import os

import pytest

from pipeline.config import CATALOG_PATH, REPORT_PATH
from pipeline.nodes.report import (
    ACCEPT,
    JYUTPING_TOKEN,
    _check,
    _jyutping_valid_rate,
    generate_report,
    percentile,
    run_report_build,
)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_percentile_empty_returns_zero():
    assert percentile([], 50) == 0.0


def test_percentile_p50_median():
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0


def test_jyutping_token_regex_matches_valid_syllable():
    assert JYUTPING_TOKEN.match("nei5")
    assert JYUTPING_TOKEN.match("hou2")


def test_jyutping_token_regex_rejects_bad_tone_and_uppercase():
    assert not JYUTPING_TOKEN.match("nei")       # missing tone digit
    assert not JYUTPING_TOKEN.match("nei7")       # tone out of 1-6 range
    assert not JYUTPING_TOKEN.match("NEI5")       # uppercase


def test_check_pass_ge():
    result = _check("X", 5.0, 3.0)
    assert result["passed"] is True
    assert "PASS" in result["line"]


def test_check_fail_ge():
    result = _check("X", 2.0, 3.0)
    assert result["passed"] is False
    assert "FAIL" in result["line"]


def test_check_le_direction():
    assert _check("Windows paths", 0, 0, good_if="<=")["passed"] is True
    assert _check("Windows paths", 1, 0, good_if="<=")["passed"] is False


def test_jyutping_valid_rate_token_weighted():
    entries = [
        {"jyutping": "nei5 hou2"},          # 2/2 valid
        {"jyutping": "nei5 bad NEI5"},       # 1/3 valid
        {"jyutping": ""},                    # 0/0
    ]
    rate, valid, total = _jyutping_valid_rate(entries)
    assert total == 5
    assert valid == 3
    assert rate == pytest.approx(3 / 5)


def test_jyutping_valid_rate_no_tokens_is_zero_not_nan():
    rate, valid, total = _jyutping_valid_rate([{"jyutping": None}, {"jyutping": ""}])
    assert rate == 0.0
    assert valid == 0
    assert total == 0


def _entry(**overrides):
    base = {
        "id": "seg001",
        "audio_path": "/mnt/Drive4/canto/segments/youtube/x.flac",
        "source": "youtube",
        "domain": "documentary",
        "duration_sec": 6.97,
        "sample_rate": 48000,
        "speaker_id": "youtube_001",
        "gender": "male",
        "text_verified": True,
        "dnsmos": 3.8,
        "snr_db": 35.2,
        "asr_agreement": 0.95,
        "jyutping": "nei5 hou2",
    }
    base.update(overrides)
    return base


def test_generate_report_empty_entries():
    md, criteria = generate_report([])
    assert "No manifest-eligible entries" in md
    assert criteria == []


def test_generate_report_all_pass_reports_ready():
    # Need: >=100h, >=100 unique speakers, >=3 sources/domains, all durations
    # within [3,20]s, all text_verified/48kHz, no Windows paths, valid
    # Jyutping -- 20000 entries * 20s = ~111h with 120 distinct speakers
    # clears every computed criterion at once so the "all pass -> READY"
    # plumbing can be exercised without needing the real catalog.
    entries = [
        _entry(
            id=f"s{i}",
            speaker_id=f"spk{i % 120}",
            source=["youtube", "rthk", "podcast"][i % 3],
            domain=["documentary", "news", "talk_show"][i % 3],
            duration_sec=20.0,
        )
        for i in range(20000)
    ]

    md, criteria = generate_report(entries)
    assert len(criteria) == 11
    assert all(c["passed"] for c in criteria)
    assert "READY FOR TRAINING" in md


def test_generate_report_windows_path_detected():
    entries = [_entry(audio_path="/mnt/D/canto/segments/x.wav")]
    md, criteria = generate_report(entries)
    win_check = next(c for c in criteria if c["label"] == "Windows paths in manifest")
    assert win_check["passed"] is False
    assert win_check["value"] == 1


def test_generate_report_min_tier_note_in_markdown():
    md, _ = generate_report([_entry()], min_tier="gold")
    assert "min_tier=gold" in md


def test_accept_thresholds_match_claude_md_current_values():
    """Guards against ACCEPT drifting stale the same way scripts/10_report.py's
    ACCEPT dict did (CLAUDE.md's Acceptance Criteria table is the SSOT)."""
    assert ACCEPT["dnsmos_p50_min"] == 3.2
    assert ACCEPT["snr_p50_min"] == 30.0
    assert ACCEPT["jyutping_valid_rate_min"] == 0.99
    assert ACCEPT["total_hours_min"] == 100.0
    assert ACCEPT["speakers_min"] == 100
    assert ACCEPT["sources_min"] == 3
    assert ACCEPT["domains_min"] == 3
    assert ACCEPT["windows_paths_max"] == 0


# ---------------------------------------------------------------------------
# integration: run_report_build() against the REAL catalog
# ---------------------------------------------------------------------------


def test_run_report_build_smoke_against_real_catalog():
    if not os.path.exists(CATALOG_PATH):
        pytest.skip("catalog not built — run: python -m pipeline.cli catalog build")

    result = run_report_build()

    assert set(result.keys()) == {
        "path", "count", "total_hours", "n_speakers", "tier_counts",
        "criteria", "overall_pass", "min_tier",
    }
    assert result["path"] == str(REPORT_PATH)
    assert result["count"] > 0
    assert result["total_hours"] > 0
    assert result["n_speakers"] > 0
    assert len(result["criteria"]) == 11
    assert os.path.exists(REPORT_PATH)

    with open(REPORT_PATH, encoding="utf-8") as f:
        content = f.read()
    assert "Acceptance Criteria" in content
    assert "report.build" in content
