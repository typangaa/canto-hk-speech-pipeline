import pytest

from pipeline.nodes.label_calibrate import (
    MIN_SPEAKER_SAMPLES,
    compute_rate_percentiles,
    compute_speaker_pitch_stats,
)


def test_compute_rate_percentiles_basic():
    values = list(range(1, 101))  # 1..100
    stats = compute_rate_percentiles(values)
    assert stats["n_samples"] == 100
    assert 25.0 <= stats["p25"] <= 26.0
    assert 75.0 <= stats["p75"] <= 76.0


def test_compute_rate_percentiles_empty_raises():
    with pytest.raises(ValueError):
        compute_rate_percentiles([])


def test_compute_speaker_pitch_stats_below_min_falls_back_to_corpus():
    # speaker "rare" has only 2 samples (< MIN_SPEAKER_SAMPLES), speaker "common" has 10
    rows = [("rare", 100.0), ("rare", 110.0)]
    rows += [("common", 200.0)] * 10
    per_speaker, corpus_fallback, counts = compute_speaker_pitch_stats(rows)

    assert "rare" not in per_speaker
    assert "common" in per_speaker
    assert counts["below_min"] == 1
    assert counts["calibrated"] == 1
    assert counts["corpus_total"] == 12
    assert corpus_fallback["mu"] > 0


def test_compute_speaker_pitch_stats_sigma_epsilon_clamped():
    # a speaker with IDENTICAL f0 every time -> raw sigma == 0, must clamp to epsilon
    rows = [("flat_speaker", 150.0)] * (MIN_SPEAKER_SAMPLES + 1)
    per_speaker, _corpus_fallback, _counts = compute_speaker_pitch_stats(rows)
    assert per_speaker["flat_speaker"]["sigma"] == 1.0  # _SIGMA_EPSILON
    assert per_speaker["flat_speaker"]["mu"] == 150.0


def test_compute_speaker_pitch_stats_empty_rows_raises():
    with pytest.raises(ValueError):
        compute_speaker_pitch_stats([])


def test_compute_speaker_pitch_stats_respects_custom_min_samples():
    rows = [("spk", 100.0), ("spk", 102.0), ("spk", 98.0)]
    per_speaker, _fallback, counts = compute_speaker_pitch_stats(rows, min_samples=3)
    assert "spk" in per_speaker
    assert counts["calibrated"] == 1
    assert counts["below_min"] == 0
