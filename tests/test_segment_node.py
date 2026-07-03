from pathlib import Path

import numpy as np
import soundfile as sf

import pipeline.nodes.segment as segment
from pipeline.nodes.segment import (
    MAX_DUR,
    MIN_DUR,
    TARGET_SR,
    _check_legacy_sidecar,
    _pregate_one,
    _segment_id,
    _vad_cut_one,
    compute_pregate_snr,
)


# ---------------------------------------------------------------------------
# _segment_id() — stable md5-based convention from docs/MANIFEST_SCHEMA.md
# ---------------------------------------------------------------------------

def test_segment_id_stable_for_same_path(tmp_path):
    p = tmp_path / "a_seg00000.wav"
    assert _segment_id(p) == _segment_id(p)


def test_segment_id_differs_for_different_paths(tmp_path):
    a = tmp_path / "a_seg00000.wav"
    b = tmp_path / "a_seg00001.wav"
    assert _segment_id(a) != _segment_id(b)


def test_segment_id_is_12_hex_chars(tmp_path):
    seg_id = _segment_id(tmp_path / "x.wav")
    assert len(seg_id) == 12
    int(seg_id, 16)  # raises ValueError if not hex


# ---------------------------------------------------------------------------
# _check_legacy_sidecar() — legacy `_segments.jsonl` reuse check (I/O, tmp_path)
# ---------------------------------------------------------------------------

def test_check_legacy_sidecar_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(segment, "SEGMENTS_ROOT", tmp_path)
    (tmp_path / "rthk").mkdir()
    wav_path = "/mnt/Drive2/canto-corpus/data/raw/rthk/20250101_prog_abc123.wav"
    sidecar = tmp_path / "rthk" / "20250101_prog_abc123_segments.jsonl"
    sidecar.write_text('{"seg_path": "x"}\n{"seg_path": "y"}\n')

    raw_id, out_path, source, n = _check_legacy_sidecar(("abc123", wav_path, "rthk"))
    assert raw_id == "abc123"
    assert out_path == wav_path
    assert source == "rthk"
    assert n == 2


def test_check_legacy_sidecar_miss_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(segment, "SEGMENTS_ROOT", tmp_path)
    (tmp_path / "rthk").mkdir()
    wav_path = "/mnt/Drive2/canto-corpus/data/raw/rthk/20250101_missing_def456.wav"

    _, _, _, n = _check_legacy_sidecar(("def456", wav_path, "rthk"))
    assert n == 0


def test_check_legacy_sidecar_miss_empty_file(tmp_path, monkeypatch):
    monkeypatch.setattr(segment, "SEGMENTS_ROOT", tmp_path)
    (tmp_path / "rthk").mkdir()
    wav_path = "/mnt/Drive2/canto-corpus/data/raw/rthk/20250101_empty_ghi789.wav"
    sidecar = tmp_path / "rthk" / "20250101_empty_ghi789_segments.jsonl"
    sidecar.write_text("")

    _, _, _, n = _check_legacy_sidecar(("ghi789", wav_path, "rthk"))
    assert n == 0  # empty sidecar counts as a cache miss, not a reuse hit


# ---------------------------------------------------------------------------
# compute_pregate_snr() — 03b's own sliding-hop percentile formula
# ---------------------------------------------------------------------------

def test_compute_pregate_snr_uniform_signal_near_zero():
    rng = np.random.default_rng(0)
    wav = rng.normal(scale=0.1, size=TARGET_SR * 2).astype(np.float32)
    snr = compute_pregate_snr(wav, TARGET_SR)
    assert -3.0 <= snr <= 3.0  # uniform-energy noise: ~flat frame energies


def test_compute_pregate_snr_loud_and_quiet_mix_is_higher():
    rng = np.random.default_rng(0)
    quiet = rng.normal(scale=0.001, size=TARGET_SR).astype(np.float32)
    loud = rng.normal(scale=0.5, size=TARGET_SR).astype(np.float32)
    mixed = np.concatenate([quiet, loud])
    uniform = rng.normal(scale=0.1, size=TARGET_SR * 2).astype(np.float32)

    snr_mixed = compute_pregate_snr(mixed, TARGET_SR)
    snr_uniform = compute_pregate_snr(uniform, TARGET_SR)
    assert snr_mixed > snr_uniform


def test_compute_pregate_snr_empty_array_returns_zero():
    assert compute_pregate_snr(np.array([], dtype=np.float32), TARGET_SR) == 0.0


# ---------------------------------------------------------------------------
# _pregate_one() — fail-open on unreadable audio (verbatim 03b behaviour)
# ---------------------------------------------------------------------------

def test_pregate_one_fail_open_on_missing_file(tmp_path):
    result = _pregate_one(
        "seg_missing", str(tmp_path / "does_not_exist.wav"), min_snr=25.0, min_dnsmos=3.0,
    )
    assert result["pass"] is True
    assert result["fail_reason"] is None
    assert result["snr_db"] is None
    assert result["dnsmos"] is None


def test_pregate_one_rejects_low_snr(tmp_path):
    wav_path = tmp_path / "quiet.wav"
    rng = np.random.default_rng(0)
    wav = rng.normal(scale=0.1, size=TARGET_SR * 2).astype(np.float32)
    sf.write(str(wav_path), wav, TARGET_SR, subtype="PCM_16")

    result = _pregate_one("seg_quiet", str(wav_path), min_snr=999.0, min_dnsmos=0.0)
    assert result["pass"] is False
    assert result["fail_reason"] == "snr"
    assert result["dnsmos"] is None  # DNSMOS skipped once SNR already failed


# ---------------------------------------------------------------------------
# _vad_cut_one() — VAD-and-cut plumbing, with a monkeypatched VAD window so the
# test exercises the cutting/writing/id-generation logic deterministically
# rather than depending on Silero VAD's actual speech-detection behaviour on
# synthetic audio.
# ---------------------------------------------------------------------------

def test_vad_cut_one_writes_expected_segment(tmp_path, monkeypatch):
    monkeypatch.setattr(segment, "SEGMENTS_ROOT", tmp_path)

    duration = 6.0
    rng = np.random.default_rng(0)
    wav48 = rng.normal(scale=0.2, size=int(TARGET_SR * duration)).astype(np.float32)
    raw_dir = tmp_path / "_raw_src"
    raw_dir.mkdir()
    wav_path = raw_dir / "20250101_prog_rawid1.wav"
    sf.write(str(wav_path), wav48, TARGET_SR, subtype="PCM_16")

    # Force one VAD window spanning the whole turn, regardless of what Silero
    # VAD would actually detect on synthetic noise.
    monkeypatch.setattr(
        segment, "_get_vad_segments_in_window",
        lambda wav16, window_start, window_end, chunk_sec=60.0: [(window_start, window_end)],
    )

    result = _vad_cut_one(
        raw_id="rawid1",
        wav_path=str(wav_path),
        source="podcast",
        source_url="https://example.com/x",
        program="Test Program",
        domain="podcast",
        style="casual",
        turns=[(0, 0.0, duration, "SPEAKER_00")],
    )

    assert result["error"] is None
    assert result["n_segments"] == 1
    row = result["segment_rows"][0]
    assert row["raw_id"] == "rawid1"
    assert row["source"] == "podcast"
    assert MIN_DUR <= row["duration_sec"] <= MAX_DUR
    assert Path(row["audio_path"]).exists()
    assert Path(row["audio_path"]).name == "20250101_prog_rawid1_seg00000.wav"


def test_vad_cut_one_skips_windows_outside_duration_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(segment, "SEGMENTS_ROOT", tmp_path)

    duration = 25.0  # deliberately long turn
    rng = np.random.default_rng(0)
    wav48 = rng.normal(scale=0.2, size=int(TARGET_SR * duration)).astype(np.float32)
    raw_dir = tmp_path / "_raw_src2"
    raw_dir.mkdir()
    wav_path = raw_dir / "20250101_prog_rawid2.wav"
    sf.write(str(wav_path), wav48, TARGET_SR, subtype="PCM_16")

    # One VAD window spanning the full 25s turn — over MAX_DUR (20s), must be
    # rejected rather than cut.
    monkeypatch.setattr(
        segment, "_get_vad_segments_in_window",
        lambda wav16, window_start, window_end, chunk_sec=60.0: [(window_start, window_end)],
    )

    result = _vad_cut_one(
        raw_id="rawid2",
        wav_path=str(wav_path),
        source="podcast",
        source_url="",
        program="",
        domain="",
        style="",
        turns=[(0, 0.0, duration, "SPEAKER_00")],
    )

    assert result["error"] is None
    assert result["n_segments"] == 0
    assert result["segment_rows"] == []


def test_vad_cut_one_unreadable_audio_reports_error(tmp_path, monkeypatch):
    monkeypatch.setattr(segment, "SEGMENTS_ROOT", tmp_path)

    result = _vad_cut_one(
        raw_id="rawid3",
        wav_path=str(tmp_path / "does_not_exist.wav"),
        source="podcast",
        source_url="",
        program="",
        domain="",
        style="",
        turns=[(0, 0.0, 5.0, "SPEAKER_00")],
    )

    assert result["n_segments"] == 0
    assert result["segment_rows"] == []
    assert result["error"] is not None
