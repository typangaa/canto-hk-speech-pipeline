from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pytest
import soundfile as sf

from pipeline.catalog.catalog import init_schema
from pipeline.nodes.raw_flac import (
    TRANSCODE_DISCOVER_SQL,
    _cap_by_size,
    _delete_one_verified,
    _transcode_one,
    _verify_bit_exact,
)


# ---------------------------------------------------------------------------
# _transcode_one() / _verify_bit_exact() — pure I/O logic, no catalog needed.
# ---------------------------------------------------------------------------

def _write_wav(path: Path, seconds: float = 1.0, sr: int = 48000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clip = rng.normal(scale=0.2, size=int(sr * seconds)).astype(np.float32)
    sf.write(str(path), clip, sr, format="WAV", subtype="PCM_16")
    return clip


def test_transcode_one_produces_verified_bit_exact_flac(tmp_path):
    wav_path = tmp_path / "raw1.wav"
    _write_wav(wav_path)

    result = _transcode_one("raw1", str(wav_path))

    assert result["verified"] is True
    assert result["provenance"] == "raw_flac"
    flac_path = Path(result["flac_path"])
    assert flac_path.exists()
    assert flac_path.suffix == ".flac"

    original, sr1 = sf.read(str(wav_path))
    transcoded, sr2 = sf.read(str(flac_path))
    assert sr1 == sr2
    np.testing.assert_array_equal(original, transcoded)


def test_transcode_one_missing_source_fails_cleanly(tmp_path):
    result = _transcode_one("missing", str(tmp_path / "does_not_exist.wav"))

    assert result["verified"] is False
    assert result["provenance"] == "transcode_failed"
    assert result["flac_path"] is None
    # no partial .flac left behind
    assert not (tmp_path / "does_not_exist.flac").exists()


def test_verify_bit_exact_true_for_matching_pair(tmp_path):
    # int16-grid-quantized input: writing the SAME arbitrary float clip
    # independently to WAV and FLAC can differ by 1 LSB (libsndfile's two
    # encoders round float->int16 differently -- see tests/test_audio_bus.py
    # for the same finding). Real production transcode never hits this
    # (one encode pass, not two independent ones), but this synthetic test
    # writes both containers separately, so it needs grid-aligned input.
    rng = np.random.default_rng(1)
    pcm16 = rng.integers(-32768, 32767, size=48000, dtype=np.int16)
    clip = (pcm16.astype(np.float32) / 32768.0)

    wav_path = tmp_path / "a.wav"
    sf.write(str(wav_path), clip, 48000, format="WAV", subtype="PCM_16")
    flac_path = tmp_path / "a.flac"
    sf.write(str(flac_path), clip, 48000, format="FLAC", subtype="PCM_16")

    ok, duration, err = _verify_bit_exact(str(wav_path), str(flac_path))
    assert ok is True
    assert err is None
    assert duration == pytest.approx(1.0, abs=1e-3)


def test_verify_bit_exact_false_for_different_content(tmp_path):
    wav_path = tmp_path / "b.wav"
    _write_wav(wav_path, seed=2)

    flac_path = tmp_path / "b.flac"
    different_clip = _write_wav(tmp_path / "_scratch.wav", seed=3)
    sf.write(str(flac_path), different_clip, 48000, format="FLAC", subtype="PCM_16")

    ok, duration, err = _verify_bit_exact(str(wav_path), str(flac_path))
    assert ok is False
    assert err is not None


# ---------------------------------------------------------------------------
# _cap_by_size() — greedy accumulation by actual on-disk file size.
# ---------------------------------------------------------------------------

def test_cap_by_size_stops_at_budget(tmp_path):
    rows = []
    for i in range(5):
        p = tmp_path / f"f{i}.wav"
        p.write_bytes(b"x" * (200 * 1024 * 1024))  # 200 MiB each
        rows.append((f"raw{i}", str(p)))

    capped = _cap_by_size(rows, batch_gb=0.5)  # 512 MiB budget

    # 200MiB, 400MiB (>=512MiB budget reached) -> stops after the 3rd item
    assert len(capped) == 3


def test_cap_by_size_no_budget_hit_returns_all(tmp_path):
    rows = []
    for i in range(3):
        p = tmp_path / f"g{i}.wav"
        p.write_bytes(b"x" * 1024)
        rows.append((f"raw{i}", str(p)))

    capped = _cap_by_size(rows, batch_gb=100.0)
    assert len(capped) == 3


# ---------------------------------------------------------------------------
# Discovery SQL — isolated scratch DuckDB (schema only, no live catalog data)
# so eligibility rules can be checked precisely without touching production.
# ---------------------------------------------------------------------------

@pytest.fixture
def scratch_conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "scratch.duckdb"))
    init_schema(conn)
    yield conn
    conn.close()


def test_discover_transcode_eligibility_rules(scratch_conn):
    conn = scratch_conn

    # segmented WAV -> eligible
    conn.execute(
        "INSERT INTO raw_files (raw_id, wav_path, source) VALUES ('seg_wav', '/x/a.wav', 'rthk')"
    )
    conn.execute(
        "INSERT INTO raw_segments (raw_id, provenance) VALUES ('seg_wav', 'legacy_reused')"
    )

    # rejected WAV, never segmented -> eligible via lang_screen reject path
    conn.execute(
        "INSERT INTO raw_files (raw_id, wav_path, source) VALUES ('reject_wav', '/x/b.wav', 'youtube')"
    )
    conn.execute(
        "INSERT INTO lang_screen (raw_id, decision) VALUES ('reject_wav', 'reject')"
    )

    # native container (webm), segmented -> NEVER eligible regardless of raw_segments
    conn.execute(
        "INSERT INTO raw_files (raw_id, wav_path, source) VALUES ('native_seg', '/x/c.webm', 'youtube')"
    )
    conn.execute(
        "INSERT INTO raw_segments (raw_id, provenance) VALUES ('native_seg', 'segment_vad_cut')"
    )

    # neither segmented nor rejected -> NOT eligible yet
    conn.execute(
        "INSERT INTO raw_files (raw_id, wav_path, source) VALUES ('untouched', '/x/d.wav', 'podcast')"
    )

    # already transcoded -> excluded even though it's segmented
    conn.execute(
        "INSERT INTO raw_files (raw_id, wav_path, source) VALUES ('done', '/x/e.wav', 'rthk')"
    )
    conn.execute(
        "INSERT INTO raw_segments (raw_id, provenance) VALUES ('done', 'legacy_reused')"
    )
    conn.execute(
        "INSERT INTO raw_flac (raw_id, verified) VALUES ('done', true)"
    )

    rows = conn.execute(TRANSCODE_DISCOVER_SQL).fetchall()
    ids = {r[0] for r in rows}

    assert ids == {"seg_wav", "reject_wav"}


# ---------------------------------------------------------------------------
# _delete_one_verified() — transactional catalog update + physical delete.
# ---------------------------------------------------------------------------

def test_delete_one_verified_updates_catalog_and_removes_wav(scratch_conn, tmp_path):
    conn = scratch_conn
    wav_path = tmp_path / "raw.wav"
    flac_path = tmp_path / "raw.flac"
    wav_path.write_bytes(b"fake wav content")
    flac_path.write_bytes(b"fake flac content")

    conn.execute(
        "INSERT INTO raw_files (raw_id, wav_path, source) VALUES (?, ?, 'rthk')",
        ["r1", str(wav_path)],
    )
    conn.execute(
        "INSERT INTO raw_flac (raw_id, flac_path, verified) VALUES (?, ?, true)",
        ["r1", str(flac_path)],
    )

    raw_id, ok, err = _delete_one_verified(conn, "r1", str(flac_path))

    assert ok is True
    assert err is None
    assert not wav_path.exists()
    assert flac_path.exists()  # only the .wav is ever deleted

    new_wav_path = conn.execute(
        "SELECT wav_path FROM raw_files WHERE raw_id = 'r1'"
    ).fetchone()[0]
    assert new_wav_path == str(flac_path)

    wav_deleted_at = conn.execute(
        "SELECT wav_deleted_at FROM raw_flac WHERE raw_id = 'r1'"
    ).fetchone()[0]
    assert wav_deleted_at is not None


def test_delete_one_verified_missing_raw_id_fails_without_side_effects(scratch_conn, tmp_path):
    conn = scratch_conn
    raw_id, ok, err = _delete_one_verified(conn, "nonexistent", str(tmp_path / "x.flac"))

    assert ok is False
    assert "not found" in err
