import json
from pathlib import Path

import duckdb
import numpy as np
import pytest
import soundfile as sf

from pipeline.catalog.catalog import init_schema
from pipeline.nodes.recover_orphans import (
    HIGH_AGREEMENT_THRESHOLD,
    build_recovery_rows,
    classify_one,
    discover,
)


# ---------------------------------------------------------------------------
# classify_one() -- pure sidecar-reading logic, no catalog needed.
# ---------------------------------------------------------------------------

def test_classify_pregate_pass(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"fake")
    (tmp_path / "a.pregate.json").write_text(json.dumps({"snr": 30.0, "pass": True}))

    result = classify_one(str(wav))
    assert result["bucket"] == "pregate_pass"
    assert result["recover"] is True


def test_classify_pregate_fail_records_reason(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"fake")
    (tmp_path / "a.pregate.json").write_text(json.dumps({"snr": 10.0, "pass": False, "reason": "snr"}))

    result = classify_one(str(wav))
    assert result["bucket"] == "pregate_fail_snr"
    assert result["recover"] is False


def test_classify_transcript_high_agreement(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"fake")
    (tmp_path / "a.transcript.json").write_text(json.dumps({
        "asr_agreement": HIGH_AGREEMENT_THRESHOLD,
        "text": "hello", "text_verified": False, "asr_candidates": [],
    }))

    result = classify_one(str(wav))
    assert result["bucket"] == "transcript_high_agreement"
    assert result["recover"] is True


def test_classify_transcript_low_agreement(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"fake")
    (tmp_path / "a.transcript.json").write_text(json.dumps({
        "asr_agreement": HIGH_AGREEMENT_THRESHOLD - 0.01,
        "text": "hello", "text_verified": False, "asr_candidates": [],
    }))

    result = classify_one(str(wav))
    assert result["bucket"] == "transcript_low_agreement"
    assert result["recover"] is False


def test_classify_pregate_takes_priority_over_transcript(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"fake")
    (tmp_path / "a.pregate.json").write_text(json.dumps({"pass": True}))
    (tmp_path / "a.transcript.json").write_text(json.dumps({
        "asr_agreement": 0.0, "text": "x", "text_verified": False, "asr_candidates": [],
    }))

    result = classify_one(str(wav))
    assert result["bucket"] == "pregate_pass"
    assert result["recover"] is True
    assert result["transcript"] is not None  # still surfaced for downstream ASR backfill


def test_classify_no_sidecar_at_all(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"fake")

    result = classify_one(str(wav))
    assert result["bucket"] == "no_sidecar_at_all"
    assert result["recover"] is False


def test_classify_unreadable_sidecar_treated_as_absent(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"fake")
    (tmp_path / "a.pregate.json").write_text("{not valid json")

    result = classify_one(str(wav))
    assert result["bucket"] == "no_sidecar_at_all"
    assert result["recover"] is False


# ---------------------------------------------------------------------------
# build_recovery_rows() -- row-shape construction for a RECOVER orphan.
# ---------------------------------------------------------------------------

def test_build_recovery_rows_with_transcript():
    transcript = {
        "asr_agreement": 0.91,
        "text": "hello world",
        "text_verified": False,
        "asr_candidates": [
            {"model": "model_a", "text": "hello world", "confidence": 0.9},
            {"model": "model_b", "text": "hello wold", "confidence": 0.7},
        ],
    }
    built = build_recovery_rows("seg1", "/x/a.wav", "podcast", transcript, 5.0, 48000, "2026-07-06")

    assert built["segments_row"]["id"] == "seg1"
    assert built["segments_row"]["audio_path"] == "/x/a.wav"
    assert built["segments_row"]["duration_sec"] == 5.0
    assert built["segments_row"]["raw_id"] is None

    assert built["asr_agreement_row"]["agreement"] == 0.91
    assert built["asr_agreement_row"]["best_text"] == "hello world"
    assert built["asr_agreement_row"]["text_verified"] is False

    assert len(built["asr_results_rows"]) == 2
    assert built["asr_results_rows"][0]["model"] == "model_a"


def test_build_recovery_rows_without_transcript():
    built = build_recovery_rows("seg1", "/x/a.wav", "podcast", None, 5.0, 48000, "2026-07-06")

    assert built["asr_agreement_row"] is None
    assert built["asr_results_rows"] == []


# ---------------------------------------------------------------------------
# discover() -- excludes catalog-known and already-classified paths.
# ---------------------------------------------------------------------------

@pytest.fixture
def scratch_conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "scratch.duckdb"))
    init_schema(conn)
    yield conn
    conn.close()


def test_discover_excludes_catalog_known_and_already_classified(scratch_conn, tmp_path, monkeypatch):
    import pipeline.nodes.recover_orphans as recover_orphans

    seg_root = tmp_path / "segments"
    (seg_root / "podcast").mkdir(parents=True)
    monkeypatch.setattr(recover_orphans, "_segments_root", lambda: seg_root)

    wav_kept = seg_root / "podcast" / "kept.wav"
    wav_classified = seg_root / "podcast" / "classified.wav"
    wav_new = seg_root / "podcast" / "new.wav"
    for p in (wav_kept, wav_classified, wav_new):
        p.write_bytes(b"fake")

    conn = scratch_conn
    conn.execute(
        "INSERT INTO segments (id, audio_path, source) VALUES ('s1', ?, 'podcast')",
        [str(wav_kept)],
    )
    conn.execute(
        "INSERT INTO orphan_segments (audio_path, source, bucket, bytes, status) "
        "VALUES (?, 'podcast', 'transcript_low_agreement', 4, 'pending_delete')",
        [str(wav_classified)],
    )

    rows = discover(conn)
    paths = {p for _, p in rows}
    assert paths == {str(wav_new)}


# ---------------------------------------------------------------------------
# run_recover_orphans() -- end-to-end on a tiny synthetic tree.
# ---------------------------------------------------------------------------

def _write_wav(path: Path, seconds: float = 2.0, sr: int = 48000) -> None:
    rng = np.random.default_rng(0)
    clip = rng.normal(scale=0.1, size=int(sr * seconds)).astype(np.float32)
    sf.write(str(path), clip, sr, subtype="PCM_16")


def test_run_recover_orphans_end_to_end(scratch_conn, tmp_path, monkeypatch):
    import asyncio

    import pipeline.nodes.recover_orphans as recover_orphans

    seg_root = tmp_path / "segments"
    (seg_root / "podcast").mkdir(parents=True)
    monkeypatch.setattr(recover_orphans, "_segments_root", lambda: seg_root)
    monkeypatch.setattr("pipeline.catalog.catalog.connect", lambda: scratch_conn)

    good_wav = seg_root / "podcast" / "good_seg00000.wav"
    bad_wav = seg_root / "podcast" / "bad_seg00000.wav"
    _write_wav(good_wav)
    _write_wav(bad_wav)
    (seg_root / "podcast" / "good_seg00000.pregate.json").write_text(json.dumps({"pass": True}))
    (seg_root / "podcast" / "bad_seg00000.pregate.json").write_text(
        json.dumps({"pass": False, "reason": "snr"})
    )

    result = asyncio.run(recover_orphans.run_recover_orphans())

    assert result["scanned"] == 2
    assert result["recovered"] == 1
    assert result["pending_delete"] == 1

    seg_rows = scratch_conn.execute("SELECT audio_path FROM segments").fetchall()
    assert (str(good_wav),) in seg_rows
    assert not any(r[0] == str(bad_wav) for r in seg_rows)

    orphan_rows = dict(
        scratch_conn.execute("SELECT audio_path, status FROM orphan_segments").fetchall()
    )
    assert orphan_rows[str(good_wav)] == "recovered"
    assert orphan_rows[str(bad_wav)] == "pending_delete"
