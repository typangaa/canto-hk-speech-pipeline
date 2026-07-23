import json

import duckdb
import pytest

from pipeline.catalog.catalog import init_schema
from pipeline.nodes.label_store import _SKIP_SENTINEL, build_label_rows, bucket_pitch, bucket_rate


def test_bucket_rate_slow_normal_fast():
    assert bucket_rate(3.0, p25=4.0, p75=5.0) == "slow"
    assert bucket_rate(4.5, p25=4.0, p75=5.0) == "normal"
    assert bucket_rate(6.0, p25=4.0, p75=5.0) == "fast"


def test_bucket_rate_boundary_values_are_normal():
    assert bucket_rate(4.0, p25=4.0, p75=5.0) == "normal"
    assert bucket_rate(5.0, p25=4.0, p75=5.0) == "normal"


def test_bucket_pitch_low_normal_high():
    z, bucket = bucket_pitch(100.0, mu=150.0, sigma=20.0)
    assert bucket == "low"
    assert z < -0.5

    z, bucket = bucket_pitch(150.0, mu=150.0, sigma=20.0)
    assert bucket == "normal"
    assert z == 0.0

    z, bucket = bucket_pitch(200.0, mu=150.0, sigma=20.0)
    assert bucket == "high"
    assert z > 0.5


def test_bucket_pitch_sigma_zero_does_not_crash():
    z, bucket = bucket_pitch(160.0, mu=150.0, sigma=0.0)
    assert isinstance(z, float)
    assert bucket in {"low", "normal", "high"}


# ---------------------------------------------------------------------------
# P3 pause-token addition -- control.pause.plan / calibration_version / unalignable
# ---------------------------------------------------------------------------

_CALIBRATION = {
    "rate": {"p25": 4.0, "p75": 5.0},
    "pitch": {"per_speaker": {}, "corpus_fallback": {"mu": 150.0, "sigma": 20.0}},
    "version": "rate-pitch-v1",
}


@pytest.fixture
def scratch_conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "scratch.duckdb"))
    init_schema(conn)
    yield conn
    conn.close()


def _seed_segment_with_lang(conn, seg_id):
    conn.execute(
        "INSERT INTO segments (id, audio_path, source, duration_sec, sample_rate) "
        "VALUES (?, '/tmp/x.flac', 'podcast', 6.0, 48000)",
        [seg_id],
    )
    conn.execute("INSERT INTO labels_lang (id, lang, cmn_prob) VALUES (?, 'yue', 0.01)", [seg_id])


def test_build_label_rows_includes_pause_plan_when_present(scratch_conn):
    conn = scratch_conn
    _seed_segment_with_lang(conn, "ag1")
    plan = [{"offset": 2, "mark": "，", "kind": "normal", "delta_t": 0.5, "verdict": "long"}]
    conn.execute(
        "INSERT INTO pause_plan (id, plan, n_punct, n_no_pause, n_short, n_long, "
        "unalignable, calibration_version, provenance) "
        "VALUES ('ag1', ?, 1, 0, 0, 1, FALSE, 'pause-v1', 'pause_plan')",
        [json.dumps(plan)],
    )

    records = [r for r in build_label_rows(conn, _CALIBRATION) if r is not _SKIP_SENTINEL]
    assert len(records) == 1
    pause = records[0]["control"]["pause"]
    assert pause["plan"] == plan
    assert pause["calibration_version"] == "pause-v1"
    assert "unalignable" not in pause


def test_build_label_rows_pause_plan_unalignable_flag_present_when_true(scratch_conn):
    conn = scratch_conn
    _seed_segment_with_lang(conn, "ag2")
    conn.execute(
        "INSERT INTO pause_plan (id, plan, n_punct, n_no_pause, n_short, n_long, "
        "unalignable, calibration_version, provenance) "
        "VALUES ('ag2', '[]', 0, 0, 0, 0, TRUE, 'pause-v1', 'pause_plan')"
    )

    records = [r for r in build_label_rows(conn, _CALIBRATION) if r is not _SKIP_SENTINEL]
    pause = records[0]["control"]["pause"]
    assert pause["plan"] == []
    assert pause["unalignable"] is True


def test_build_label_rows_omits_pause_when_no_pause_plan_row(scratch_conn):
    conn = scratch_conn
    _seed_segment_with_lang(conn, "sv1")

    records = [r for r in build_label_rows(conn, _CALIBRATION) if r is not _SKIP_SENTINEL]
    assert "pause" not in records[0].get("control", {})
