import asyncio

import duckdb
import pytest

from pipeline.catalog.catalog import init_schema, upsert_rows
from pipeline.cli import split_run_many_groups
from pipeline.nodes.asr import run_asr_agreement, run_asr_transcribe
from pipeline.nodes.filter import run_filter_acoustic
from pipeline.nodes.label_music import run_label_music
from pipeline.nodes.segment import run_segment_diarize


@pytest.fixture
def scratch_conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "scratch.duckdb"))
    init_schema(conn)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# split_run_many_groups — pure argv-splitting logic
# ---------------------------------------------------------------------------

def test_split_run_many_groups_two_nodes():
    tokens = ["segment.diarize", "--devices", "cuda:0", "--",
              "filter.acoustic", "--workers", "2"]
    assert split_run_many_groups(tokens) == [
        ["segment.diarize", "--devices", "cuda:0"],
        ["filter.acoustic", "--workers", "2"],
    ]


def test_split_run_many_groups_no_separator_is_single_group():
    assert split_run_many_groups(["filter.acoustic", "--workers", "2"]) == [
        ["filter.acoustic", "--workers", "2"],
    ]


def test_split_run_many_groups_empty_input():
    assert split_run_many_groups([]) == []


def test_split_run_many_groups_ignores_stray_trailing_separator():
    tokens = ["segment.diarize", "--", "filter.acoustic", "--workers", "2", "--"]
    assert split_run_many_groups(tokens) == [
        ["segment.diarize"],
        ["filter.acoustic", "--workers", "2"],
    ]


# ---------------------------------------------------------------------------
# conn injection — an injected conn must be used as-is, connect() must never
# be called (regression guard for the run-many dependency-injection change).
# ---------------------------------------------------------------------------

def test_run_filter_acoustic_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_filter_acoustic must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_filter_acoustic(conn=scratch_conn, n_workers=1))

    assert result == {"processed": 0, "errors": 0}


def test_run_segment_diarize_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_segment_diarize must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_segment_diarize(["cuda:0"], conn=scratch_conn))

    assert result == {"reused": 0, "gpu_computed": 0, "errors": 0}


def test_run_label_music_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_label_music must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_label_music(["cuda:0"], conn=scratch_conn))

    assert result == {"processed": 0, "errors": 0}


def test_run_asr_transcribe_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_asr_transcribe must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_asr_transcribe([("canto_ft", "cuda:0")], conn=scratch_conn))

    assert result == {"processed": 0, "errors": 0}


def test_run_asr_agreement_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_asr_agreement must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_asr_agreement(conn=scratch_conn))

    assert result == {"processed": 0, "errors": 0}


def test_run_filter_acoustic_defaults_to_self_connect(monkeypatch, scratch_conn):
    """Standalone `pipe run filter.acoustic` (no conn passed) must keep
    self-connecting exactly as before — the conn=None default path."""
    monkeypatch.setattr("pipeline.catalog.catalog.connect", lambda: scratch_conn)

    result = asyncio.run(run_filter_acoustic(n_workers=1))

    assert result == {"processed": 0, "errors": 0}


# ---------------------------------------------------------------------------
# Real concurrency proof: two coroutines, each with its own conn.cursor() on
# one shared connection, writing to different tables at once — mirrors the
# throwaway verification that motivated docs/ORCHESTRATOR_PLAN.md.
# ---------------------------------------------------------------------------

def test_two_nodes_write_concurrently_via_cursor_per_coroutine(scratch_conn):
    async def _write_filters_text(conn, n):
        rows = [{"id": f"a{i}", "sample_rate_ok": True, "duration_ok": True,
                  "length_ok": True, "eng_ratio_ok": True, "mandarin_ratio_ok": True,
                  "pass": True} for i in range(n)]
        for row in rows:
            conn.cursor().execute(
                "INSERT INTO filters_text (id, pass) VALUES (?, ?)",
                [row["id"], row["pass"]],
            )
        return n

    async def _write_filters_acoustic(conn, n):
        rows = [{"id": f"b{i}", "snr_db": 30.0, "dnsmos_sig": 3.5,
                  "dnsmos_ovrl": 3.5, "pass": True, "fail_reason": None} for i in range(n)]
        upsert_rows(conn.cursor(), "filters_acoustic", rows, ["id"])
        return n

    async def _run_both():
        return await asyncio.gather(
            _write_filters_text(scratch_conn, 200),
            _write_filters_acoustic(scratch_conn, 200),
        )

    results = asyncio.run(_run_both())
    assert results == [200, 200]

    text_count = scratch_conn.execute("SELECT count(*) FROM filters_text").fetchone()[0]
    acoustic_count = scratch_conn.execute("SELECT count(*) FROM filters_acoustic").fetchone()[0]
    assert text_count == 200
    assert acoustic_count == 200
