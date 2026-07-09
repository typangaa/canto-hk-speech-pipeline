import asyncio

import duckdb
import pytest

from pipeline.catalog.catalog import init_schema, upsert_rows
from pipeline.cli import split_run_many_groups
from pipeline.nodes import ingest_download
from pipeline.nodes.asr import run_asr_agreement, run_asr_transcribe
from pipeline.nodes.filter import (
    run_filter_acoustic,
    run_filter_decide,
    run_filter_text,
)
from pipeline.nodes.g2p import run_g2p
from pipeline.nodes.ingest_download import run_ingest_commit
from pipeline.nodes.ingest_probe import run_ingest_probe
from pipeline.nodes.label_music import run_label_music
from pipeline.nodes.label_prosody import run_label_prosody
from pipeline.nodes.label_suite import run_label_suite
from pipeline.nodes.lang_screen import run_lang_screen_auto
from pipeline.nodes.raw_flac import run_raw_flac_delete_verified, run_raw_flac_transcode
from pipeline.nodes.rebalance import run_rebalance_copy, run_rebalance_delete_verified
from pipeline.nodes import recover_orphans
from pipeline.nodes.recover_orphans import run_recover_orphans
from pipeline.nodes.segment import run_pregate_snr, run_segment_diarize, run_segment_vad_cut
from pipeline.nodes.speaker import run_speaker_cluster, run_speaker_embed
from pipeline.nodes.tier import run_tier_assign


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


def test_run_speaker_embed_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_speaker_embed must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_speaker_embed(["cuda:0"], conn=scratch_conn))

    assert result == {"reused": 0, "gpu_computed": 0, "errors": 0}


def test_run_speaker_cluster_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_speaker_cluster must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_speaker_cluster(conn=scratch_conn))

    assert result == {"sources_processed": 0, "total_segments": 0, "total_speakers": 0}


def test_run_lang_screen_auto_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_lang_screen_auto must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_lang_screen_auto(["cuda:0"], conn=scratch_conn))

    assert result == {"processed": 0, "pass": 0, "reject": 0, "mixed": 0, "errors": 0}


def test_run_tier_assign_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_tier_assign must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_tier_assign(conn=scratch_conn))

    assert result == {"processed": 0, "gold": 0, "silver": 0, "excluded": 0, "errors": 0}


def test_run_ingest_commit_uses_injected_conn(scratch_conn, monkeypatch, tmp_path):
    def _boom(*a, **kw):
        raise AssertionError("run_ingest_commit must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)
    monkeypatch.setattr(ingest_download, "METADATA_DIR", tmp_path)
    monkeypatch.setattr(ingest_download, "STAGING_FILE", tmp_path / "ingest_download_staging.jsonl")
    monkeypatch.setattr(ingest_download, "KNOWN_IDS_SNAPSHOT", tmp_path / "raw_files_known_ids.json")

    result = asyncio.run(run_ingest_commit(conn=scratch_conn))

    assert result["committed"] == 0
    assert result["archived_to"] is None


def test_run_filter_text_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_filter_text must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_filter_text(conn=scratch_conn))

    assert result == {"processed": 0, "errors": 0}


def test_run_filter_decide_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_filter_decide must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_filter_decide(conn=scratch_conn))

    assert result == {"processed": 0, "errors": 0}


def test_run_segment_vad_cut_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_segment_vad_cut must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_segment_vad_cut(conn=scratch_conn))

    assert result == {"processed": 0, "total_segments": 0, "errors": 0}


def test_run_pregate_snr_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_pregate_snr must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_pregate_snr(conn=scratch_conn))

    assert result == {"processed": 0, "passed": 0, "failed_snr": 0, "failed_dnsmos": 0, "errors": 0}


def test_run_g2p_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_g2p must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_g2p(conn=scratch_conn))

    assert result == {"processed": 0, "errors": 0}


def test_run_ingest_probe_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_ingest_probe must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_ingest_probe(conn=scratch_conn))

    assert result == {"processed": 0, "errors": 0}


def test_run_label_suite_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_label_suite must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_label_suite(["cuda:0"], conn=scratch_conn))

    assert result == {"processed": 0, "errors": 0}


def test_run_label_prosody_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_label_prosody must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_label_prosody(conn=scratch_conn))

    assert result == {"processed": 0, "errors": 0}


def test_run_recover_orphans_uses_injected_conn(scratch_conn, monkeypatch, tmp_path):
    def _boom(*a, **kw):
        raise AssertionError("run_recover_orphans must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)
    # discover() walks the real SEGMENTS_ROOT on disk (not the catalog) --
    # point it at an empty tmp dir so this test never touches the live,
    # multi-hundred-thousand-file production segments tree.
    monkeypatch.setattr(recover_orphans, "_segments_root", lambda: tmp_path)

    result = asyncio.run(run_recover_orphans(conn=scratch_conn))

    assert result == {"scanned": 0, "recovered": 0, "pending_delete": 0, "errors": 0}


def test_run_rebalance_copy_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_rebalance_copy must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_rebalance_copy(conn=scratch_conn))

    assert result == {"processed": 0, "verified": 0, "failed": 0, "already_in_place": 0}


def test_run_rebalance_delete_verified_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_rebalance_delete_verified must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_rebalance_delete_verified(conn=scratch_conn))

    assert result == {"deleted": 0, "errors": 0}


def test_run_raw_flac_transcode_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_raw_flac_transcode must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_raw_flac_transcode(conn=scratch_conn))

    assert result == {"processed": 0, "verified": 0, "failed": 0}


def test_run_raw_flac_delete_verified_uses_injected_conn(scratch_conn, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_raw_flac_delete_verified must not call connect() when conn is given")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)

    result = asyncio.run(run_raw_flac_delete_verified(conn=scratch_conn))

    assert result == {"deleted": 0, "errors": 0}


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
