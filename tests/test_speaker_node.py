import asyncio

import duckdb
import numpy as np
import pytest

import pipeline.nodes.speaker as speaker
from pipeline.catalog.catalog import init_schema
from pipeline.nodes.speaker import (
    _check_sidecar,
    _load_npy,
    cluster_embeddings,
    discover_stale_embed,
    run_speaker_embed,
)


# ---------------------------------------------------------------------------
# cluster_embeddings() — ported verbatim from scripts/08_speaker_id.py
# ---------------------------------------------------------------------------

def test_cluster_embeddings_single_point():
    emb = np.array([[1.0, 0.0, 0.0]])
    labels = cluster_embeddings(emb, "test")
    assert list(labels) == [0]


def test_cluster_embeddings_two_clear_clusters():
    # Two tight groups of vectors, far apart in cosine distance.
    rng = np.random.default_rng(0)
    a = np.array([1.0, 0.0, 0.0]) + rng.normal(scale=0.01, size=(5, 3))
    b = np.array([0.0, 1.0, 0.0]) + rng.normal(scale=0.01, size=(5, 3))
    emb = np.vstack([a, b])
    labels = cluster_embeddings(emb, "test", threshold=0.25)
    assert len(set(labels[:5])) == 1
    assert len(set(labels[5:])) == 1
    assert labels[0] != labels[5]


def test_cluster_embeddings_scalable_sample_and_assign_path(monkeypatch):
    """Above _CLUSTER_SAMPLE_MAX, clustering falls back to sample-then-assign.
    Every point still gets a label, and points identical to the sampled
    centroid seed should be assigned to that centroid's cluster."""
    monkeypatch.setattr(speaker, "_CLUSTER_SAMPLE_MAX", 20)
    rng = np.random.default_rng(0)
    a = np.tile([1.0, 0.0, 0.0], (15, 1)) + rng.normal(scale=0.01, size=(15, 3))
    b = np.tile([0.0, 1.0, 0.0], (15, 1)) + rng.normal(scale=0.01, size=(15, 3))
    emb = np.vstack([a, b])  # 30 rows > 20 sample cap
    labels = cluster_embeddings(emb, "test", threshold=0.25)
    assert len(labels) == 30
    assert len(set(labels[:15])) == 1
    assert len(set(labels[15:])) == 1
    assert labels[0] != labels[15]


def test_cluster_embeddings_returns_int_array():
    emb = np.eye(3)
    labels = cluster_embeddings(emb, "test")
    assert labels.dtype.kind in ("i", "u")


# ---------------------------------------------------------------------------
# _check_sidecar() — legacy .embed.npy reuse check (I/O, tmp_path)
# ---------------------------------------------------------------------------

def test_check_sidecar_hit(tmp_path):
    wav = tmp_path / "seg00001.wav"
    wav.write_bytes(b"")
    sidecar = tmp_path / "seg00001.embed.npy"
    np.save(str(sidecar), np.zeros(192, dtype=np.float32))

    seg_id, source, ref = _check_sidecar(("id1", "rthk", str(wav)))
    assert seg_id == "id1"
    assert source == "rthk"
    assert ref == str(sidecar)


def test_check_sidecar_miss(tmp_path):
    wav = tmp_path / "seg00002.wav"
    wav.write_bytes(b"")

    seg_id, source, ref = _check_sidecar(("id2", "rthk", str(wav)))
    assert ref is None


# ---------------------------------------------------------------------------
# _load_npy() — embedding load helper, must not raise on a bad/missing file
# ---------------------------------------------------------------------------

def test_load_npy_success(tmp_path):
    ref = tmp_path / "emb.npy"
    arr = np.arange(10, dtype=np.float32)
    np.save(str(ref), arr)

    seg_id, out_ref, loaded = _load_npy(("id1", str(ref)))
    assert seg_id == "id1"
    assert out_ref == str(ref)
    assert np.array_equal(loaded, arr)


def test_load_npy_missing_file_returns_none(tmp_path):
    ref = tmp_path / "does_not_exist.npy"
    seg_id, out_ref, loaded = _load_npy(("id1", str(ref)))
    assert loaded is None


def test_load_npy_corrupt_file_returns_none(tmp_path):
    ref = tmp_path / "corrupt.npy"
    ref.write_bytes(b"not a valid npy file")
    seg_id, out_ref, loaded = _load_npy(("id1", str(ref)))
    assert loaded is None


# ---------------------------------------------------------------------------
# discover_stale_embed() / verify_existing repair pass — orphaned
# embedding_ref rows left behind by the filtered/ tree retirement (§7.3)
# ---------------------------------------------------------------------------

@pytest.fixture
def scratch_conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "scratch.duckdb"))
    init_schema(conn)
    yield conn
    conn.close()


def test_discover_stale_embed_returns_all_rows_with_a_ref(scratch_conn):
    conn = scratch_conn
    conn.execute(
        "INSERT INTO segments (id, audio_path, source) VALUES "
        "('s1', '/mnt/Drive3/canto/segments/podcast/a.flac', 'podcast')"
    )
    conn.execute(
        "INSERT INTO speaker_embeddings (id, source, embedding_ref, provenance) VALUES "
        "('s1', 'podcast', '/mnt/Drive4/canto/filtered/podcast/a.embed.npy', 'legacy_reused')"
    )
    rows = discover_stale_embed(conn)
    assert rows == [
        ("s1", "podcast", "/mnt/Drive3/canto/segments/podcast/a.flac",
         "/mnt/Drive4/canto/filtered/podcast/a.embed.npy")
    ]


def test_discover_stale_embed_excludes_null_ref(scratch_conn):
    conn = scratch_conn
    conn.execute(
        "INSERT INTO segments (id, audio_path, source) VALUES "
        "('s1', '/mnt/Drive3/canto/segments/podcast/a.flac', 'podcast')"
    )
    conn.execute(
        "INSERT INTO speaker_embeddings (id, source, embedding_ref, provenance) VALUES "
        "('s1', 'podcast', NULL, 'read_failed')"
    )
    assert discover_stale_embed(conn) == []


def test_run_speaker_embed_verify_existing_requeues_missing_ref(scratch_conn, tmp_path, monkeypatch):
    """A row whose embedding_ref file is gone must be re-queued and picked up
    by the normal reuse pass if a fresh sidecar exists at the segment's
    CURRENT audio_path (simulating a segment that moved shard but whose
    embedding was later regenerated at the new location)."""
    conn = scratch_conn

    current_audio = tmp_path / "a.flac"
    current_audio.write_bytes(b"")
    fresh_sidecar = tmp_path / "a.embed.npy"
    np.save(str(fresh_sidecar), np.zeros(192, dtype=np.float32))

    conn.execute(
        "INSERT INTO segments (id, audio_path, source) VALUES (?, ?, 'podcast')",
        ["s1", str(current_audio)],
    )
    conn.execute(
        "INSERT INTO speaker_embeddings (id, source, embedding_ref, provenance) VALUES "
        "('s1', 'podcast', '/mnt/Drive4/canto/filtered/podcast/a.embed.npy', 'legacy_reused')"
    )

    result = asyncio.run(run_speaker_embed([], conn=conn, verify_existing=True))

    assert result["reused"] == 1
    assert result["gpu_computed"] == 0
    row = conn.execute(
        "SELECT embedding_ref, provenance FROM speaker_embeddings WHERE id = 's1'"
    ).fetchone()
    assert row == (str(fresh_sidecar), "legacy_reused")


def test_run_speaker_embed_no_verify_existing_leaves_stale_ref_untouched(scratch_conn, tmp_path):
    """Default behaviour (verify_existing=False) must not touch existing rows
    at all -- discovery is a pure anti-join, so a stale ref is left as-is."""
    conn = scratch_conn
    audio = tmp_path / "a.flac"
    audio.write_bytes(b"")
    conn.execute(
        "INSERT INTO segments (id, audio_path, source) VALUES (?, ?, 'podcast')",
        ["s1", str(audio)],
    )
    conn.execute(
        "INSERT INTO speaker_embeddings (id, source, embedding_ref, provenance) VALUES "
        "('s1', 'podcast', '/mnt/Drive4/canto/filtered/podcast/a.embed.npy', 'legacy_reused')"
    )

    result = asyncio.run(run_speaker_embed([], conn=conn, verify_existing=False))

    assert result == {"reused": 0, "gpu_computed": 0, "errors": 0}
    row = conn.execute(
        "SELECT embedding_ref FROM speaker_embeddings WHERE id = 's1'"
    ).fetchone()
    assert row == ("/mnt/Drive4/canto/filtered/podcast/a.embed.npy",)
