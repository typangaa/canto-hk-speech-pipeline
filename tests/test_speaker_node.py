import numpy as np

import pipeline.nodes.speaker as speaker
from pipeline.nodes.speaker import _check_sidecar, _load_npy, cluster_embeddings


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
