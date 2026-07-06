from pathlib import Path

import duckdb
import pytest

from pipeline.catalog.catalog import init_schema
from pipeline.nodes.rebalance import (
    DISCOVER_SQL,
    _copy_one,
    _delete_one_verified,
    _target,
    _verify_copy,
    discover_copy,
)


# ---------------------------------------------------------------------------
# _target() — shard-key resolution (raw_id when present, else the segment id).
# ---------------------------------------------------------------------------

def test_target_prefers_raw_id_over_segment_id(monkeypatch):
    calls = []

    # _target imports shard_index/shard_root locally from config.storage_layout
    # on every call, so patching that module's attributes is what takes effect.
    import config.storage_layout as storage_layout
    monkeypatch.setattr(storage_layout, "shard_index", lambda key: calls.append(key) or 0)
    monkeypatch.setattr(storage_layout, "shard_root", lambda key: Path("/shard0"))

    _target("seg1", "/old/podcast/a.flac", "raw1", "podcast")
    assert calls == ["raw1"]

    calls.clear()
    _target("seg2", "/old/podcast/b.flac", None, "podcast")
    assert calls == ["seg2"]


def test_target_preserves_filename_and_source_subdir(monkeypatch):
    import config.storage_layout as storage_layout
    monkeypatch.setattr(storage_layout, "shard_index", lambda key: 1)
    monkeypatch.setattr(storage_layout, "shard_root", lambda key: Path("/mnt/Drive3/canto/segments"))

    idx, new_path = _target("seg1", "/mnt/Drive4/canto/segments/podcast/x_seg00000.flac", "raw1", "podcast")
    assert idx == 1
    assert new_path == "/mnt/Drive3/canto/segments/podcast/x_seg00000.flac"


# ---------------------------------------------------------------------------
# _copy_one() / _verify_copy() — pure I/O logic, no catalog needed.
# ---------------------------------------------------------------------------

def test_copy_one_produces_verified_byte_identical_copy(tmp_path):
    old_path = tmp_path / "old" / "a.flac"
    old_path.parent.mkdir()
    old_path.write_bytes(b"fake flac content" * 1000)
    new_path = tmp_path / "new" / "a.flac"

    result = _copy_one("seg1", str(old_path), str(new_path), 1)

    assert result["verified"] is True
    assert result["provenance"] == "rebalance"
    assert new_path.exists()
    assert new_path.read_bytes() == old_path.read_bytes()
    assert old_path.exists()  # source is never touched by the copy pass


def test_copy_one_missing_source_fails_cleanly(tmp_path):
    new_path = tmp_path / "new" / "a.flac"

    result = _copy_one("seg1", str(tmp_path / "does_not_exist.flac"), str(new_path), 0)

    assert result["verified"] is False
    assert result["provenance"] == "copy_failed"
    assert result["new_path"] is None
    assert not new_path.exists()


def test_verify_copy_true_for_identical_files(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"same content" * 500)
    b.write_bytes(b"same content" * 500)

    ok, err = _verify_copy(str(a), str(b))
    assert ok is True
    assert err is None


def test_verify_copy_false_for_different_size(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"x" * 100)
    b.write_bytes(b"x" * 50)

    ok, err = _verify_copy(str(a), str(b))
    assert ok is False
    assert "size" in err


def test_verify_copy_false_for_same_size_different_content(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"a" * 100)
    b.write_bytes(b"b" * 100)

    ok, err = _verify_copy(str(a), str(b))
    assert ok is False
    assert "checksum" in err


# ---------------------------------------------------------------------------
# Discovery SQL — isolated scratch DuckDB (schema only, no live catalog data).
# ---------------------------------------------------------------------------

@pytest.fixture
def scratch_conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "scratch.duckdb"))
    init_schema(conn)
    yield conn
    conn.close()


def test_discover_copy_excludes_already_migrated(scratch_conn, monkeypatch):
    import config.storage_layout as storage_layout
    monkeypatch.setattr(storage_layout, "shard_index", lambda key: 0)
    monkeypatch.setattr(storage_layout, "shard_root", lambda key: Path("/mnt/Drive2/canto/segments"))

    conn = scratch_conn
    conn.execute(
        "INSERT INTO segments (id, audio_path, source, raw_id) VALUES "
        "('s1', '/mnt/Drive4/canto/segments/podcast/a.flac', 'podcast', 'r1')"
    )
    conn.execute(
        "INSERT INTO segments (id, audio_path, source, raw_id) VALUES "
        "('s2', '/mnt/Drive4/canto/segments/podcast/b.flac', 'podcast', 'r2')"
    )
    conn.execute(
        "INSERT INTO segment_shard_migrations (id, old_path, new_path, target_shard, verified) "
        "VALUES ('s2', '/mnt/Drive4/canto/segments/podcast/b.flac', "
        "'/mnt/Drive2/canto/segments/podcast/b.flac', 0, true)"
    )

    rows = discover_copy(conn)
    ids = {r[0] for r in rows}
    assert ids == {"s1"}


def test_discover_copy_marks_already_in_place_by_matching_path(scratch_conn, monkeypatch):
    import config.storage_layout as storage_layout
    # Segment s1 already lives at the exact path shard_root() would compute.
    monkeypatch.setattr(storage_layout, "shard_index", lambda key: 2)
    monkeypatch.setattr(storage_layout, "shard_root", lambda key: Path("/mnt/Drive4/canto/segments"))

    conn = scratch_conn
    conn.execute(
        "INSERT INTO segments (id, audio_path, source, raw_id) VALUES "
        "('s1', '/mnt/Drive4/canto/segments/podcast/a.flac', 'podcast', 'r1')"
    )

    rows = discover_copy(conn)
    assert len(rows) == 1
    seg_id, old_path, new_path, idx = rows[0]
    assert seg_id == "s1"
    assert old_path == new_path  # signals "already on target shard" to run_rebalance_copy


# ---------------------------------------------------------------------------
# _delete_one_verified() — transactional catalog update + physical delete.
# ---------------------------------------------------------------------------

def test_delete_one_verified_updates_catalog_and_removes_old_copy(scratch_conn, tmp_path):
    conn = scratch_conn
    old_path = tmp_path / "old.flac"
    new_path = tmp_path / "new.flac"
    old_path.write_bytes(b"old copy")
    new_path.write_bytes(b"new copy")

    conn.execute(
        "INSERT INTO segments (id, audio_path, source) VALUES (?, ?, 'podcast')",
        ["seg1", str(old_path)],
    )
    conn.execute(
        "INSERT INTO segment_shard_migrations (id, old_path, new_path, target_shard, verified) "
        "VALUES (?, ?, ?, 1, true)",
        ["seg1", str(old_path), str(new_path)],
    )

    seg_id, ok, err = _delete_one_verified(conn, "seg1", str(new_path))

    assert ok is True
    assert err is None
    assert not old_path.exists()
    assert new_path.exists()  # only the old copy is ever deleted

    new_audio_path = conn.execute(
        "SELECT audio_path FROM segments WHERE id = 'seg1'"
    ).fetchone()[0]
    assert new_audio_path == str(new_path)

    migrated_at = conn.execute(
        "SELECT migrated_at FROM segment_shard_migrations WHERE id = 'seg1'"
    ).fetchone()[0]
    assert migrated_at is not None


def test_delete_one_verified_missing_id_fails_without_side_effects(scratch_conn, tmp_path):
    conn = scratch_conn
    seg_id, ok, err = _delete_one_verified(conn, "nonexistent", str(tmp_path / "x.flac"))

    assert ok is False
    assert "not found" in err
