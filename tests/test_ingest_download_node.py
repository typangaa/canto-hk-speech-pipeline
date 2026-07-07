import asyncio
import json

import duckdb
import pytest

from pipeline.catalog.catalog import init_schema
import pipeline.nodes.ingest_download as ingest_download
from pipeline.nodes.ingest_download import (
    append_staged_rows,
    load_known_ids_snapshot,
    load_staged_ids,
    run_ingest_commit,
    run_ingest_download,
    write_known_ids_snapshot,
)


@pytest.fixture
def scratch_conn(tmp_path):
    conn = duckdb.connect(str(tmp_path / "scratch.duckdb"))
    init_schema(conn)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def staging_paths(tmp_path, monkeypatch):
    """Redirect every module-level staging path constant to tmp_path so tests
    never touch the real metadata/ directory."""
    monkeypatch.setattr(ingest_download, "METADATA_DIR", tmp_path)
    monkeypatch.setattr(ingest_download, "STAGING_FILE", tmp_path / "ingest_download_staging.jsonl")
    monkeypatch.setattr(ingest_download, "KNOWN_IDS_SNAPSHOT", tmp_path / "raw_files_known_ids.json")
    return tmp_path


# ---------------------------------------------------------------------------
# Snapshot dedup file
# ---------------------------------------------------------------------------

def test_load_known_ids_snapshot_missing_file_returns_empty():
    assert load_known_ids_snapshot() == {}


def test_write_and_load_known_ids_snapshot_roundtrip(scratch_conn):
    scratch_conn.execute(
        "INSERT INTO raw_files (raw_id, wav_path, source) VALUES (?, ?, ?)",
        ["abc123", "/x/a.webm", "youtube"],
    )
    scratch_conn.execute(
        "INSERT INTO raw_files (raw_id, wav_path, source) VALUES (?, ?, ?)",
        ["def456", "/x/b.mp3", "podcast"],
    )

    write_known_ids_snapshot(scratch_conn)
    snapshot = load_known_ids_snapshot()

    assert snapshot["youtube"] == ["abc123"]
    assert snapshot["podcast"] == ["def456"]
    assert snapshot["rthk"] == []


# ---------------------------------------------------------------------------
# Staging file (append-only, no DB)
# ---------------------------------------------------------------------------

def test_append_staged_rows_and_load_staged_ids():
    append_staged_rows([
        {"raw_id": "id1", "source": "podcast", "wav_path": "/x/1.mp3"},
        {"raw_id": "id2", "source": "youtube", "wav_path": "/x/2.webm"},
    ])
    append_staged_rows([
        {"raw_id": "id3", "source": "podcast", "wav_path": "/x/3.mp3"},
    ])

    assert load_staged_ids("podcast") == {"id1", "id3"}
    assert load_staged_ids("youtube") == {"id2"}
    assert load_staged_ids("rthk") == set()


def test_append_staged_rows_empty_list_is_noop(staging_paths):
    append_staged_rows([])
    assert not ingest_download.STAGING_FILE.exists()


# ---------------------------------------------------------------------------
# run_ingest_download() — must NEVER open a DuckDB connection.
# ---------------------------------------------------------------------------

def test_run_ingest_download_never_touches_the_catalog(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("run_ingest_download must not open a DB connection")

    monkeypatch.setattr("pipeline.catalog.catalog.connect", _boom)
    monkeypatch.setattr(
        ingest_download, "discover_active_entries",
        lambda source: [{"name": "Test Show", "type": "rss", "url": "http://example.com/feed"}],
    )
    monkeypatch.setattr(
        ingest_download, "_download_rss_source",
        lambda entry, known_ids, args: [
            {"raw_id": "new1", "source": "podcast", "wav_path": "/x/new1.mp3",
             "downloaded_at": "2026-07-06"},
        ],
    )

    result = asyncio.run(run_ingest_download(source="podcast"))

    assert result == {"staged": 1}
    assert load_staged_ids("podcast") == {"new1"}


def test_run_ingest_download_dedups_against_snapshot_and_staged(monkeypatch, scratch_conn):
    scratch_conn.execute(
        "INSERT INTO raw_files (raw_id, wav_path, source) VALUES (?, ?, ?)",
        ["already_known", "/x/a.mp3", "podcast"],
    )
    write_known_ids_snapshot(scratch_conn)
    append_staged_rows([{"raw_id": "already_staged", "source": "podcast", "wav_path": "/x/b.mp3"}])

    seen_known_ids = {}

    def _fake_download(entry, known_ids, args):
        seen_known_ids["podcast"] = set(known_ids)
        return []

    monkeypatch.setattr(
        ingest_download, "discover_active_entries",
        lambda source: [{"name": "Test Show", "type": "rss", "url": "http://example.com/feed"}],
    )
    monkeypatch.setattr(ingest_download, "_download_rss_source", _fake_download)

    asyncio.run(run_ingest_download(source="podcast"))

    assert seen_known_ids["podcast"] == {"already_known", "already_staged"}


# ---------------------------------------------------------------------------
# run_ingest_commit() — the only DB-touching step.
# ---------------------------------------------------------------------------

def test_run_ingest_commit_end_to_end(monkeypatch, tmp_path):
    catalog_path = tmp_path / "catalog.duckdb"
    conn = duckdb.connect(str(catalog_path))
    init_schema(conn)
    monkeypatch.setattr("pipeline.catalog.catalog.connect", lambda: conn)

    append_staged_rows([
        {"raw_id": "r1", "source": "youtube", "wav_path": "/x/r1.webm",
         "downloaded_at": "2026-07-06"},
        {"raw_id": "r2", "source": "podcast", "wav_path": "/x/r2.mp3",
         "downloaded_at": "2026-07-06"},
    ])

    result = asyncio.run(run_ingest_commit())

    assert result["committed"] == 2
    rows = dict(conn.execute("SELECT raw_id, source FROM raw_files").fetchall())
    assert rows == {"r1": "youtube", "r2": "podcast"}

    # snapshot refreshed
    snapshot = load_known_ids_snapshot()
    assert snapshot["youtube"] == ["r1"]
    assert snapshot["podcast"] == ["r2"]

    # staging file archived, not left in place
    assert not ingest_download.STAGING_FILE.exists()
    assert result["archived_to"] is not None
    archived = list(tmp_path.glob("ingest_download_staging.committed-*.jsonl"))
    assert len(archived) == 1
    with open(archived[0]) as f:
        lines = [json.loads(l) for l in f]
    assert {r["raw_id"] for r in lines} == {"r1", "r2"}


def test_run_ingest_commit_dedups_duplicate_raw_id_last_write_wins(monkeypatch, tmp_path):
    catalog_path = tmp_path / "catalog.duckdb"
    conn = duckdb.connect(str(catalog_path))
    init_schema(conn)
    monkeypatch.setattr("pipeline.catalog.catalog.connect", lambda: conn)

    append_staged_rows([{"raw_id": "dup", "source": "podcast", "wav_path": "/x/old.mp3"}])
    append_staged_rows([{"raw_id": "dup", "source": "podcast", "wav_path": "/x/new.mp3"}])

    result = asyncio.run(run_ingest_commit())

    assert result["committed"] == 1
    row = conn.execute("SELECT wav_path FROM raw_files WHERE raw_id = 'dup'").fetchone()
    assert row[0] == "/x/new.mp3"


def test_run_ingest_commit_nothing_staged_still_refreshes_snapshot(monkeypatch, tmp_path):
    catalog_path = tmp_path / "catalog.duckdb"
    conn = duckdb.connect(str(catalog_path))
    init_schema(conn)
    conn.execute(
        "INSERT INTO raw_files (raw_id, wav_path, source) VALUES (?, ?, ?)",
        ["pre_existing", "/x/p.mp3", "rthk"],
    )
    monkeypatch.setattr("pipeline.catalog.catalog.connect", lambda: conn)

    result = asyncio.run(run_ingest_commit())

    assert result["committed"] == 0
    assert load_known_ids_snapshot()["rthk"] == ["pre_existing"]
