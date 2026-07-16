"""Unit tests for upsert_rows() — the shared bulk-upsert helper used by every
DAG node (and, via record_batch(), the orchestrator journal too). Run:
pytest tests/test_upsert_rows.py -v

Uses a fully in-memory DuckDB connection — independent of the real catalog
(metadata/corpus.duckdb), so these tests never touch the DuckDB single-writer
lock and can run regardless of whether a pipeline node holds it. See
docs/UPSERT_PERFORMANCE_FIX_PLAN.md for the full investigation and design.

Covers both code paths in upsert_rows(): the small-batch conn.executemany()
path (< UPSERT_BULK_THRESHOLD rows) and the bulk DataFrame-register path
(>= UPSERT_BULK_THRESHOLD rows) — every behavioural test below runs against
both paths to confirm they produce identical results.
"""

import json

import duckdb
import pytest

from pipeline.catalog.catalog import UPSERT_BULK_THRESHOLD, upsert_rows

TEST_TABLE_SQL = """
    CREATE TABLE t (
        id VARCHAR PRIMARY KEY,
        n INTEGER,
        label VARCHAR,
        tags JSON,
        meta JSON
    )
"""


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    c.execute(TEST_TABLE_SQL)
    yield c
    c.close()


def make_rows(n: int, *, start: int = 0) -> list[dict]:
    return [
        {
            "id": f"id{start + i:06d}",
            "n": start + i,
            "label": f"row-{start + i}",
            "tags": ["a", "b"] if (start + i) % 2 == 0 else None,
            "meta": {"k": start + i} if (start + i) % 3 == 0 else None,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_empty_rows_returns_zero_and_is_noop(conn):
    assert upsert_rows(conn, "t", [], ["id"]) == 0
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Small batch (executemany path) vs large batch (bulk DataFrame path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 5, UPSERT_BULK_THRESHOLD - 1])
def test_small_batch_uses_executemany_path_and_is_correct(conn, n):
    rows = make_rows(n)
    written = upsert_rows(conn, "t", rows, ["id"])
    assert written == n
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == n


@pytest.mark.parametrize("n", [UPSERT_BULK_THRESHOLD, UPSERT_BULK_THRESHOLD + 1, 5_000])
def test_large_batch_uses_bulk_path_and_is_correct(conn, n):
    rows = make_rows(n)
    written = upsert_rows(conn, "t", rows, ["id"])
    assert written == n
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == n


def test_small_and_large_paths_produce_identical_rows(conn):
    """Write the same logical data twice — once under the threshold (into
    table t) and once over it (into a second table with identical schema) —
    and confirm both paths write byte-identical row content."""
    conn.execute("CREATE TABLE t2 (id VARCHAR PRIMARY KEY, n INTEGER, label VARCHAR, tags JSON, meta JSON)")

    small_rows = make_rows(50)
    upsert_rows(conn, "t", small_rows, ["id"])

    # Pad to cross the bulk threshold, but only compare the first 50 ids.
    large_rows = make_rows(UPSERT_BULK_THRESHOLD + 50)
    upsert_rows(conn, "t2", large_rows, ["id"])

    small = conn.execute("SELECT id, n, label, tags, meta FROM t ORDER BY id").fetchall()
    large_subset = conn.execute(
        "SELECT id, n, label, tags, meta FROM t2 WHERE n < 50 ORDER BY id"
    ).fetchall()
    assert small == large_subset


# ---------------------------------------------------------------------------
# JSON (list/dict) column serialisation
# ---------------------------------------------------------------------------

def test_list_and_dict_columns_json_round_trip_small_path(conn):
    rows = [
        {"id": "a", "n": 1, "label": "x", "tags": ["p", "q"], "meta": {"k": "v"}},
        {"id": "b", "n": 2, "label": "y", "tags": None, "meta": None},
    ]
    upsert_rows(conn, "t", rows, ["id"])
    got = conn.execute("SELECT id, tags, meta FROM t ORDER BY id").fetchall()
    assert json.loads(got[0][1]) == ["p", "q"]
    assert json.loads(got[0][2]) == {"k": "v"}
    assert got[1][1] is None
    assert got[1][2] is None


def test_list_and_dict_columns_json_round_trip_bulk_path(conn):
    rows = make_rows(UPSERT_BULK_THRESHOLD)
    upsert_rows(conn, "t", rows, ["id"])
    # id000000: n=0, even -> tags=["a","b"]; 0 % 3 == 0 -> meta={"k": 0}
    row0 = conn.execute("SELECT tags, meta FROM t WHERE id = 'id000000'").fetchone()
    assert json.loads(row0[0]) == ["a", "b"]
    assert json.loads(row0[1]) == {"k": 0}
    # id000001: n=1, odd -> tags=None; 1 % 3 != 0 -> meta=None
    row1 = conn.execute("SELECT tags, meta FROM t WHERE id = 'id000001'").fetchone()
    assert row1[0] is None
    assert row1[1] is None


# ---------------------------------------------------------------------------
# NULL handling (not the string "None"/"nan")
# ---------------------------------------------------------------------------

def test_none_values_become_sql_null_not_string_bulk_path(conn):
    rows = [
        {"id": f"id{i:06d}", "n": None if i % 2 == 0 else i, "label": None, "tags": None, "meta": None}
        for i in range(UPSERT_BULK_THRESHOLD)
    ]
    upsert_rows(conn, "t", rows, ["id"])
    null_n_count = conn.execute("SELECT COUNT(*) FROM t WHERE n IS NULL").fetchone()[0]
    assert null_n_count == UPSERT_BULK_THRESHOLD // 2
    label_vals = conn.execute("SELECT DISTINCT label FROM t").fetchall()
    assert label_vals == [(None,)]
    # Explicitly rule out the pandas NaN-coercion gotcha: an integer column
    # with interspersed None must not silently upcast to float and leave
    # non-null values as e.g. 1.0 instead of 1.
    non_null_n = conn.execute(
        "SELECT n FROM t WHERE id = 'id000001'"
    ).fetchone()[0]
    assert non_null_n == 1
    assert isinstance(non_null_n, int)


# ---------------------------------------------------------------------------
# INSERT OR REPLACE semantics preserved (second write wins, no duplicates)
# ---------------------------------------------------------------------------

def test_upsert_overwrites_existing_row_small_path(conn):
    upsert_rows(conn, "t", [{"id": "x", "n": 1, "label": "first", "tags": None, "meta": None}], ["id"])
    upsert_rows(conn, "t", [{"id": "x", "n": 2, "label": "second", "tags": None, "meta": None}], ["id"])
    rows = conn.execute("SELECT n, label FROM t WHERE id = 'x'").fetchall()
    assert rows == [(2, "second")]


def test_upsert_overwrites_existing_rows_bulk_path(conn):
    first = make_rows(UPSERT_BULK_THRESHOLD)
    upsert_rows(conn, "t", first, ["id"])
    second = [{**r, "label": r["label"] + "-v2"} for r in first]
    upsert_rows(conn, "t", second, ["id"])

    total = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    assert total == UPSERT_BULK_THRESHOLD  # no duplicates
    label = conn.execute("SELECT label FROM t WHERE id = 'id000000'").fetchone()[0]
    assert label.endswith("-v2")


# ---------------------------------------------------------------------------
# Column-order robustness (defends the explicit-column-list choice over
# SELECT * in the bulk path's INSERT ... SELECT)
# ---------------------------------------------------------------------------

def test_varying_dict_key_order_does_not_misalign_columns_bulk_path(conn):
    # rows[0].keys() sets the column order used for the whole batch (existing,
    # unchanged contract) -- but later rows may have keys in a different
    # insertion order; this must not shift values into the wrong SQL column.
    rows = []
    for i in range(UPSERT_BULK_THRESHOLD):
        if i % 2 == 0:
            rows.append({"id": f"id{i:06d}", "n": i, "label": f"row-{i}", "tags": None, "meta": None})
        else:
            # same keys, different insertion order
            rows.append({"label": f"row-{i}", "id": f"id{i:06d}", "meta": None, "n": i, "tags": None})
    upsert_rows(conn, "t", rows, ["id"])
    got = conn.execute("SELECT id, n, label FROM t WHERE id = 'id000001'").fetchone()
    assert got == ("id000001", 1, "row-1")
