"""
pipeline/catalog/catalog.py — DuckDB catalog connection factory and schema management.

Single-writer discipline
------------------------
DuckDB does not support concurrent writers to the same file. This module enforces
a strict single-writer model:

- ``connect()``    → READ-WRITE connection. Must be used by **exactly one**
                     process at a time (the catalog build / import process).
                     Concurrent calls from multiple processes will cause
                     DuckDB to raise a ``duckdb.IOException``.

- ``connect_ro()`` → READ-ONLY connection. Safe to open from any number of
                     processes simultaneously (CLI verify, tests, ad-hoc
                     queries). Cannot create the database file if it is absent.

See docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md §3.2 for the rationale.
"""

import json
import uuid
from pathlib import Path

import duckdb
import pandas as pd

from pipeline.config import CATALOG_PATH

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCHEMA_PATH: Path = Path(__file__).resolve().parent / "schema.sql"


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Execute every DDL statement in schema.sql against *conn*.

    DuckDB's ``conn.execute()`` accepts exactly one statement at a time, so
    the SQL file is split on semicolons and each non-empty fragment is
    executed individually.  Full-line SQL comments (``-- …``) are stripped
    BEFORE splitting — several comments in schema.sql use semicolons in
    prose (e.g. "one row per corpus source; may be sparsely populated"),
    which would otherwise fragment mid-comment and produce invalid SQL.
    """
    sql_text = SCHEMA_PATH.read_text(encoding="utf-8")

    # Strip full-line comments first so their semicolons can't split statements.
    code_only = "\n".join(
        line for line in sql_text.splitlines()
        if not line.strip().startswith("--")
    )

    for fragment in code_only.split(";"):
        if fragment.strip():
            conn.execute(fragment)


# ---------------------------------------------------------------------------
# Connection factories
# ---------------------------------------------------------------------------

def connect(
    catalog_path: Path = CATALOG_PATH,
    *,
    ensure_schema: bool = True,
) -> duckdb.DuckDBPyConnection:
    """Open a **read-write** DuckDB connection to *catalog_path*.

    The parent directory is created if it does not exist.  When
    *ensure_schema* is ``True`` (the default), :func:`init_schema` is called
    before the connection is returned so that all tables / indexes defined in
    ``schema.sql`` are present.

    .. warning::
        Only **one** process must hold a read-write connection at a time.
        Attempting to open a second writer while the first is alive will
        raise ``duckdb.IOException``.
    """
    catalog_path = Path(catalog_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(catalog_path))

    if ensure_schema:
        init_schema(conn)

    return conn


def connect_ro(
    catalog_path: Path = CATALOG_PATH,
) -> duckdb.DuckDBPyConnection:
    """Open a **read-only** DuckDB connection to *catalog_path*.

    Safe to call from multiple processes simultaneously.

    Raises
    ------
    RuntimeError
        If *catalog_path* does not exist.  Read-only mode cannot create the
        database file; you must run ``pipe catalog build`` first to
        initialise it.
    """
    catalog_path = Path(catalog_path)

    if not catalog_path.exists():
        raise RuntimeError(
            f"Catalog database not found at '{catalog_path}'. "
            "Please run 'pipe catalog build' first to create and populate it."
        )

    return duckdb.connect(str(catalog_path), read_only=True)


# ---------------------------------------------------------------------------
# Generic bulk-upsert helper
# ---------------------------------------------------------------------------

# Below this row count, conn.executemany()'s per-statement overhead is cheap
# enough that building a DataFrame isn't worth it. At/above it, executemany()
# becomes the dominant cost (DuckDB's DBAPI executemany() binds and executes
# one parameterised statement per row -- an OLAP engine paying OLTP-style
# per-row overhead). See docs/UPSERT_PERFORMANCE_FIX_PLAN.md for the full
# investigation (measured: a 538,310-row executemany() upsert against an
# already-populated table took ~78 minutes; DuckDB's own issue tracker
# documents executemany() as 100x+ slower than its vectorised bulk-insert
# path for comparable row counts).
UPSERT_BULK_THRESHOLD = 2_000


def upsert_rows(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    rows: list[dict],
    key_columns: list[str],  # noqa: ARG001  (reserved for future ON CONFLICT use)
) -> int:
    """Bulk-upsert *rows* into *table* using ``INSERT OR REPLACE INTO``.

    Parameters
    ----------
    conn:
        An open read-write DuckDB connection.
    table:
        Target table name (not sanitised — callers must supply trusted names).
    rows:
        List of dicts, all sharing the same set of keys.  Column order is
        derived from ``rows[0].keys()`` (dict insertion order).
    key_columns:
        The primary-key column(s) that define uniqueness.  DuckDB's
        ``INSERT OR REPLACE INTO`` replaces any existing row whose primary
        key conflicts, so the table must have the corresponding PK / UNIQUE
        constraint defined in ``schema.sql``.

    Returns
    -------
    int
        Number of rows upserted (``len(rows)``), or ``0`` if *rows* is empty.

    Notes
    -----
    Python ``list`` / ``dict`` values are serialised to JSON strings before
    binding because DuckDB's parameterised interface does not automatically
    coerce them to its native JSON type.  Plain ``str``, ``int``, ``float``,
    ``bool``, and ``None`` are passed through unchanged.

    Below ``UPSERT_BULK_THRESHOLD`` rows, this uses the original
    ``conn.executemany()`` path (cheap for small batches, e.g. a `--limit`
    test run). At/above the threshold, it registers a pandas DataFrame as a
    temporary view and does a single vectorised ``INSERT ... SELECT`` instead
    — see docs/UPSERT_PERFORMANCE_FIX_PLAN.md for the full rationale and
    benchmark. Both paths produce identical rows; only the write mechanism
    differs.
    """
    if not rows:
        return 0

    columns: list[str] = list(rows[0].keys())
    col_list = ", ".join(columns)

    def _coerce(value: object) -> object:
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return value

    if len(rows) < UPSERT_BULK_THRESHOLD:
        placeholders = ", ".join("?" * len(columns))
        sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"
        tuples = [
            tuple(_coerce(row[col]) for col in columns)
            for row in rows
        ]
        conn.executemany(sql, tuples)
        return len(rows)

    # Bulk path: register a DataFrame, single vectorised INSERT ... SELECT.
    # dtype=object keeps every cell as its original Python value (int/str/
    # None/...) instead of letting pandas infer a numpy dtype per column --
    # numpy's automatic dtype inference would silently upcast an int column
    # containing None to float64 and turn None into NaN, corrupting integer
    # PK/FK columns. object columns are slower to build in pure pandas terms
    # but DuckDB's registration still reads them as one vectorised scan, not
    # a per-row Python round-trip -- correctness here matters more than
    # squeezing out the last bit of DataFrame-construction speed.
    coerced = [
        {col: _coerce(row[col]) for col in columns}
        for row in rows
    ]
    df = pd.DataFrame(coerced, columns=columns, dtype=object)

    # Unique view name: this connection may be shared across concurrently
    # running `pipe run-many` nodes (each on its own conn.cursor()) -- a
    # uuid4 suffix means two simultaneous bulk upserts can never collide on
    # the same registered view name regardless of how DuckDB scopes
    # registration internally.
    view_name = f"_upsert_bulk_{uuid.uuid4().hex}"
    try:
        conn.register(view_name, df)
        conn.execute(
            f"INSERT OR REPLACE INTO {table} ({col_list}) "
            f"SELECT {col_list} FROM {view_name}"
        )
    finally:
        conn.unregister(view_name)

    return len(rows)
