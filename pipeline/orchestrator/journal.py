"""
Observability journal for the pipeline orchestrator.
Writes per-item task_run rows to DuckDB for throughput history and status reporting.
"""

import logging
import uuid

import duckdb

from pipeline.catalog.catalog import upsert_rows

logger = logging.getLogger(__name__)


def new_run_id(node: str) -> str:
    safe = node.replace(".", "_")
    return f"{safe}_{uuid.uuid4().hex[:12]}"


def record_batch(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    node: str,
    item_ids: list[str],
    status: str,
    *,
    started: str | None = None,
    finished: str | None = None,
    error: str | None = None,
    metrics: dict | None = None,
) -> int:
    if not item_ids:
        return 0

    rows = [
        {
            "run_id": run_id,
            "node": node,
            "item_id": item_id,
            "status": status,
            "started": started,
            "finished": finished,
            "error": error,
            "metrics": metrics,
        }
        for item_id in item_ids
    ]

    return upsert_rows(conn, "task_runs", rows, ["run_id", "node", "item_id"])


def run_summary(conn: duckdb.DuckDBPyConnection, run_id: str) -> dict:
    result = conn.execute(
        """
        SELECT
            COUNT(*)                                        AS total,
            COUNT(*) FILTER (WHERE status = 'ok')          AS ok,
            COUNT(*) FILTER (WHERE status = 'error')       AS error,
            array_agg(DISTINCT node ORDER BY node)         AS nodes
        FROM task_runs
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()

    total, ok, err, nodes = result
    return {
        "run_id": run_id,
        "total": total or 0,
        "ok": ok or 0,
        "error": err or 0,
        "nodes": nodes or [],
    }
