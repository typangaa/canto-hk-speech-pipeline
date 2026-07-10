"""
pipeline/nodes/calibrate.py
calibrate.sample DAG node -- selects a random sample of filter-passing,
not-yet-reviewed segments and queues them into `calibration_review` for
human review via pipeline/tools/calibrate_server.py's local web UI.

Why sample-based, not full-corpus (owner decision, 2026-07-10)
----------------------------------------------------------------
text_verified/gold is structurally dead in the current DAG: asr.transcribe
always writes text_verified=False (pipeline/nodes/asr.py) and no live node
ever flips it -- the only path that once did was the legacy
scripts/05_calibrate.py, which read/wrote data/segments/*.transcript.json
sidecars that no longer exist under the catalog-driven architecture.
Exhaustively re-verifying all ~369,700 manifest-eligible segments (~820h) by
hand is not a realistic ask of one owner. A random sample instead (a)
produces a real, defensible measurement of the silver tier's actual text
quality, and (b) produces a genuine (if partial) pool of gold-tier segments.
Checked 2026-07-10: auto-promoting via a high cross-model-agreement threshold
instead of human review is NOT viable -- only 26 filter-passing segments
clear agreement >= 0.95 corpus-wide (median agreement is 0.71), nowhere near
enough to matter.

This node only SELECTS the sample and reserves it (writes 'pending' rows) --
it never touches audio or text itself. The actual review happens
interactively in the browser tool; see calibrate_server.py / record_decision
below for how 'pending' rows become 'verified'/'skipped'/'rejected' and how
a 'verified' decision propagates into asr_agreement.text_verified and
tiers.tier='gold'.
"""

import logging
import time

log = logging.getLogger(__name__)

SAMPLE_DISCOVER_SQL = """
    SELECT a.id
    FROM asr_agreement a
    JOIN filters f ON a.id = f.id AND f.pass = TRUE
    LEFT JOIN calibration_review c ON a.id = c.id
    WHERE c.id IS NULL
    ORDER BY random()
    LIMIT ?
"""


def discover(conn, n: int) -> list[str]:
    return [row[0] for row in conn.execute(SAMPLE_DISCOVER_SQL, [n]).fetchall()]


async def run_calibrate_sample(*, conn=None, n: int = 300) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) -- pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run calibrate.sample` usage."""
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    ids = discover(conn, n)
    log.info(f"calibrate.sample: queuing {len(ids)} segments for human review")
    if not ids:
        return {"queued": 0, "run_id": None}

    run_id = new_run_id("calibrate.sample")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        {
            "id": seg_id,
            "decision": "pending",
            "reviewed_text": None,
            "sample_batch": run_id,
            "queued_at": now,
            "reviewed_at": None,
        }
        for seg_id in ids
    ]
    upsert_rows(conn, "calibration_review", rows, ["id"])
    record_batch(conn, run_id, "calibrate.sample", ids, "ok")
    return {"queued": len(ids), "run_id": run_id}


def record_decision(conn, seg_id: str, decision: str, text: str | None) -> None:
    """Write one human review decision back to the catalog. Called by
    calibrate_server.py's POST /api/submit handler -- NOT a DAG node itself
    (interactive, not idempotent-anti-join discovery), but kept alongside
    run_calibrate_sample since both write to `calibration_review`.

    decision must be one of 'verified' / 'skipped' / 'rejected'.

    decision='verified' also flips asr_agreement.text_verified=True (and
    overwrites best_text with the reviewer's -- possibly corrected -- text)
    and directly upserts tiers.tier='gold' for this id. The direct tiers
    write sidesteps a known limitation: tier.assign's discovery anti-joins on
    provenance='tier_assign' and will not re-tier a row that already has a
    tier_assign row, even after text_verified flips true underneath it -- so
    without this direct write a freshly-verified segment would stay stuck at
    its old silver/excluded tier forever. assign_tier(True, *) in
    pipeline/nodes/tier.py is unconditionally 'gold', so writing 'gold'
    directly here reproduces exactly what a re-run of tier.assign would
    compute for this row -- not a shortcut around its logic, just doing it
    inline since the anti-join can't reach this row again.
    """
    from pipeline.catalog.catalog import upsert_rows

    if decision not in ("verified", "skipped", "rejected"):
        raise ValueError(f"invalid decision: {decision!r}")

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "UPDATE calibration_review SET decision = ?, reviewed_text = ?, reviewed_at = ? "
        "WHERE id = ?",
        [decision, text, now, seg_id],
    )
    if decision == "verified":
        conn.execute(
            "UPDATE asr_agreement SET text_verified = TRUE, best_text = ? WHERE id = ?",
            [text, seg_id],
        )
        upsert_rows(
            conn, "tiers",
            [{"id": seg_id, "tier": "gold", "provenance": "calibrate_verify"}],
            ["id"],
        )


def next_pending(conn, sample_batch: str | None = None) -> dict | None:
    """Return the next 'pending' review item (segment + ASR context) for the
    browser UI, or None if the queue (optionally scoped to one sample_batch)
    is empty."""
    where_batch = "AND c.sample_batch = ?" if sample_batch else ""
    params = [sample_batch] if sample_batch else []
    row = conn.execute(
        f"""
        SELECT c.id, a.best_text, a.agreement, s.source, s.audio_path,
               s.duration_sec, s.program
        FROM calibration_review c
        JOIN asr_agreement a ON c.id = a.id
        JOIN segments s ON c.id = s.id
        WHERE c.decision = 'pending' {where_batch}
        ORDER BY c.queued_at
        LIMIT 1
        """,
        params,
    ).fetchone()
    if row is None:
        return None

    seg_id, best_text, agreement, source, audio_path, duration_sec, program = row
    candidates = conn.execute(
        "SELECT model, text, confidence FROM asr_results WHERE id = ? ORDER BY model",
        [seg_id],
    ).fetchall()
    return {
        "id": seg_id,
        "best_text": best_text,
        "agreement": agreement,
        "source": source,
        "audio_path": audio_path,
        "duration_sec": duration_sec,
        "program": program,
        "candidates": [
            {"model": m, "text": t, "confidence": c} for m, t, c in candidates
        ],
    }


def queue_stats(conn, sample_batch: str | None = None) -> dict:
    where_batch = "WHERE sample_batch = ?" if sample_batch else ""
    params = [sample_batch] if sample_batch else []
    rows = conn.execute(
        f"SELECT decision, count(*) FROM calibration_review {where_batch} GROUP BY decision",
        params,
    ).fetchall()
    stats = {"pending": 0, "verified": 0, "skipped": 0, "rejected": 0}
    for decision, n in rows:
        stats[decision] = n
    stats["total"] = sum(stats.values())
    return stats
