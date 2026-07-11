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

CORRECTION (same day, later): that check included whisper_v3 (Systran/faster-
whisper-large-v3+zh) in the 4-way agreement score. The owner separately flagged
whisper_v3 as measurably inaccurate and asked for it to be retired from the
pipeline (see ASR_MODELS["whisper_v3"] in pipeline/nodes/asr.py and
DECISIONS.md). Re-measured with whisper_v3 excluded (3-way: canto_ft/qwen3_asr/
sense_voice), agreement >= 0.90 clears **41.1%** of the corpus (~446h), not
~0% -- the earlier "not viable" conclusion was an artifact of whisper_v3
dragging down the whole distribution, not a property of the corpus. Full
numbers: docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md. This is why tiers.tier now
has a 4th value, 'auto_gold' (pipeline/nodes/tier.py), for segments clearing
that bar WITHOUT human review -- a statistical-confidence tier, sample-QA'd via
this node's tier/min_agreement-scoped sampling below, not exhaustively
reviewed. Random sampling (this node's original purpose) remains how auto_gold
itself gets spot-checked, and remains the only path for segments below the
auto_gold bar to reach true human-verified 'gold'.

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
    SELECT a.id, a.best_text
    FROM asr_agreement a
    JOIN filters f ON a.id = f.id AND f.pass = TRUE
    {tier_join}
    LEFT JOIN calibration_review c ON a.id = c.id
    WHERE c.id IS NULL
      {min_agreement_cond}
    ORDER BY random()
    LIMIT ?
"""


def discover(
    conn, n: int, tier: str | None = None, min_agreement: float | None = None
) -> list[tuple[str, str]]:
    """tier/min_agreement (added 2026-07-10) narrow the sample population for scoped QA:
      - tier='auto_gold' -- QA the statistical-confidence tier specifically
        (pipeline/nodes/tier.py's auto_gold, agreement>=0.90 AND canto_ft_confidence>0.8).
      - min_agreement=0.95/0.85/etc -- QA a specific --min-agreement export cut
        (pipeline/nodes/manifest.py), independent of tier labels.
    Both default to None, reproducing the original unscoped behaviour exactly (random
    sample of all filter-passing, not-yet-reviewed segments)."""
    params: list = []
    tier_join = ""
    if tier is not None:
        tier_join = "JOIN tiers t2 ON a.id = t2.id AND t2.tier = ?"
        params.append(tier)
    min_agreement_cond = ""
    if min_agreement is not None:
        min_agreement_cond = "AND a.agreement >= ?"
        params.append(min_agreement)
    params.append(n)
    sql = SAMPLE_DISCOVER_SQL.format(tier_join=tier_join, min_agreement_cond=min_agreement_cond)
    return conn.execute(sql, params).fetchall()


async def run_calibrate_sample(
    *, conn=None, n: int = 300, tier: str | None = None, min_agreement: float | None = None
) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) -- pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run calibrate.sample` usage.

    tier/min_agreement: see discover()'s docstring -- scoped QA sampling, added 2026-07-10."""
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    picked = discover(conn, n, tier=tier, min_agreement=min_agreement)
    log.info(f"calibrate.sample: queuing {len(picked)} segments for human review")
    if not picked:
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
            "original_best_text": best_text,
        }
        for seg_id, best_text in picked
    ]
    ids = [r["id"] for r in rows]
    upsert_rows(conn, "calibration_review", rows, ["id"])
    record_batch(conn, run_id, "calibrate.sample", ids, "ok")
    return {"queued": len(ids), "run_id": run_id}


def record_decision(
    conn,
    seg_id: str,
    decision: str,
    text: str | None,
    flag_reason: str | None = None,
) -> None:
    """Write one human review decision back to the catalog. Called by
    calibrate_server.py's POST /api/submit handler -- NOT a DAG node itself
    (interactive, not idempotent-anti-join discovery), but kept alongside
    run_calibrate_sample since both write to `calibration_review`.

    decision must be one of 'verified' / 'skipped' / 'rejected' / 'flagged'.

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

    decision='flagged' is a pipeline-bug report (bad segmentation, non-
    Cantonese content, corrupt audio, etc. spotted during review), distinct
    from 'rejected' (text just isn't verifiable). It never touches
    asr_agreement/tiers -- flag_reason is free text for later triage.
    """
    from pipeline.catalog.catalog import upsert_rows

    if decision not in ("verified", "skipped", "rejected", "flagged"):
        raise ValueError(f"invalid decision: {decision!r}")

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "UPDATE calibration_review SET decision = ?, reviewed_text = ?, reviewed_at = ?, "
        "flag_reason = ? WHERE id = ?",
        [decision, text, now, flag_reason, seg_id],
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


def jyutping_preview(text: str) -> dict:
    """Live Jyutping-validity preview for the browser UI's text box -- reuses
    g2p.py's actual conversion + validation functions (not a reimplementation)
    so the preview can never drift from what the real g2p DAG node will
    compute once this segment's text is verified. Returns
    {jyutping, valid_fraction, accept, bad_tokens}."""
    from pipeline.nodes.g2p import text_to_jyutping, validate_jyutping

    text = (text or "").strip()
    if not text:
        return {"jyutping": "", "valid_fraction": 1.0, "accept": True, "bad_tokens": []}

    jyutping = text_to_jyutping(text)
    if not jyutping:
        return {"jyutping": "", "valid_fraction": 0.0, "accept": False, "bad_tokens": []}

    accept, frac, bad = validate_jyutping(jyutping)
    return {"jyutping": jyutping, "valid_fraction": frac, "accept": accept, "bad_tokens": bad}


def _levenshtein(a: str, b: str) -> int:
    """Character-level edit distance -- used as a cheap WER-ish proxy for
    'how much did the human need to change' between best_text and
    reviewed_text on verified segments (see summary_stats)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def summary_stats(conn, sample_batch: str | None = None) -> dict:
    """Aggregate quality metrics across a (optionally batch-scoped) review
    sample -- feeds the browser UI's summary dashboard. Distinct from
    queue_stats (raw decision counts): this adds a per-source breakdown,
    average cross-model agreement by decision (sanity check: rejected
    segments should skew lower-agreement than verified ones), an average
    edit-distance between ASR best_text and the human-corrected text for
    verified segments (the actual quality-ceiling measurement this whole
    tool exists to produce), and a flagged-reason leaderboard for triage."""
    def _where(extra: str | None = None) -> tuple[str, list]:
        conds = []
        params: list = []
        if sample_batch:
            conds.append("c.sample_batch = ?")
            params.append(sample_batch)
        if extra:
            conds.append(extra)
        return (f"WHERE {' AND '.join(conds)}" if conds else ""), params

    decision_counts = queue_stats(conn, sample_batch)

    where_sql, params = _where()
    by_source_rows = conn.execute(
        f"SELECT s.source, c.decision, count(*) FROM calibration_review c "
        f"JOIN segments s ON c.id = s.id {where_sql} GROUP BY s.source, c.decision",
        params,
    ).fetchall()
    by_source: dict[str, dict[str, int]] = {}
    for source, decision, n in by_source_rows:
        by_source.setdefault(source, {})[decision] = n

    where_sql, params = _where("c.decision != 'pending'")
    agreement_rows = conn.execute(
        f"SELECT c.decision, avg(a.agreement) FROM calibration_review c "
        f"JOIN asr_agreement a ON c.id = a.id {where_sql} GROUP BY c.decision",
        params,
    ).fetchall()
    avg_agreement_by_decision = {d: round(v, 3) for d, v in agreement_rows if v is not None}

    where_sql, params = _where("c.decision = 'verified'")
    verified_rows = conn.execute(
        f"SELECT c.original_best_text, c.reviewed_text FROM calibration_review c {where_sql}",
        params,
    ).fetchall()
    edits = [
        _levenshtein(original or "", reviewed) for original, reviewed in verified_rows
        if reviewed is not None
    ]
    avg_edit_distance = round(sum(edits) / len(edits), 2) if edits else None

    where_sql, params = _where("c.decision = 'flagged' AND c.flag_reason IS NOT NULL")
    flag_rows = conn.execute(
        f"SELECT c.flag_reason, count(*) FROM calibration_review c {where_sql} "
        f"GROUP BY c.flag_reason ORDER BY count(*) DESC LIMIT 10",
        params,
    ).fetchall()

    return {
        "decision_counts": decision_counts,
        "by_source": by_source,
        "avg_agreement_by_decision": avg_agreement_by_decision,
        "avg_edit_distance_verified": avg_edit_distance,
        "verified_edit_sample_size": len(edits),
        "top_flag_reasons": [{"reason": r, "count": n} for r, n in flag_rows],
    }


_ORDER_SQL = {
    "queued": "c.queued_at ASC",
    "agreement_asc": "a.agreement ASC",
    "agreement_desc": "a.agreement DESC",
}


def _fetch_candidates(conn, seg_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT model, text, confidence FROM asr_results WHERE id = ? ORDER BY model",
        [seg_id],
    ).fetchall()
    return [{"model": m, "text": t, "confidence": c} for m, t, c in rows]


def _row_to_item(row, candidates) -> dict:
    (seg_id, best_text, agreement, source, audio_path, duration_sec, program,
     decision, reviewed_text, flag_reason) = row
    return {
        "id": seg_id,
        "best_text": best_text,
        "agreement": agreement,
        "source": source,
        "audio_path": audio_path,
        "duration_sec": duration_sec,
        "program": program,
        "decision": decision,
        "reviewed_text": reviewed_text,
        "flag_reason": flag_reason,
        "candidates": candidates,
    }


def next_pending(
    conn,
    sample_batch: str | None = None,
    source: str | None = None,
    order: str = "queued",
) -> dict | None:
    """Return the next 'pending' review item (segment + ASR context) for the
    browser UI, or None if the queue (optionally scoped to sample_batch /
    source) is empty. `order` picks which segment surfaces first: 'queued'
    (FIFO, default), 'agreement_asc' (lowest cross-model agreement first --
    the segments most likely to actually need correction), or
    'agreement_desc'."""
    where = ["c.decision = 'pending'"]
    params: list = []
    if sample_batch:
        where.append("c.sample_batch = ?")
        params.append(sample_batch)
    if source:
        where.append("s.source = ?")
        params.append(source)
    order_sql = _ORDER_SQL.get(order, _ORDER_SQL["queued"])

    row = conn.execute(
        f"""
        SELECT c.id, a.best_text, a.agreement, s.source, s.audio_path,
               s.duration_sec, s.program, c.decision, c.reviewed_text, c.flag_reason
        FROM calibration_review c
        JOIN asr_agreement a ON c.id = a.id
        JOIN segments s ON c.id = s.id
        WHERE {' AND '.join(where)}
        ORDER BY {order_sql}
        LIMIT 1
        """,
        params,
    ).fetchone()
    if row is None:
        return None
    return _row_to_item(row, _fetch_candidates(conn, row[0]))


def get_item(conn, seg_id: str) -> dict | None:
    """Fetch one review item by id regardless of its decision state -- used
    by the history panel to reopen an already-decided segment for re-edit
    (re-submitting just calls record_decision again, which is a plain
    UPDATE, so changing your mind about a past decision is safe)."""
    row = conn.execute(
        """
        SELECT c.id, a.best_text, a.agreement, s.source, s.audio_path,
               s.duration_sec, s.program, c.decision, c.reviewed_text, c.flag_reason
        FROM calibration_review c
        JOIN asr_agreement a ON c.id = a.id
        JOIN segments s ON c.id = s.id
        WHERE c.id = ?
        """,
        [seg_id],
    ).fetchone()
    if row is None:
        return None
    return _row_to_item(row, _fetch_candidates(conn, row[0]))


def list_history(conn, sample_batch: str | None = None, limit: int = 20) -> list[dict]:
    """Most-recently-decided items (any non-'pending' decision), newest
    first -- feeds the browser UI's 'recently reviewed' panel so a decision
    can be revisited without re-queuing."""
    where = ["c.decision != 'pending'"]
    params: list = []
    if sample_batch:
        where.append("c.sample_batch = ?")
        params.append(sample_batch)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT c.id, c.decision, c.reviewed_text, c.reviewed_at, s.source, c.flag_reason
        FROM calibration_review c
        JOIN segments s ON c.id = s.id
        WHERE {' AND '.join(where)}
        ORDER BY c.reviewed_at DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        {
            "id": seg_id,
            "decision": decision,
            "reviewed_text": reviewed_text,
            "reviewed_at": str(reviewed_at) if reviewed_at else None,
            "source": source,
            "flag_reason": flag_reason,
        }
        for seg_id, decision, reviewed_text, reviewed_at, source, flag_reason in rows
    ]


def list_batches(conn) -> list[dict]:
    """Every sample_batch that has queued rows, each with its own
    queue_stats -- feeds the browser UI's batch-jump dropdown."""
    rows = conn.execute(
        "SELECT DISTINCT sample_batch FROM calibration_review "
        "WHERE sample_batch IS NOT NULL ORDER BY sample_batch"
    ).fetchall()
    return [{"sample_batch": b, **queue_stats(conn, b)} for (b,) in rows]


def list_sources(conn) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT source FROM segments WHERE source IS NOT NULL ORDER BY source"
    ).fetchall()
    return [r[0] for r in rows]


def queue_stats(conn, sample_batch: str | None = None, source: str | None = None) -> dict:
    where = []
    params: list = []
    if sample_batch:
        where.append("c.sample_batch = ?")
        params.append(sample_batch)
    if source:
        where.append("s.source = ?")
        params.append(source)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    join_sql = "JOIN segments s ON c.id = s.id" if source else ""
    rows = conn.execute(
        f"SELECT c.decision, count(*) FROM calibration_review c {join_sql} "
        f"{where_sql} GROUP BY c.decision",
        params,
    ).fetchall()
    stats = {"pending": 0, "verified": 0, "skipped": 0, "rejected": 0, "flagged": 0}
    for decision, n in rows:
        stats[decision] = n
    stats["total"] = sum(stats.values())
    return stats
