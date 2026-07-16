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
numbers: docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md. This is why tiers.tier
gained a 4th value, 'auto_gold' (pipeline/nodes/tier.py), for segments clearing
that bar WITHOUT human review -- a statistical-confidence tier, sample-QA'd via
this node's tier/min_agreement-scoped sampling below, not exhaustively
reviewed. Random sampling (this node's original purpose) remains how auto_gold
itself gets spot-checked, and remains the only path for segments below the
auto_gold bar to reach true human-verified 'gold'.

REVISION (2026-07-11, owner decision): thresholds tightened and a 5th tier,
'bronze', added -- auto_gold raised 0.90->0.95, silver raised 0.65->0.85, and
the manifest-eligibility floor raised 0.65->0.70 (bronze = agreement>=0.70,
below that is 'excluded'). See DECISIONS.md 2026-07-11. QA sample rate is now
risk-scaled per tier (QA_SAMPLE_RATE_BY_TIER below) rather than a flat 2-5%:
bronze is the noisiest manifest-eligible tier and gets the highest sample rate,
auto_gold the lowest (highest a-priori confidence).

This node only SELECTS the sample and reserves it (writes 'pending' rows) --
it never touches audio or text itself. The actual review happens
interactively in the browser tool; see calibrate_server.py / record_decision
below for how 'pending' rows become 'verified'/'skipped'/'rejected' and how
a 'verified' decision propagates into asr_agreement.text_verified and
tiers.tier='gold'.
"""

import json
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Offline-review support (added 2026-07-13, see DECISIONS.md): DuckDB's writer
# lock is held for a long batch node's ENTIRE runtime (hours for something like
# asr.transcribe), which blocks even connect_ro() -- the per-request short-lived
# connections calibrate_server.py already uses (2026-07-10 redesign) only help
# with brief overlaps, not a multi-hour hold. Two pieces close that gap:
#   1. run_calibrate_export_snapshot() -- dump the pending review queue (audio
#      paths + ASR candidates, everything the browser UI needs) to a JSON file
#      while the catalog is free, so calibrate_server can fall back to reading
#      it when the live DB is unreachable.
#   2. append_pending_decision() / run_calibrate_flush_pending() -- every
#      review decision is appended to a local JSONL buffer instead of calling
#      record_decision() inline from the HTTP handler (removes the write path's
#      dependency on DB availability entirely); run_calibrate_flush_pending()
#      replays the buffer into record_decision() once the writer is free again
#      (idempotent -- record_decision is a plain UPDATE, safe to replay).
from pipeline.config import REPO_ROOT

SNAPSHOT_PATH = REPO_ROOT / "metadata" / "calibration_offline_queue.json"
PENDING_DECISIONS_PATH = REPO_ROOT / "metadata" / "calibration_pending_decisions.jsonl"

_pending_write_lock = threading.Lock()

# Risk-scaled QA sample rate per tier (owner decision, 2026-07-11; auto_gold gate rebuilt
# 2026-07-15, see pipeline/nodes/tier.py's module docstring): the noisier the tier's
# a-priori confidence, the higher the fraction of it that gets human review. auto_gold clears
# the strictest statistical bar (agreement>=0.92 AND dnsmos>=3.5) so it needs the
# least checking; bronze is the lowest manifest-eligible bar (agreement>=0.70) so it needs the
# most. 'gold' (already human-verified) and 'excluded' (never enters the manifest) are not QA'd.
QA_SAMPLE_RATE_BY_TIER = {
    "auto_gold": 0.015,  # ~1-2%
    "silver": 0.04,      # ~3-5%
    "bronze": 0.10,      # ~8-12%
}

# Code-switch QA oversampling multiplier (owner decision, 2026-07-15, T18 /
# pending_task.md): T16's distribution analysis found code-switched segments
# (filters.english_ratio > 0) clear ASR-agreement thresholds far less often than
# pure-Cantonese ones (e.g. 18.8% vs 48.5% at agreement>=0.90) -- the two active ASR
# backends (qwen3_asr AR / sense_voice CTC) diverge systematically on English-token
# transliteration/spelling, not necessarily on audio/text quality. That means a given
# agreement score is a WEAKER trust signal for a code-switch segment than the same score
# on a pure-Cantonese segment, so a dedicated code-switch QA sample (recommended_sample_n
# / discover with code_switch='only') uses this multiplier on top of the tier's normal
# base rate rather than relying on the population-wide rate to naturally cover enough
# code-switch examples.
CODE_SWITCH_QA_MULTIPLIER = 10.0

# Fixed flag_reason used by calibrate_server.py's one-click "Mandarin" button
# (added 2026-07-15): a segment surfaced for text QA but that is actually
# Mandarin (or otherwise non-HK-Cantonese) content, not a text-quality issue.
# Submitted as decision='rejected' (see record_decision below -- 'rejected'
# now excludes the segment from the manifest) with this reason string, so it
# shows up distinctly in summary_stats()'s top_flag_reasons leaderboard
# instead of being buried among free-text rejection notes.
MANDARIN_FLAG_REASON = "mandarin"


def recommended_sample_n(
    conn, tier: str, min_n: int = 50, *, code_switch: bool = False
) -> int:
    """Population-scaled QA sample size for `tier`, per QA_SAMPLE_RATE_BY_TIER --
    population is filter-passing segments currently carrying that tier. Floors at
    `min_n` so a small/newly-introduced tier still gets a meaningful spot-check.

    code_switch=True (added 2026-07-15, T18) scopes BOTH the population count and the
    rate to code-switched segments only (filters.english_ratio > 0), applying
    CODE_SWITCH_QA_MULTIPLIER on top of the tier's base rate -- e.g. auto_gold's base
    1.5% becomes 15% for its code-switch subset. Rate is capped at 1.0 (100%) so a small
    population with a high multiplier never asks for more than "review all of them"."""
    if tier not in QA_SAMPLE_RATE_BY_TIER:
        raise ValueError(f"no QA sample rate defined for tier {tier!r}")
    code_switch_cond = "AND f.english_ratio > 0" if code_switch else ""
    population = conn.execute(
        f"SELECT count(*) FROM tiers t JOIN filters f ON f.id = t.id "
        f"WHERE t.tier = ? AND f.pass = TRUE {code_switch_cond}",
        [tier],
    ).fetchone()[0]
    rate = QA_SAMPLE_RATE_BY_TIER[tier]
    if code_switch:
        rate = min(rate * CODE_SWITCH_QA_MULTIPLIER, 1.0)
    return max(min_n, round(population * rate))


SAMPLE_DISCOVER_SQL = """
    SELECT a.id, a.best_text
    FROM asr_agreement a
    JOIN filters f ON a.id = f.id AND f.pass = TRUE
    {tier_join}
    LEFT JOIN calibration_review c ON a.id = c.id
    WHERE c.id IS NULL
      {min_agreement_cond}
      {code_switch_cond}
    ORDER BY random()
    LIMIT ?
"""

_CODE_SWITCH_SAMPLE_CONDITIONS = {
    None: "",
    "only": "AND f.english_ratio > 0",
    "exclude": "AND f.english_ratio = 0",
}


def discover(
    conn,
    n: int,
    tier: str | None = None,
    min_agreement: float | None = None,
    code_switch: str | None = None,
) -> list[tuple[str, str]]:
    """tier/min_agreement (added 2026-07-10) narrow the sample population for scoped QA:
      - tier='auto_gold' -- QA the statistical-confidence tier specifically
        (pipeline/nodes/tier.py's auto_gold, agreement>=0.92 AND dnsmos>=3.5).
        Also valid: tier='silver' / tier='bronze' for those tiers' own QA pools.
      - min_agreement=0.95/0.85/etc -- QA a specific --min-agreement export cut
        (pipeline/nodes/manifest.py), independent of tier labels.
    code_switch (added 2026-07-15, T18): 'only' scopes the sample to
    filters.english_ratio > 0 (pair with recommended_sample_n(..., code_switch=True) for
    the matching oversampled `n`), 'exclude' scopes to english_ratio = 0. All three
    default to None, reproducing the original unscoped behaviour exactly (random sample
    of all filter-passing, not-yet-reviewed segments)."""
    if code_switch not in _CODE_SWITCH_SAMPLE_CONDITIONS:
        raise ValueError(
            f"code_switch must be one of {sorted(k for k in _CODE_SWITCH_SAMPLE_CONDITIONS if k)}, "
            f"got {code_switch!r}"
        )
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
    sql = SAMPLE_DISCOVER_SQL.format(
        tier_join=tier_join,
        min_agreement_cond=min_agreement_cond,
        code_switch_cond=_CODE_SWITCH_SAMPLE_CONDITIONS[code_switch],
    )
    return conn.execute(sql, params).fetchall()


async def run_calibrate_sample(
    *,
    conn=None,
    n: int = 300,
    tier: str | None = None,
    min_agreement: float | None = None,
    code_switch: str | None = None,
) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) -- pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run calibrate.sample` usage.

    tier/min_agreement/code_switch: see discover()'s docstring -- scoped QA sampling,
    added 2026-07-10 (tier/min_agreement) and 2026-07-15 (code_switch, T18)."""
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    picked = discover(conn, n, tier=tier, min_agreement=min_agreement, code_switch=code_switch)
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

    decision='rejected' (added 2026-07-15: propagation fix) directly upserts
    tiers.tier='excluded' for this id, same mechanism/rationale as the
    'verified'->'gold' write above. Before this, a human 'rejected' decision
    was recorded in calibration_review but never reached manifest.py's
    eligibility join (which only reads segments/asr_agreement/g2p/filters/
    tiers, never calibration_review) -- a reviewer rejecting a segment had no
    actual effect on what shipped in the manifest. flag_reason (e.g. the
    calibrate_server.py Mandarin-flag button's fixed reason "mandarin", see
    MANDARIN_FLAG_REASON) is stored alongside for triage but does not change
    this behaviour -- any 'rejected' decision excludes, regardless of reason.

    decision='flagged' is a pipeline-bug report (bad segmentation, corrupt
    audio, etc. spotted during review) that the reviewer is NOT simultaneously
    rejecting -- distinct from 'rejected' (text/audio is bad enough to drop
    from the manifest). It never touches asr_agreement/tiers -- flag_reason is
    free text for later triage. Language-mislabel issues (segment is actually
    Mandarin, not Cantonese) should go through decision='rejected' with
    flag_reason=MANDARIN_FLAG_REASON instead, since those segments should not
    ship in an HK-Cantonese-only corpus (CLAUDE.md hard constraint #1).
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
    elif decision == "rejected":
        upsert_rows(
            conn, "tiers",
            [{"id": seg_id, "tier": "excluded", "provenance": "calibrate_reject"}],
            ["id"],
        )


def pending_queue_rows(
    conn,
    sample_batch: str | None = None,
    source: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """All 'pending' review items (segment + ASR context), for
    run_calibrate_export_snapshot() -- like next_pending() but returns every
    match instead of just the first one, and includes sample_batch/queued_at
    (needed offline for the batch-jump dropdown, absent from _row_to_item's
    online shape since the live UI gets those via /api/batches instead)."""
    where = ["c.decision = 'pending'"]
    params: list = []
    if sample_batch:
        where.append("c.sample_batch = ?")
        params.append(sample_batch)
    if source:
        where.append("s.source = ?")
        params.append(source)
    limit_sql = ""
    if limit:
        limit_sql = "LIMIT ?"
        params.append(limit)
    rows = conn.execute(
        f"""
        SELECT c.id, a.best_text, a.agreement, s.source, s.audio_path,
               s.duration_sec, s.program, c.decision, c.reviewed_text, c.flag_reason,
               c.sample_batch, c.queued_at
        FROM calibration_review c
        JOIN asr_agreement a ON c.id = a.id
        JOIN segments s ON c.id = s.id
        WHERE {' AND '.join(where)}
        ORDER BY c.queued_at ASC
        {limit_sql}
        """,
        params,
    ).fetchall()
    items = []
    for row in rows:
        (seg_id, best_text, agreement, source_, audio_path, duration_sec, program,
         decision, reviewed_text, flag_reason, sample_batch_, queued_at) = row
        items.append({
            "id": seg_id,
            "best_text": best_text,
            "agreement": agreement,
            "source": source_,
            "audio_path": audio_path,
            "duration_sec": duration_sec,
            "program": program,
            "decision": decision,
            "reviewed_text": reviewed_text,
            "flag_reason": flag_reason,
            "sample_batch": sample_batch_,
            "queued_at": str(queued_at) if queued_at else None,
            "candidates": _fetch_candidates(conn, seg_id),
        })
    return items


async def run_calibrate_export_snapshot(
    *, conn=None, out_path: Path | None = None,
    limit: int | None = None, sample_batch: str | None = None, source: str | None = None,
) -> dict:
    """Dump the current pending review queue to a JSON file (default
    SNAPSHOT_PATH) so calibrate_server.py can serve reads from it when the
    live catalog is unreachable (a long batch node holding the writer lock --
    see module docstring). Read-only: uses connect_ro() by default, not
    connect(), since this never writes anything and read-only connections
    don't contend with other processes' own read-only connections."""
    from pipeline.catalog.catalog import CATALOG_PATH, connect_ro

    out_path = out_path or SNAPSHOT_PATH
    owns_conn = conn is None
    conn = conn or connect_ro(CATALOG_PATH)
    try:
        items = pending_queue_rows(conn, sample_batch=sample_batch, source=source, limit=limit)
    finally:
        if owns_conn:
            conn.close()

    snapshot = {
        "snapshot_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "items": items,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, ensure_ascii=False))
    log.info(f"calibrate.export_snapshot: wrote {len(items)} pending items to {out_path}")
    return {"exported": len(items), "path": str(out_path)}


def append_pending_decision(
    seg_id: str,
    decision: str,
    text: str | None,
    flag_reason: str | None,
    *,
    sample_batch: str | None = None,
    source: str | None = None,
    path: Path | None = None,
) -> dict:
    """Append one review decision to the local JSONL buffer (default
    PENDING_DECISIONS_PATH) instead of writing it straight to the catalog.
    Called by calibrate_server.py's /api/submit for EVERY decision (not just
    as a busy-catalog fallback -- see module docstring) so a click never
    blocks on DB availability. run_calibrate_flush_pending() later replays
    these into record_decision(). Thread-safety: guarded by
    _pending_write_lock so two browser tabs submitting at once can't
    interleave partial JSON lines."""
    if decision not in ("verified", "skipped", "rejected", "flagged"):
        raise ValueError(f"invalid decision: {decision!r}")
    path = path or PENDING_DECISIONS_PATH
    entry = {
        "id": seg_id,
        "decision": decision,
        "text": text,
        "flag_reason": flag_reason,
        "sample_batch": sample_batch,
        "source": source,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with _pending_write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def load_pending_decisions(path: Path | None = None) -> dict[str, dict]:
    """Read the JSONL buffer into {id: latest_entry} -- a segment reviewed
    twice in the same offline session (e.g. changed their mind before flush)
    keeps only its last decision, matching how record_decision's plain UPDATE
    would behave once flushed anyway."""
    path = path or PENDING_DECISIONS_PATH
    if not path.exists():
        return {}
    entries: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries[entry["id"]] = entry
    return entries


async def run_calibrate_flush_pending(*, conn=None, in_path: Path | None = None) -> dict:
    """Replay the local decision buffer into the catalog via record_decision(),
    then archive the buffer file so it isn't replayed again. Safe to run
    anytime the writer lock is free; safe to re-run if interrupted mid-way
    (record_decision is a plain UPDATE, idempotent per id)."""
    from pipeline.catalog.catalog import connect

    in_path = in_path or PENDING_DECISIONS_PATH
    entries = load_pending_decisions(in_path)
    if not entries:
        return {"flushed": 0, "errors": 0, "archived_to": None}

    owns_conn = conn is None
    conn = conn or connect()
    flushed = 0
    errors = []
    try:
        for seg_id, entry in entries.items():
            try:
                record_decision(conn, seg_id, entry["decision"], entry.get("text"), entry.get("flag_reason"))
                flushed += 1
            except Exception as exc:  # noqa: BLE001 -- one bad row must not abort the whole flush
                log.error(f"calibrate.flush_pending: failed to flush {seg_id}: {exc}")
                errors.append(seg_id)
    finally:
        if owns_conn:
            conn.close()

    archived_to = None
    if not errors:
        archived_to = in_path.with_name(f"{in_path.stem}.flushed_{time.strftime('%Y%m%dT%H%M%S')}{in_path.suffix}")
        in_path.rename(archived_to)
        archived_to = str(archived_to)
    else:
        # Leave failed ids in the buffer (as the only remaining lines) so a
        # re-run only retries what actually failed, and rewrite the file
        # atomically instead of leaving successfully-flushed lines mixed in.
        with _pending_write_lock:
            tmp_path = in_path.with_suffix(in_path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                for seg_id in errors:
                    f.write(json.dumps(entries[seg_id], ensure_ascii=False) + "\n")
            tmp_path.replace(in_path)

    log.info(f"calibrate.flush_pending: flushed {flushed}, errors {len(errors)}, archived_to={archived_to}")
    return {"flushed": flushed, "errors": len(errors), "archived_to": archived_to}


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

    # Includes 'rejected' as well as 'flagged' (added 2026-07-15) so the Mandarin-flag
    # button's flag_reason='mandarin' rejections (see MANDARIN_FLAG_REASON) show up here
    # too, not just informational 'flagged' reports.
    where_sql, params = _where(
        "c.decision IN ('flagged', 'rejected') AND c.flag_reason IS NOT NULL"
    )
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
     decision, reviewed_text, flag_reason, sample_batch) = row
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
        # sample_batch (added 2026-07-13): the client echoes this back on
        # /api/submit so append_pending_decision() can scope the offline
        # decision buffer by batch (see calibrate_server.py).
        "sample_batch": sample_batch,
        "candidates": candidates,
    }


def next_pending(
    conn,
    sample_batch: str | None = None,
    source: str | None = None,
    order: str = "queued",
    exclude_ids: "set[str] | list[str] | None" = None,
) -> dict | None:
    """Return the next 'pending' review item (segment + ASR context) for the
    browser UI, or None if the queue (optionally scoped to sample_batch /
    source) is empty. `order` picks which segment surfaces first: 'queued'
    (FIFO, default), 'agreement_asc' (lowest cross-model agreement first --
    the segments most likely to actually need correction), or
    'agreement_desc'.

    exclude_ids (added 2026-07-13): ids already decided in the local
    calibrate_server offline-decision buffer (see module docstring) but not
    yet flushed to the catalog -- calibration_review.decision still reads
    'pending' for them in the DB, so without this exclusion the same segment
    would be served for review again before its own just-submitted decision
    lands."""
    where = ["c.decision = 'pending'"]
    params: list = []
    if sample_batch:
        where.append("c.sample_batch = ?")
        params.append(sample_batch)
    if source:
        where.append("s.source = ?")
        params.append(source)
    if exclude_ids:
        placeholders = ",".join("?" for _ in exclude_ids)
        where.append(f"c.id NOT IN ({placeholders})")
        params.extend(exclude_ids)
    order_sql = _ORDER_SQL.get(order, _ORDER_SQL["queued"])

    row = conn.execute(
        f"""
        SELECT c.id, a.best_text, a.agreement, s.source, s.audio_path,
               s.duration_sec, s.program, c.decision, c.reviewed_text, c.flag_reason,
               c.sample_batch
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
               s.duration_sec, s.program, c.decision, c.reviewed_text, c.flag_reason,
               c.sample_batch
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
