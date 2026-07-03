"""
pipeline/nodes/tier.py
tier.assign DAG node -- assigns each segment a verification-confidence tier
("gold" / "silver" / "excluded"), ported from the inline tier logic inside
scripts/09_manifest.py's build_entry() function into its own catalog-driven node
that writes one `tiers` table row per segment, independent of manifest building.
Runs entirely in-supervisor (no worker subprocess, no GPU).

Legacy-row-collision note
--------------------------
The `tiers` table already contains 455,299 rows imported from the legacy pipeline
(all tier IN ('gold','silver'), provenance IS NULL).  A bare row-existence anti-join
would find zero unassigned work on every run.  Discovery therefore anti-joins
specifically on ``provenance = 'tier_assign'`` -- the same fix pattern used by
pipeline/nodes/filter.py (``filters.provenance = 'filter_decide'``) and
pipeline/nodes/g2p.py (``g2p.provenance = 'g2p_node'``).

Tier-axis disambiguation
-------------------------
NOTE: the 'gold' / 'silver' tiers produced here reflect ASR *verification confidence*
(text_verified flag + inter-model agreement score).  This is a DIFFERENT axis from
the 'A' / 'B' (pretrain / clean) TTS-quality tier described in
docs/LABEL_FRAMEWORK_SPEC.md section 10.  That quality-grading tier is separate,
not-yet-built, future work.  Do not conflate the two.
"""

import logging
import time

log = logging.getLogger(__name__)

SILVER_AGREE_MIN = 0.65


def assign_tier(text_verified: bool, agreement: float) -> str:
    """Return the verification-confidence tier for a single segment.

    Rules (thresholds are production-verified -- do not change):
      - ``text_verified`` is True  -> "gold"
      - ``agreement`` >= SILVER_AGREE_MIN (0.65) -> "silver"
      - otherwise                  -> "excluded"

    An "excluded" row is always written so that discovery does not re-process
    the segment on every subsequent run (same 'always write a row, even on
    reject' precedent as g2p.py / filter.acoustic for unreadable-audio rows).
    Downstream manifest.build must filter ``tier IN ('gold', 'silver')`` to
    exclude these rows from the final manifest.
    """
    if text_verified:
        return "gold"
    elif agreement >= SILVER_AGREE_MIN:
        return "silver"
    else:
        return "excluded"


TIER_DISCOVER_SQL = """
    SELECT a.id, a.text_verified, a.agreement
    FROM asr_agreement a
    LEFT JOIN tiers t ON a.id = t.id AND t.provenance = 'tier_assign'
    WHERE t.id IS NULL
"""


def discover(conn) -> list[tuple]:
    return conn.execute(TIER_DISCOVER_SQL).fetchall()


async def run_tier_assign(*, batch_size: int = 5000, limit: int | None = None) -> dict:
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = connect()
    rows = discover(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"tier.assign: {len(rows)} segments to tier")
    if not rows:
        return {"processed": 0, "gold": 0, "silver": 0, "excluded": 0, "errors": 0}

    run_id = new_run_id("tier.assign")
    processed = 0
    gold = 0
    silver = 0
    excluded = 0
    t0 = time.time()

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        out_rows = []
        for seg_id, text_verified, agreement in batch:
            tier = assign_tier(bool(text_verified), float(agreement))
            out_rows.append({"id": seg_id, "tier": tier, "provenance": "tier_assign"})
            if tier == "gold":
                gold += 1
            elif tier == "silver":
                silver += 1
            else:
                excluded += 1
        upsert_rows(conn, "tiers", out_rows, ["id"])
        record_batch(conn, run_id, "tier.assign", [r["id"] for r in out_rows], "ok")
        processed += len(out_rows)
        rate = processed / (time.time() - t0) if time.time() > t0 else 0.0
        log.info(
            f"{processed}/{len(rows)} tiered ({rate:.1f}/s) "
            f"gold={gold} silver={silver} excluded={excluded}"
        )

    elapsed = time.time() - t0
    log.info(
        f"DONE: {processed} tiered in {elapsed:.0f}s "
        f"gold={gold} silver={silver} excluded={excluded} run_id={run_id}"
    )
    return {
        "processed": processed,
        "gold": gold,
        "silver": silver,
        "excluded": excluded,
        "errors": 0,
        "run_id": run_id,
    }
