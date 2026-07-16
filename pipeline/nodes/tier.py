"""
pipeline/nodes/tier.py
tier.assign DAG node -- assigns each segment a verification-confidence tier
("gold" / "auto_gold" / "silver" / "bronze" / "excluded"), ported from the inline
tier logic inside scripts/09_manifest.py's build_entry() function into its own
catalog-driven node that writes one `tiers` table row per segment, independent of
manifest building. Runs entirely in-supervisor (no worker subprocess, no GPU).

auto_gold (added 2026-07-10, thresholds raised 2026-07-11, gate rebuilt 2026-07-15 --
owner decision -- see DECISIONS.md and docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md): a
statistical-confidence tier for segments that were NEVER human-reviewed but clear a
high enough cross-model agreement + acoustic-quality bar that treating them as
gold-equivalent is defensible without 100%-reviewing a 1000h+ corpus by hand. It
deliberately does NOT set asr_agreement.text_verified (that stays a strict "a human
actually looked at this" signal) -- auto_gold is sample-QA'd via
calibrate.sample(tier='auto_gold') + calibrate_server, not exhaustively reviewed. Do
not treat tier=='auto_gold' as equivalent to text_verified==True anywhere downstream.

**2026-07-15 gate rebuild (T16, pending_task.md / docs/PIPELINE_REVIEW_2026-07-13.md
Issue #17)**: the original auto_gold gate's confidence signal was `canto_ft`'s own
logprob-derived confidence -- `canto_ft` retired 2026-07-13, so that column is now
always NULL for new segments and the gate failed closed (new segments capped at
silver/bronze). Two changes landed together: (1) `asr_agreement.agreement` was
backfilled corpus-wide using the already-shipped char_agreement() punctuation/digit
normalization (Issue #20) and excluding canto_ft from the comparison -- previously
existing rows were computed pre-normalization/3-way and systematically understated
agreement (2026-07-10 whisper_v3-era rows and earlier). (2) The confidence gate was
replaced with `filters.dnsmos` (an existing, zero-new-compute acoustic-quality signal)
as the third, non-ASR trust signal recommended by targeted external research (2-model
text agreement alone is an insufficient auto-trust signal per industry consensus --
GigaSpeech 2 / Emilia-Pipe-style pipelines layer LID/DNSMOS on top of ASR agreement;
see docs/PIPELINE_REVIEW_2026-07-13.md §5). Owner-picked bundle: "Balanced" --
AUTO_GOLD_AGREE_MIN 0.95->0.92, AUTO_GOLD_DNSMOS_MIN=3.5 (new), silver/bronze
unchanged (0.85/0.70) since normalization alone already substantially grew those
pools. This is provisional pending T1 pilot QA ground-truth validation (still 0/~900
reviewed as of this decision) -- revisit once real precision data exists.

bronze (added 2026-07-11, owner decision): a lower-confidence band below silver,
still manifest-eligible but flagged as the noisiest of the three statistical tiers
-- QA'd at a higher sample rate than silver/auto_gold (see calibrate.py's
QA_SAMPLE_RATE_BY_TIER) precisely because it is the least trustworthy tier admitted
into the manifest at all. Introducing it also RAISED the silver/auto_gold bars
(silver 0.65->0.85, auto_gold 0.90->0.95) and moved the manifest-eligibility floor
up from 0.65 to 0.70 -- segments with agreement in [0.65, 0.70) that used to be
'silver' are now 'excluded'. This is a stricter, more conservative re-cut of the
same corpus, not an additive change; see DECISIONS.md 2026-07-11 for the before/after
segment and hour counts.

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

SILVER_AGREE_MIN = 0.85
BRONZE_AGREE_MIN = 0.70
AUTO_GOLD_AGREE_MIN = 0.92
AUTO_GOLD_DNSMOS_MIN = 3.5


def assign_tier(text_verified: bool, agreement: float, dnsmos: float | None = None) -> str:
    """Return the verification-confidence tier for a single segment.

    Rules (thresholds are production-verified -- do not change without an
    owner decision + re-running docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md's analysis):
      - ``text_verified`` is True (human-reviewed via calibrate_server) -> "gold";
        wins even if agreement/dnsmos would also qualify for auto_gold.
      - ``agreement`` >= AUTO_GOLD_AGREE_MIN (0.92) AND
        ``dnsmos`` >= AUTO_GOLD_DNSMOS_MIN (3.5) -> "auto_gold"
        (statistical confidence, NOT human-reviewed -- see module docstring's
        2026-07-15 gate rebuild note for why dnsmos replaced canto_ft_confidence).
      - ``agreement`` >= SILVER_AGREE_MIN (0.85) -> "silver"
      - ``agreement`` >= BRONZE_AGREE_MIN (0.70) -> "bronze"
      - otherwise                  -> "excluded"

    dnsmos may be None (e.g. filters row not yet written for this id) -- treated
    as failing the auto_gold gate, same as any dnsmos < 3.5 (fails closed).

    An "excluded" row is always written so that discovery does not re-process
    the segment on every subsequent run (same 'always write a row, even on
    reject' precedent as g2p.py / filter.acoustic for unreadable-audio rows).
    Downstream manifest.build must filter ``tier IN ('gold', 'auto_gold', 'silver', 'bronze')``
    to exclude these rows from the final manifest.
    """
    if text_verified:
        return "gold"
    elif agreement >= AUTO_GOLD_AGREE_MIN and (dnsmos or 0.0) >= AUTO_GOLD_DNSMOS_MIN:
        return "auto_gold"
    elif agreement >= SILVER_AGREE_MIN:
        return "silver"
    elif agreement >= BRONZE_AGREE_MIN:
        return "bronze"
    else:
        return "excluded"


TIER_DISCOVER_SQL = """
    SELECT a.id, a.text_verified, a.agreement, f.dnsmos
    FROM asr_agreement a
    LEFT JOIN filters f ON a.id = f.id
    LEFT JOIN tiers t ON a.id = t.id AND t.provenance = 'tier_assign'
    WHERE t.id IS NULL
"""


def discover(conn) -> list[tuple]:
    return conn.execute(TIER_DISCOVER_SQL).fetchall()


async def run_tier_assign(*, conn=None, batch_size: int = 5000, limit: int | None = None) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run tier.assign` usage."""
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    rows = discover(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"tier.assign: {len(rows)} segments to tier")
    if not rows:
        return {"processed": 0, "gold": 0, "auto_gold": 0, "silver": 0, "bronze": 0, "excluded": 0, "errors": 0}

    run_id = new_run_id("tier.assign")
    processed = 0
    gold = 0
    auto_gold = 0
    silver = 0
    bronze = 0
    excluded = 0
    t0 = time.time()

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        out_rows = []
        for seg_id, text_verified, agreement, dnsmos in batch:
            tier = assign_tier(bool(text_verified), float(agreement), dnsmos)
            out_rows.append({"id": seg_id, "tier": tier, "provenance": "tier_assign"})
            if tier == "gold":
                gold += 1
            elif tier == "auto_gold":
                auto_gold += 1
            elif tier == "silver":
                silver += 1
            elif tier == "bronze":
                bronze += 1
            else:
                excluded += 1
        upsert_rows(conn, "tiers", out_rows, ["id"])
        record_batch(conn, run_id, "tier.assign", [r["id"] for r in out_rows], "ok")
        processed += len(out_rows)
        rate = processed / (time.time() - t0) if time.time() > t0 else 0.0
        log.info(
            f"{processed}/{len(rows)} tiered ({rate:.1f}/s) "
            f"gold={gold} auto_gold={auto_gold} silver={silver} bronze={bronze} excluded={excluded}"
        )

    elapsed = time.time() - t0
    log.info(
        f"DONE: {processed} tiered in {elapsed:.0f}s "
        f"gold={gold} auto_gold={auto_gold} silver={silver} bronze={bronze} excluded={excluded} run_id={run_id}"
    )
    return {
        "processed": processed,
        "gold": gold,
        "auto_gold": auto_gold,
        "silver": silver,
        "bronze": bronze,
        "excluded": excluded,
        "errors": 0,
        "run_id": run_id,
    }
