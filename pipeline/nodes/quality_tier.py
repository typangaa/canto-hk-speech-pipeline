"""
pipeline/nodes/quality_tier.py
quality_tier.assign DAG node -- builds the A/B TTS-quality axis from
docs/LABEL_FRAMEWORK_SPEC.md section 10 (Tier A = pretrain, permissive; Tier B =
clean fine-tune, strict). Writes one `quality_tiers` row per segment.

**Different axis from `tiers`/`tier.assign`.** `tiers.tier` (gold/auto_gold/silver/
bronze/excluded) is a *verification-confidence* axis -- how much we trust the
transcribed TEXT. `quality_tiers.quality_tier` (A/B) is an *acoustic-cleanliness*
axis -- how usable the AUDIO is for TTS training, independent of text trust. Do not
conflate the two; see tier.py's module docstring for the same disambiguation from
the other side.

Scope (owner decision, 2026-07-16, made because canto-tts training is about to
consume this and only wants the verification-confidence top band): this node only
tiers segments already at `tiers.tier IN ('gold', 'auto_gold')` AND `filters.pass =
TRUE` -- NOT the full manifest-eligible pool (which also includes silver/bronze).
Segments outside that scope never get a `quality_tiers` row and are implicitly
excluded from both A and B exports.

Tier A ("pretrain") is the *entire* gold+auto_gold scope above -- every row this
node writes gets at least 'A'. Tier B ("clean") is a strict subset of A (B implies
A, not a disjoint bucket), gated on three label-store/filter signals together:
  - `filters.dnsmos`       >= B_DNSMOS_MIN  (3.7 -- above the auto_gold floor of 3.5)
  - `labels_music.music_prob`   < B_MUSIC_MAX   (0.10)
  - `labels_overlap.overlap_ratio` < B_OVERLAP_MAX (0.05)
Thresholds are the owner-picked "strict" bundle from a real distribution check
against the gold+auto_gold pool (2026-07-16): measured 55,596 segments / 152.1h at
this bar, vs. 279,285 segs / 640.9h for the full Tier-A scope. See DECISIONS.md
2026-07-16 for the full bundle comparison (loose/medium/strict) presented to the
owner.

A segment missing a `labels_music`/`labels_overlap` row (label.suite not yet run
for it -- ~3-5% of the scope as of 2026-07-16) fails the Tier B gate closed (can't
verify cleanliness) but still gets Tier A -- same "fail closed on missing signal,
never fail closed on the base tier" pattern as tier.assign's auto_gold gate.

Legacy-row-collision note
--------------------------
`quality_tiers` is a brand-new table (no P0 legacy import touched it), so a bare
row-existence anti-join would be correct on its own -- but this node still tags
`provenance = 'quality_tier_assign'` for consistency with every other node in this
codebase, and in case a future backfill needs to distinguish its own writes from a
hypothetical future direct-write path (e.g. a human-QA override, mirroring
calibrate.py's record_decision() direct writes to `tiers`).
"""

import logging
import time

log = logging.getLogger(__name__)

B_DNSMOS_MIN = 3.7
B_MUSIC_MAX = 0.10
B_OVERLAP_MAX = 0.05


def assign_quality_tier(
    dnsmos: float | None,
    music_prob: float | None,
    overlap_ratio: float | None,
) -> str:
    """Return 'B' (clean) if all three acoustic-cleanliness gates pass, else 'A'
    (pretrain -- the base tier every scoped segment gets). Any missing signal
    (None) fails the B gate closed, same as tier.assign's auto_gold dnsmos gate."""
    if (
        dnsmos is not None
        and dnsmos >= B_DNSMOS_MIN
        and music_prob is not None
        and music_prob < B_MUSIC_MAX
        and overlap_ratio is not None
        and overlap_ratio < B_OVERLAP_MAX
    ):
        return "B"
    return "A"


QUALITY_TIER_DISCOVER_SQL = """
    SELECT s.id, f.dnsmos, lm.music_prob, lo.overlap_ratio
    FROM segments s
    JOIN tiers t ON t.id = s.id AND t.tier IN ('gold', 'auto_gold')
    JOIN filters f ON f.id = s.id AND f.pass = TRUE
    LEFT JOIN labels_music lm ON lm.id = s.id
    LEFT JOIN labels_overlap lo ON lo.id = s.id
    LEFT JOIN quality_tiers qt ON qt.id = s.id AND qt.provenance = 'quality_tier_assign'
    WHERE qt.id IS NULL
"""


def discover(conn) -> list[tuple]:
    return conn.execute(QUALITY_TIER_DISCOVER_SQL).fetchall()


async def run_quality_tier_assign(*, conn=None, batch_size: int = 5000, limit: int | None = None) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many`. Defaults to a fresh
    self-managed connect() for standalone `pipe run quality_tier.assign` usage."""
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    rows = discover(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"quality_tier.assign: {len(rows)} segments to tier")
    if not rows:
        return {"processed": 0, "tier_a": 0, "tier_b": 0, "errors": 0}

    run_id = new_run_id("quality_tier.assign")
    processed = 0
    tier_a = 0
    tier_b = 0
    t0 = time.time()

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        out_rows = []
        for seg_id, dnsmos, music_prob, overlap_ratio in batch:
            quality_tier = assign_quality_tier(dnsmos, music_prob, overlap_ratio)
            out_rows.append({"id": seg_id, "quality_tier": quality_tier, "provenance": "quality_tier_assign"})
            if quality_tier == "B":
                tier_b += 1
            else:
                tier_a += 1
        upsert_rows(conn, "quality_tiers", out_rows, ["id"])
        record_batch(conn, run_id, "quality_tier.assign", [r["id"] for r in out_rows], "ok")
        processed += len(out_rows)
        rate = processed / (time.time() - t0) if time.time() > t0 else 0.0
        log.info(f"{processed}/{len(rows)} quality-tiered ({rate:.1f}/s) A={tier_a} B={tier_b}")

    elapsed = time.time() - t0
    log.info(f"DONE: {processed} quality-tiered in {elapsed:.0f}s A={tier_a} B={tier_b} run_id={run_id}")
    return {"processed": processed, "tier_a": tier_a, "tier_b": tier_b, "errors": 0, "run_id": run_id}
