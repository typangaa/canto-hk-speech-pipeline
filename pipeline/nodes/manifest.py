"""
pipeline/nodes/manifest.py
manifest.build / manifest.export -- P4 "metadata cutover" nodes that replace
scripts/09_manifest.py's file-glob-and-sidecar-read approach with a single catalog join.
The external interface is UNCHANGED (CLAUDE.md hard constraint #9 / zero-risk policy):
metadata/manifest.jsonl, metadata/train.jsonl, metadata/val.jsonl keep the exact schema in
docs/MANIFEST_SCHEMA.md; canto-tts reads the same files the same way. Only the build side
moved from data/filtered/*.json sidecars to DuckDB.

Eligibility (legacy-row-collision-safe)
----------------------------------------
A segment is included in the manifest iff it has rows in asr_agreement, g2p, filters, and
tiers, AND all four say "accepted" -- but "accepted" must be evaluated carefully because
the 455,299 P0-legacy-imported rows in filters/g2p predate the pass/valid_fraction/
provenance columns those nodes' P3 ports added (see filter.py's and g2p.py's own module
docstrings for the identical issue). A legacy row's mere EXISTENCE means "this segment was
already in data/filtered/ under the old pipeline", i.e. implicitly accepted; a
freshly-decided row (provenance = 'filter_decide' / 'g2p_node') must be checked explicitly
via its pass / valid_fraction column. Both cases are handled by the same
"(new-node-flag) OR (legacy-row-with-no-provenance)" OR-condition on filters and g2p below.
tiers needs no such OR-condition: EVERY existing tiers row (legacy or tier_assign-written)
already resolves to one of 'gold'/'silver'/'excluded', so `tier IN ('gold','silver')` alone
correctly includes both legacy and freshly-tiered rows and excludes tier.assign's
'excluded' sentinel.

Deliberately NOT joining the `speakers` table for speaker_id/gender
--------------------------------------------------------------------
`speakers` (pipeline/nodes/speaker.py's speaker.cluster output) currently only covers the
`rthk` source (a P3 S3 gate test, not a full-corpus run -- see
docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md §8 P3 S3 write-up). Using it here would silently
overwrite rthk's speaker_id with fresher (but different, ARI=0.82 vs legacy per that
write-up) cluster assignments while leaving every other source on the legacy value -- an
inconsistent, surprising partial change that would also blow up the P4 gate's "diff shows
only expected differences" requirement. `segments.speaker_id`/`segments.gender` (the plain
columns, populated by the P0 import for all 455,299 legacy rows) are used instead, matching
the CURRENT manifest.jsonl exactly. Switching manifest.build to consume `speakers` is a
reasonable future follow-up once ALL sources have a full speaker.cluster run, not before.

Text source
-----------
`text` comes from `asr_agreement.best_text`, matching manifest.jsonl's current field
(itself the "best" ASR candidate at filter time, not truly human-verified in almost all
cases -- see g2p.py's module docstring for the same "text_verified is tracked but not yet
enforced" observation; text_verified is still copied through into the manifest as its own
field, unchanged).

Train/val split: preserve, then extend
----------------------------------------
`train_val_split()` below deliberately does NOT re-run scripts/09_manifest.py's
stratified-by-source/held-out-speaker-tail split algorithm against the WHOLE manifest on
every export. That algorithm's speaker order was an accident of filesystem glob order, not
a principled or reproducible key -- re-running it from scratch every export would shuffle
which speakers are held out, silently leaking previously-held-out validation speakers into
train (and vice versa) on every re-export. Instead: existing ids keep whatever split they
were already in (read straight from the current train.jsonl/val.jsonl on disk), and only
GENUINELY NEW ids (not present in either file) run through the legacy stratification
algorithm to decide their split. This is the only way to satisfy the P4 gate's "diff shows
only expected differences" criterion for train/val membership.

Output ordering
----------------
manifest.jsonl is written sorted by `id` (matches scripts/09_manifest.py's own
`sorted(entries, key=lambda x: x["id"])`, so an unchanged corpus produces a byte-identical
file). train.jsonl/val.jsonl are ALSO written sorted by id here (the legacy script wrote
them in glob-then-by-speaker order, an unreconstructable and non-meaningful order) -- this
is a deliberate, documented formatting change: split MEMBERSHIP is preserved 1:1, but line
ORDER within each split file is not byte-identical to a pre-P4 export. `docs/MANIFEST_SCHEMA.md`
does not require any particular line order, only correct membership/no-speaker-leakage.
"""

import json
import logging
import time
from collections import defaultdict
from datetime import date

log = logging.getLogger(__name__)

# Matches scripts/09_manifest.py's SILVER_AGREE_MIN-derived tier gate: gold/auto_gold/silver/
# bronze segments enter the manifest. tier.assign's "excluded" sentinel (agreement < 0.70 and
# not human-verified) is deliberately not in this tuple. auto_gold added 2026-07-10, bronze
# added + thresholds raised 2026-07-11 (DECISIONS.md, docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md)
# -- auto_gold/bronze are statistical-confidence tiers, see pipeline/nodes/tier.py's module
# docstring; NOT equivalent to text_verified.
INCLUDED_TIERS = ("gold", "auto_gold", "silver", "bronze")

# Best-to-worst precedence, matching pipeline/nodes/tier.py's assign_tier() cascade. Used by
# min_tier (added 2026-07-11) to compute an "at-or-above" tier cut -- e.g. min_tier='auto_gold'
# includes {'gold', 'auto_gold'} (gold is strictly better, so it's always included alongside
# whatever floor tier is requested), matching min_agreement's >= semantics.
TIER_PRECEDENCE = ("gold", "auto_gold", "silver", "bronze")


def _tiers_at_or_above(min_tier: str) -> tuple[str, ...]:
    if min_tier not in TIER_PRECEDENCE:
        raise ValueError(f"min_tier must be one of {TIER_PRECEDENCE}, got {min_tier!r}")
    idx = TIER_PRECEDENCE.index(min_tier)
    return TIER_PRECEDENCE[: idx + 1]


# min_agreement (added 2026-07-10) lets a caller cut a smaller, higher-confidence subset of
# the manifest by asr_agreement.agreement -- e.g. --min-agreement 0.95 for a ~100h "cleanest"
# export, 0.85 for a ~500h export, or omitted (None) for the full ~1000h pool at today's
# existing silver bar. See docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md for the hours-per-threshold
# data behind these numbers. Implemented as `(? IS NULL OR a.agreement >= ?)` so the same query
# serves both the default (unfiltered) and cut exports -- no separate SQL string needed.
#
# min_tier (added 2026-07-11) lets a caller cut by the tier column directly instead of/as well
# as raw agreement -- e.g. min_tier='auto_gold' for exactly the segments tier.assign already
# classified gold-equivalent-or-better (agreement>=0.95 AND canto_ft_confidence>0.8, OR
# human-verified). This is NOT the same as min_agreement=0.95: a segment can have agreement>=0.95
# but fail the canto_ft_confidence gate and land in 'silver' instead of 'auto_gold' -- min_tier
# reads the tier tier.assign already computed rather than re-deriving a agreement-only cut.
# The `{tier_list}` placeholder is filled via .format() with a validated (against
# TIER_PRECEDENCE) `?`-placeholder list, never raw user SQL -- see discover().
MANIFEST_DISCOVER_SQL = """
    SELECT
        s.id, s.audio_path, s.source, s.source_url, s.program, s.domain,
        s.duration_sec, s.sample_rate, s.speaker_id, s.gender, s.style, s.created_at,
        a.best_text, a.text_verified, a.agreement,
        g.jyutping,
        f.snr_db, f.dnsmos, f.english_ratio,
        t.tier
    FROM segments s
    JOIN asr_agreement a ON s.id = a.id
    JOIN g2p g ON s.id = g.id
    JOIN filters f ON s.id = f.id
    JOIN tiers t ON s.id = t.id
    WHERE
        (f.pass = TRUE OR (f.pass IS NULL AND f.provenance IS NULL))
        AND (g.valid_fraction >= 0.80 OR (g.valid_fraction IS NULL AND g.provenance IS NULL))
        AND t.tier IN ({tier_list})
        AND g.jyutping IS NOT NULL AND g.jyutping != ''
        AND (? IS NULL OR a.agreement >= ?)
    ORDER BY s.id
"""


def _fetch_asr_candidates_by_id(conn) -> dict[str, list[dict]]:
    """One dict, built once per run, reused for every manifest row -- cheaper than a
    per-id query for 455k+ segments. Order matches scripts/04_transcribe.py's ASR_MODELS
    dict order (canto_ft, whisper_v3), not alphabetical, to match the existing
    manifest.jsonl's asr_candidates ordering exactly (minimises the P4 gate diff)."""
    from pipeline.nodes.asr import ASR_MODELS, model_field

    model_order = {model_field(key): i for i, key in enumerate(ASR_MODELS)}
    by_id: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for seg_id, model, text, confidence in conn.execute(
        "SELECT id, model, text, confidence FROM asr_results"
    ).fetchall():
        rank = model_order.get(model, len(model_order))
        by_id[seg_id].append((rank, {"model": model, "text": text, "confidence": confidence}))

    return {
        seg_id: [cand for _, cand in sorted(candidates, key=lambda pair: pair[0])]
        for seg_id, candidates in by_id.items()
    }


def build_entry(row: tuple, asr_candidates: list[dict]) -> dict:
    """Pure logic: one catalog row + its asr_candidates -> one manifest.jsonl entry.
    Field order and rounding match docs/MANIFEST_SCHEMA.md exactly."""
    (
        seg_id, audio_path, source, source_url, program, domain,
        duration_sec, sample_rate, speaker_id, gender, style, created_at,
        best_text, text_verified, agreement,
        jyutping,
        snr_db, dnsmos, english_ratio,
        tier,
    ) = row

    return {
        "id": seg_id,
        "audio_path": audio_path,
        "source": source or "",
        "source_url": source_url or "",
        "program": program or "",
        "domain": domain or "other",
        "text": best_text or "",
        "text_verified": bool(text_verified),
        "asr_candidates": asr_candidates,
        "asr_agreement": round(float(agreement), 3),
        "jyutping": jyutping,
        "duration_sec": round(float(duration_sec), 3),
        "sample_rate": int(sample_rate),
        "speaker_id": speaker_id or f"{source}_unk",
        "gender": gender or "unknown",
        "style": style or "formal",
        "snr_db": round(float(snr_db), 1),
        "dnsmos": round(float(dnsmos), 2),
        "english_ratio": round(float(english_ratio), 3),
        "created_at": str(created_at) if created_at else str(date.today()),
        "tier": tier,
    }


def discover(conn, min_agreement: float | None = None, min_tier: str | None = None) -> list[tuple]:
    tiers = _tiers_at_or_above(min_tier) if min_tier is not None else INCLUDED_TIERS
    tier_list = ", ".join("?" for _ in tiers)
    sql = MANIFEST_DISCOVER_SQL.format(tier_list=tier_list)
    return conn.execute(sql, [*tiers, min_agreement, min_agreement]).fetchall()


def build_manifest(
    conn, min_agreement: float | None = None, min_tier: str | None = None
) -> list[dict]:
    """Runs the full catalog join and returns every eligible manifest entry, sorted by id."""
    rows = discover(conn, min_agreement, min_tier)
    candidates_by_id = _fetch_asr_candidates_by_id(conn)
    return [build_entry(row, candidates_by_id.get(row[0], [])) for row in rows]


def run_manifest_build(
    *,
    limit: int | None = None,
    min_agreement: float | None = None,
    min_tier: str | None = None,
) -> dict:
    """Synchronous, CLI-style -- builds the manifest in-memory and returns it plus a
    summary; does not write files (see run_manifest_export for that). Kept separate from
    export so a caller can inspect/validate the built list before committing it to disk.

    min_agreement: optional asr_agreement.agreement cutoff for a smaller, higher-confidence
    subset (see MANIFEST_DISCOVER_SQL's comment / docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md).
    min_tier: optional tier-column cutoff, e.g. 'auto_gold' for {'gold','auto_gold'} only --
    see TIER_PRECEDENCE / _tiers_at_or_above(). Both default to None = full pool, unchanged
    behaviour; the two can be combined (e.g. min_tier='silver' AND min_agreement=0.90 for an
    extra-strict cut within the silver-or-better tiers)."""
    from pipeline.catalog.catalog import connect_ro

    t0 = time.time()
    conn = connect_ro()
    entries = build_manifest(conn, min_agreement, min_tier)
    conn.close()
    if limit:
        entries = entries[:limit]

    total_hours = sum(e["duration_sec"] for e in entries) / 3600
    n_speakers = len(set(e["speaker_id"] for e in entries))
    tier_counts: dict[str, int] = defaultdict(int)
    for e in entries:
        tier_counts[e["tier"]] += 1

    elapsed = time.time() - t0
    log.info(
        f"manifest.build: {len(entries)} entries ({total_hours:.1f}h, {n_speakers} speakers) "
        f"in {elapsed:.1f}s -- gold={tier_counts.get('gold', 0)} auto_gold={tier_counts.get('auto_gold', 0)} "
        f"silver={tier_counts.get('silver', 0)} bronze={tier_counts.get('bronze', 0)}"
        + (f" [min_agreement={min_agreement}]" if min_agreement is not None else "")
        + (f" [min_tier={min_tier}]" if min_tier is not None else "")
    )
    return {
        "entries": entries,
        "count": len(entries),
        "total_hours": round(total_hours, 1),
        "n_speakers": n_speakers,
        "tier_counts": dict(tier_counts),
    }


# ---------------------------------------------------------------------------
# Train/val split -- preserve existing membership, extend for new ids only
# ---------------------------------------------------------------------------

def _load_existing_ids(path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def _stratified_split_new_entries(new_entries: list[dict], val_frac: float) -> tuple[list[dict], list[dict]]:
    """Verbatim port of scripts/09_manifest.py's train_val_split() -- stratified by
    source, no speaker_id split across train/val -- but applied ONLY to genuinely new
    entries (ids not already present in the existing train.jsonl/val.jsonl on disk)."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for e in new_entries:
        by_source[e["source"]].append(e)

    train, val = [], []
    for _src, src_entries in by_source.items():
        by_speaker: dict[str, list[dict]] = defaultdict(list)
        for e in src_entries:
            by_speaker[e["speaker_id"]].append(e)

        speakers = list(by_speaker.keys())
        n_val_spk = max(1, int(len(speakers) * val_frac))
        val_speakers = set(speakers[-n_val_spk:])

        for spk, segs in by_speaker.items():
            if spk in val_speakers:
                val.extend(segs)
            else:
                train.extend(segs)

    return train, val


def train_val_split(
    entries: list[dict], *, train_path, val_path, val_frac: float = 0.05
) -> tuple[list[dict], list[dict]]:
    """Existing ids keep whatever split they were already assigned (read from disk);
    only ids not seen in either existing file are freshly split. See module docstring
    ("Train/val split: preserve, then extend") for why this matters."""
    existing_train_ids = _load_existing_ids(train_path)
    existing_val_ids = _load_existing_ids(val_path)

    train, val, new_entries = [], [], []
    for e in entries:
        if e["id"] in existing_train_ids:
            train.append(e)
        elif e["id"] in existing_val_ids:
            val.append(e)
        else:
            new_entries.append(e)

    if new_entries:
        new_train, new_val = _stratified_split_new_entries(new_entries, val_frac)
        train.extend(new_train)
        val.extend(new_val)
        log.info(f"train/val split: {len(new_entries)} new ids split ({len(new_train)} train, {len(new_val)} val)")

    return train, val


def _agreement_tag(min_agreement: float) -> str:
    """0.95 -> 'agree095', 0.855 -> 'agree086' (rounded) -- used to name a cut export's
    files distinctly from the default manifest.jsonl/train.jsonl/val.jsonl."""
    return f"agree{round(min_agreement * 100):03d}"


def _export_tag(min_agreement: float | None, min_tier: str | None) -> str | None:
    """Combines min_agreement/min_tier into one filename tag, e.g. min_tier='auto_gold' ->
    'tier_auto_gold'; both set -> 'tier_auto_gold_agree090'. None if neither is set (the
    default, unfiltered export keeps the original filenames -- see run_manifest_export)."""
    parts = []
    if min_tier is not None:
        parts.append(f"tier_{min_tier}")
    if min_agreement is not None:
        parts.append(_agreement_tag(min_agreement))
    return "_".join(parts) if parts else None


def run_manifest_export(
    *,
    limit: int | None = None,
    dry_run: bool = False,
    min_agreement: float | None = None,
    min_tier: str | None = None,
) -> dict:
    """Builds the manifest then writes metadata/manifest.jsonl + train.jsonl + val.jsonl.
    manifest.jsonl is sorted by id; train/val preserve existing split membership (see
    train_val_split) and are also written sorted by id (see module docstring's
    "Output ordering" section for why this is not a byte-identical reorder of the legacy
    files, only a membership-preserving one).

    min_agreement (added 2026-07-10) / min_tier (added 2026-07-11): when both are None
    (default), writes the exact same three filenames as always -- CLAUDE.md hard constraint
    #9 (zero-risk / external interface unchanged) requires this default call to stay
    byte-compatible with what canto-tts already reads. When either is set, writes to
    SEPARATE files instead (e.g. manifest_tier_auto_gold.jsonl / train_tier_auto_gold.jsonl /
    val_tier_auto_gold.jsonl for min_tier='auto_gold'; see _export_tag()), so a cut export
    never overwrites the full-pool manifest. Each cut's train/val split membership is
    tracked independently against its own prior train_<tag>.jsonl/val_<tag>.jsonl (same
    preserve-then-extend logic as the default export, just scoped to that cut's own files)."""
    from pipeline.config import MANIFEST_PATH, TRAIN_PATH, VAL_PATH, VAL_FRAC

    result = run_manifest_build(limit=limit, min_agreement=min_agreement, min_tier=min_tier)
    entries = result["entries"]

    tag = _export_tag(min_agreement, min_tier)
    if tag is not None:
        manifest_path = MANIFEST_PATH.with_name(f"manifest_{tag}.jsonl")
        train_path = TRAIN_PATH.with_name(f"train_{tag}.jsonl")
        val_path = VAL_PATH.with_name(f"val_{tag}.jsonl")
    else:
        manifest_path, train_path, val_path = MANIFEST_PATH, TRAIN_PATH, VAL_PATH

    train, val = train_val_split(entries, train_path=train_path, val_path=val_path, val_frac=VAL_FRAC)
    train.sort(key=lambda e: e["id"])
    val.sort(key=lambda e: e["id"])

    if dry_run:
        log.info(
            f"[DRY-RUN] would write {len(entries)} entries "
            f"({result['total_hours']}h) to {manifest_path} -- train={len(train)} val={len(val)}"
        )
        return {**result, "train_count": len(train), "val_count": len(val), "written": False}

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for e in entries:  # already sorted by id via MANIFEST_DISCOVER_SQL's ORDER BY
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(train_path, "w") as f:
        for e in train:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for e in val:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    log.info(
        f"DONE: wrote {len(entries)} entries ({result['total_hours']}h) to {manifest_path} "
        f"-- train={len(train)}, val={len(val)}"
    )
    return {**result, "train_count": len(train), "val_count": len(val), "written": True}
