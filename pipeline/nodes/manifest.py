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

P3 pause-token fields (docs/PAUSE_TOKEN_PUNCTUATION_PLAN.md P3)
-----------------------------------------------------------------
Two ADDITIVE fields, LEFT JOINed off `pause_plan` (pipeline/nodes/pause_plan.py, P2):
canonical `text` is never touched (owner decision (1), frozen 2026-07-21).
  - `text_pause`: a derived copy of `text` with `<pause-short>`/`<pause-long>` literal
    tokens inserted after mid-sentence punctuation marks the frozen
    metadata/labels/pause_calibration.json bucket_rule verdicts as short/long; marks
    verdicted `no_pause` (LM-hallucinated, no acoustic basis) are stripped from this
    field only; `trailing_tail`/`leading_tail` marks (no verdict) are left untouched.
    See `build_text_pause()` below.
  - `punct_audit`: `{n_punct, n_no_pause, n_short, n_long}` summary copied straight from
    `pause_plan`'s own precomputed columns.
`pause_plan` only has rows for the gold/auto_gold scope P0/P2 were run against -- silver/
bronze rows have no `pause_plan` match (LEFT JOIN -> NULL), so both fields are OMITTED
entirely for those rows (LABEL_FRAMEWORK_SPEC.md §7 "unreliable -> omit, never null").
A `pause_plan` row with `unalignable=TRUE` still joins (plan='[]', all counts 0) --
`text_pause` degrades to an exact copy of `text` and `punct_audit` reports all-zero
counts, which is itself informative (mirrors label_store.py's "empty list is
informative" treatment of prosody gaps), not omitted.
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
# classified gold-equivalent-or-better (agreement>=0.92 AND dnsmos>=3.5, OR human-verified --
# gate rebuilt 2026-07-15, see pipeline/nodes/tier.py's module docstring). This is NOT the
# same as min_agreement=0.92: a segment can have agreement>=0.92 but fail the dnsmos gate and
# land in 'silver' instead of 'auto_gold' -- min_tier reads the tier tier.assign already
# computed rather than re-deriving an agreement-only cut.
# The `{tier_list}` placeholder is filled via .format() with a validated (against
# TIER_PRECEDENCE) `?`-placeholder list, never raw user SQL -- see discover().
# code_switch (added 2026-07-15, T18 / pending_task.md): lets a caller cut by
# filters.english_ratio -- 'only' for code-switched segments (english_ratio > 0, e.g. for
# a dedicated QA/eval subset), 'exclude' for pure-Cantonese-only (english_ratio = 0, e.g.
# for a monolingual curriculum-training stage). None (default) = no filter, same as every
# other cut here. See docs/PIPELINE_REVIEW... T16 analysis: code-switched segments clear
# ASR-agreement thresholds far less often than pure-Cantonese ones (systematic AR-vs-CTC
# English-transliteration divergence, not necessarily a quality signal) -- this cut lets
# downstream training stratify/curriculum around that without physically forking the corpus
# (CLAUDE.md hard constraint: code-switching is a natural HK Cantonese feature, kept in the
# single default export; this is purely an opt-in slicing tool).
CODE_SWITCH_CONDITIONS = {
    None: "",
    "only": "AND f.english_ratio > 0",
    "exclude": "AND f.english_ratio = 0",
}

# min_quality_tier (added 2026-07-16, T13): cuts by the SEPARATE A/B acoustic-cleanliness
# axis (pipeline/nodes/quality_tier.py), not the tiers.tier verification-confidence axis
# above. `quality_tiers` only has rows for the gold/auto_gold scope that node was pointed
# at (owner decision) -- a LEFT JOIN, since most manifest-eligible rows (silver/bronze)
# have no quality_tiers row at all and must stay included when this filter is unused
# (None, the default). 'B' (strict/clean) implies 'A' is NOT stored redundantly -- a
# segment gets exactly one value, its best-earned grade -- so "at or above 'A'" means
# "has ANY quality_tiers row" (everything the node scored), and "at or above 'B'" means
# "quality_tier = 'B'" only. Same best-to-worst precedence-list pattern as
# TIER_PRECEDENCE/_tiers_at_or_above() above, mirrored for this axis.
QUALITY_TIER_PRECEDENCE = ("B", "A")


def _quality_tiers_at_or_above(min_quality_tier: str) -> tuple[str, ...]:
    if min_quality_tier not in QUALITY_TIER_PRECEDENCE:
        raise ValueError(
            f"min_quality_tier must be one of {QUALITY_TIER_PRECEDENCE}, got {min_quality_tier!r}"
        )
    idx = QUALITY_TIER_PRECEDENCE.index(min_quality_tier)
    return QUALITY_TIER_PRECEDENCE[: idx + 1]


MANIFEST_DISCOVER_SQL = """
    SELECT
        s.id, s.audio_path, s.source, s.source_url, s.program, s.domain,
        s.duration_sec, s.sample_rate, s.speaker_id, s.gender, s.style, s.created_at,
        a.best_text, a.text_verified, a.agreement,
        g.jyutping, g.jyutping_cs,
        f.snr_db, f.dnsmos, f.english_ratio,
        t.tier,
        pp.plan, pp.n_punct, pp.n_no_pause, pp.n_short, pp.n_long
    FROM segments s
    JOIN asr_agreement a ON s.id = a.id
    JOIN g2p g ON s.id = g.id
    JOIN filters f ON s.id = f.id
    JOIN tiers t ON s.id = t.id
    LEFT JOIN quality_tiers qt ON s.id = qt.id
    LEFT JOIN pause_plan pp ON s.id = pp.id AND pp.provenance = 'pause_plan'
    WHERE
        (f.pass = TRUE OR (f.pass IS NULL AND f.provenance IS NULL))
        AND (g.valid_fraction >= 0.80 OR (g.valid_fraction IS NULL AND g.provenance IS NULL))
        AND t.tier IN ({tier_list})
        AND g.jyutping IS NOT NULL AND g.jyutping != ''
        AND (? IS NULL OR a.agreement >= ?)
        {code_switch_cond}
        {quality_tier_cond}
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


# P3 pause-token literal vocab tokens (docs/PAUSE_TOKEN_CALIBRATION_HANDOFF.md §2.3 --
# canto-tts's engine-side vocab already reserves these two exact strings).
PAUSE_TOKENS = {"short": "<pause-short>", "long": "<pause-long>"}


def build_text_pause(text: str, plan: list[dict] | None) -> str:
    """Derive the P3 `text_pause` convenience field from `text` + a `pause_plan.plan`
    list (see module docstring's "P3 pause-token fields" section). Never mutates or
    reads back `text` in-place -- always returns a fresh string; `text` itself
    (the canonical field) is never touched by this function's caller.

    Per event (kind="normal" only get a verdict):
      - verdict="no_pause": the punctuation mark is dropped (no acoustic basis).
      - verdict="short"/"long": mark is kept, `<pause-short>`/`<pause-long>` is
        appended immediately after it.
      - no verdict (trailing_tail/leading_tail): mark is left untouched.

    Defensive against text/plan drift (e.g. a gold segment's `best_text` edited by
    `calibrate serve` after the P0 alignment that produced `plan` ran): any event
    whose recorded `mark` no longer matches the character at its `offset` in `text`
    is silently skipped, so a stale plan can only under-annotate, never corrupt text.
    """
    if not plan:
        return text

    events_by_offset: dict[int, dict] = {}
    for ev in plan:
        offset = ev.get("offset")
        mark = ev.get("mark")
        if offset is None or not (0 <= offset < len(text)):
            continue
        if text[offset] != mark:
            continue  # drift guard -- skip rather than corrupt
        events_by_offset[offset] = ev

    if not events_by_offset:
        return text

    out: list[str] = []
    for i, ch in enumerate(text):
        ev = events_by_offset.get(i)
        if ev is None:
            out.append(ch)
            continue
        verdict = ev.get("verdict")
        if verdict is None:
            out.append(ch)  # trailing_tail / leading_tail: untouched
            continue
        if verdict == "no_pause":
            continue  # strip -- no acoustic basis (pause_calibration.json)
        out.append(ch)
        token = PAUSE_TOKENS.get(verdict)
        if token:
            out.append(token)
    return "".join(out)


def build_entry(row: tuple, asr_candidates: list[dict]) -> dict:
    """Pure logic: one catalog row + its asr_candidates -> one manifest.jsonl entry.
    Field order and rounding match docs/MANIFEST_SCHEMA.md exactly."""
    (
        seg_id, audio_path, source, source_url, program, domain,
        duration_sec, sample_rate, speaker_id, gender, style, created_at,
        best_text, text_verified, agreement,
        jyutping, jyutping_cs,
        snr_db, dnsmos, english_ratio,
        tier,
        pause_plan_raw, n_punct, n_no_pause, n_short, n_long,
    ) = row

    text = best_text or ""
    entry = {
        "id": seg_id,
        "audio_path": audio_path,
        "source": source or "",
        "source_url": source_url or "",
        "program": program or "",
        "domain": domain or "other",
        "text": text,
        "text_verified": bool(text_verified),
        "asr_candidates": asr_candidates,
        "asr_agreement": round(float(agreement), 3),
        "jyutping": jyutping,
        "jyutping_cs": jyutping_cs or "",
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

    # P3 additive pause-token fields -- only present when pause_plan has a row for
    # this id (gold/auto_gold scope, see module docstring); omitted otherwise.
    if pause_plan_raw is not None:
        plan = json.loads(pause_plan_raw) if isinstance(pause_plan_raw, str) else pause_plan_raw
        entry["text_pause"] = build_text_pause(text, plan)
        entry["punct_audit"] = {
            "n_punct": n_punct or 0,
            "n_no_pause": n_no_pause or 0,
            "n_short": n_short or 0,
            "n_long": n_long or 0,
        }

    return entry


def discover(
    conn,
    min_agreement: float | None = None,
    min_tier: str | None = None,
    code_switch: str | None = None,
    min_quality_tier: str | None = None,
) -> list[tuple]:
    if code_switch not in CODE_SWITCH_CONDITIONS:
        raise ValueError(
            f"code_switch must be one of {sorted(k for k in CODE_SWITCH_CONDITIONS if k)}, "
            f"got {code_switch!r}"
        )
    tiers = _tiers_at_or_above(min_tier) if min_tier is not None else INCLUDED_TIERS
    tier_list = ", ".join("?" for _ in tiers)
    params = [*tiers, min_agreement, min_agreement]
    quality_tier_cond = ""
    if min_quality_tier is not None:
        quality_tiers = _quality_tiers_at_or_above(min_quality_tier)
        quality_tier_cond = f"AND qt.quality_tier IN ({', '.join('?' for _ in quality_tiers)})"
        params.extend(quality_tiers)
    sql = MANIFEST_DISCOVER_SQL.format(
        tier_list=tier_list,
        code_switch_cond=CODE_SWITCH_CONDITIONS[code_switch],
        quality_tier_cond=quality_tier_cond,
    )
    return conn.execute(sql, params).fetchall()


def build_manifest(
    conn,
    min_agreement: float | None = None,
    min_tier: str | None = None,
    code_switch: str | None = None,
    min_quality_tier: str | None = None,
) -> list[dict]:
    """Runs the full catalog join and returns every eligible manifest entry, sorted by id."""
    rows = discover(conn, min_agreement, min_tier, code_switch, min_quality_tier)
    candidates_by_id = _fetch_asr_candidates_by_id(conn)
    return [build_entry(row, candidates_by_id.get(row[0], [])) for row in rows]


def run_manifest_build(
    *,
    limit: int | None = None,
    min_agreement: float | None = None,
    min_tier: str | None = None,
    code_switch: str | None = None,
    min_quality_tier: str | None = None,
) -> dict:
    """Synchronous, CLI-style -- builds the manifest in-memory and returns it plus a
    summary; does not write files (see run_manifest_export for that). Kept separate from
    export so a caller can inspect/validate the built list before committing it to disk.

    min_agreement: optional asr_agreement.agreement cutoff for a smaller, higher-confidence
    subset (see MANIFEST_DISCOVER_SQL's comment / docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md).
    min_tier: optional tier-column cutoff, e.g. 'auto_gold' for {'gold','auto_gold'} only --
    see TIER_PRECEDENCE / _tiers_at_or_above(). code_switch: optional 'only'/'exclude' cut on
    filters.english_ratio (see CODE_SWITCH_CONDITIONS). min_quality_tier: optional cut on the
    SEPARATE A/B acoustic-cleanliness axis (see QUALITY_TIER_PRECEDENCE / quality_tier.py) --
    'B' for the strict clean subset only, 'A' for everything the quality_tier.assign node
    scored (its gold/auto_gold scope). All four default to None = full pool, unchanged
    behaviour; they can be combined freely (e.g. min_tier='silver' AND code_switch='only' for
    a code-switch-focused eval subset within silver-or-better)."""
    from pipeline.catalog.catalog import connect_ro

    t0 = time.time()
    conn = connect_ro()
    entries = build_manifest(conn, min_agreement, min_tier, code_switch, min_quality_tier)
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
        + (f" [code_switch={code_switch}]" if code_switch is not None else "")
        + (f" [min_quality_tier={min_quality_tier}]" if min_quality_tier is not None else "")
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


def _export_tag(
    min_agreement: float | None,
    min_tier: str | None,
    code_switch: str | None = None,
    min_quality_tier: str | None = None,
) -> str | None:
    """Combines min_agreement/min_tier/code_switch/min_quality_tier into one filename tag,
    e.g. min_tier='auto_gold' -> 'tier_auto_gold'; code_switch='only' -> 'codeswitch_only';
    min_quality_tier='B' -> 'qualityB'; all set -> 'tier_auto_gold_agree090_codeswitch_only_qualityB'.
    None if none are set (the default, unfiltered export keeps the original filenames --
    see run_manifest_export)."""
    parts = []
    if min_tier is not None:
        parts.append(f"tier_{min_tier}")
    if min_agreement is not None:
        parts.append(_agreement_tag(min_agreement))
    if code_switch is not None:
        parts.append(f"codeswitch_{code_switch}")
    if min_quality_tier is not None:
        parts.append(f"quality{min_quality_tier}")
    return "_".join(parts) if parts else None


def run_manifest_export(
    *,
    limit: int | None = None,
    dry_run: bool = False,
    min_agreement: float | None = None,
    min_tier: str | None = None,
    code_switch: str | None = None,
    min_quality_tier: str | None = None,
) -> dict:
    """Builds the manifest then writes metadata/manifest.jsonl + train.jsonl + val.jsonl.
    manifest.jsonl is sorted by id; train/val preserve existing split membership (see
    train_val_split) and are also written sorted by id (see module docstring's
    "Output ordering" section for why this is not a byte-identical reorder of the legacy
    files, only a membership-preserving one).

    min_agreement (added 2026-07-10) / min_tier (added 2026-07-11) / code_switch (added
    2026-07-15, T18) / min_quality_tier (added 2026-07-16, T13): when all are None (default),
    writes the exact same three filenames as always -- CLAUDE.md hard constraint #9
    (zero-risk / external interface unchanged) requires this default call to stay
    byte-compatible with what canto-tts already reads. When any is set, writes to SEPARATE
    files instead (e.g. manifest_tier_auto_gold.jsonl / train_tier_auto_gold.jsonl /
    val_tier_auto_gold.jsonl for min_tier='auto_gold'; manifest_codeswitch_only.jsonl for
    code_switch='only'; manifest_qualityB.jsonl for min_quality_tier='B'; see _export_tag()),
    so a cut export never overwrites the full-pool manifest. Each cut's train/val split
    membership is tracked independently against its own prior train_<tag>.jsonl/
    val_<tag>.jsonl (same preserve-then-extend logic as the default export, just scoped to
    that cut's own files). code_switch='only' is intended for a dedicated QA/eval subset
    (see pipeline/nodes/calibrate.py's CODE_SWITCH_QA_MULTIPLIER for the matching QA
    oversampling), not for training on a permanently-forked monolingual-vs-code-switch
    corpus (CLAUDE.md: code-switching stays in the single default export as a natural HK
    Cantonese feature). min_quality_tier='B' is intended for canto-tts's clean fine-tune
    stage (see pipeline/nodes/quality_tier.py); min_quality_tier is only meaningful combined
    with a min_tier of 'gold' or 'auto_gold' (or omitted), since quality_tiers only covers
    that scope -- combining it with min_tier='silver'/'bronze' silently returns zero rows."""
    from pipeline.config import MANIFEST_PATH, TRAIN_PATH, VAL_PATH, VAL_FRAC

    result = run_manifest_build(
        limit=limit,
        min_agreement=min_agreement,
        min_tier=min_tier,
        code_switch=code_switch,
        min_quality_tier=min_quality_tier,
    )
    entries = result["entries"]

    tag = _export_tag(min_agreement, min_tier, code_switch, min_quality_tier)
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
