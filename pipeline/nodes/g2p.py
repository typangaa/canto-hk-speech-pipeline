"""
pipeline/nodes/g2p.py
g2p DAG node — Cantonese text -> Jyutping romanisation, ported from
scripts/07_g2p.py onto the catalog-driven discovery pattern. Runs entirely
in-supervisor (no worker subprocess, no GPU) — canto-hk-g2p is a Rust core via
PyO3 and is already "the fastest hot path in the pipeline"
(REARCHITECTURE_IMPLEMENTATION_PLAN.md §2.2), so distributing it across CPU
worker subprocesses (the label_prosody.py / filter.acoustic pattern) would add
IPC/JSONL overhead for no throughput gain — the same reasoning asr.agreement
uses for staying in-process (pipeline/nodes/asr.py's run_asr_agreement).

Discovery requires filters.pass = TRUE — i.e. this node only runs G2P on
segments that already passed filter.decide, exactly matching
scripts/07_g2p.py's behaviour of reading from data/filtered/ (only
Stage-6-passing WAVs ever got a .filter.json to read text from). Hard
constraint #8 (CLAUDE.md) says G2P must run on human-verified text, never raw
ASR output — but scripts/07_g2p.py's actual on-disk behaviour reads
data/filtered/*.filter.json's "text" field, which is filter_segment()'s
`transcript.get("text", "")`, i.e. the (uncalibrated) best ASR candidate text,
not a `verified` table row (`text_verified` is tracked but never gates
anything in the legacy script, "soft flag only"). This node preserves that
same actual practice for parity: text comes from asr_agreement.best_text, not
the (currently-empty) `verified` table. Stage-5 human calibration remains a
separate, not-yet-built node; when it lands, this node's text source should
switch to `verified.text` for verified segments per the hard constraint.

Deviation from scripts/07_g2p.py (deliberate, documented): the legacy script
never writes an output file for a rejected segment (valid_fraction < 0.80),
so its `todo` list (built from file-existence) retries rejects forever. The
DAG's anti-join discovery has the same problem unless a row is always written
— this node always upserts a g2p row (jyutping="" + valid_fraction=0.0 for
empty/failed G2P, or the real value even when < 0.80) so discovery stops
resurfacing permanently-bad segments. Downstream consumers (manifest.build,
not yet ported) must filter on valid_fraction >= 0.80 themselves — this node
does not delete/hide low-fraction rows, just stops re-computing them. Same
"always write a row, even on reject/skip" precedent as asr.transcribe's
skipped_ids handling and filter.acoustic's unreadable-audio rows.

`g2p.provenance` note (same shape as `filters.provenance` in pipeline/nodes/filter.py):
every one of the 455,299 legacy-imported segments already has a g2p row (id,
jyutping), so a bare row-existence anti-join would find zero "unconverted" work
forever. This node tags its own writes `provenance = 'g2p_node'` and anti-joins on
that exact value, so legacy rows (`provenance IS NULL`) read as "not yet converted
by this node" — relevant once a legacy segment also gains a `filters.pass = TRUE`
row from filter.decide (see G2P_DISCOVER_SQL below; `f.pass = TRUE` alone already
excludes the legacy `filters` rows too, since those have `pass IS NULL`, not TRUE).

Second deviation (library upgrade, not this node's choice): scripts/07_g2p.py
reached into the private `_canto_hk_g2p.PyPipeline` and hand-resolved a data
directory three parent levels up from `canto_hk_g2p.__file__` — a path that
only resolves under the editable/source dev install this script was written
against (~/Documents/canto-g2p). canto-hk-g2p is deliberately excluded from
uv.lock (pyproject.toml: "install from source") and has since shipped a
public `canto_hk_g2p.Pipeline` wrapper (now on PyPI, v1.9.0 as of 2026-07-19
— see below — vs. whatever version originally produced manifest.jsonl's
frozen `jyutping` field) that resolves its own bundled data/ correctly
regardless of install layout — used here instead of the private API.
Byte-exact parity against the legacy snapshot's `jyutping` field is NOT
expected as a result (the library gained punctuation-normalisation and other
behaviour changes between versions, outside this pipeline's control) — same
"dependency drifted, not a bug" shape as asr.transcribe's golden-parity note
(REARCHITECTURE_IMPLEMENTATION_PLAN.md
§9.1). The correctness gate for this node is per-token validity
(_is_valid_token() below: regex shape + phonological-inventory check, hard
constraint #8), not legacy-byte-match.

Library upgrade to v1.9.0 (2026-07-19): reinstalled from
~/Documents/canto-g2p (editable) after four releases (v1.6.0-v1.9.0) landed
on top of the v1.5.0 this node was originally written against. Two changes
made here as a direct result:
  1. v1.7.0/v1.7.1 corrected ~1,293 common polyphone mis-tie-breaks
     (e.g. 一本正經 zing1->zing3, 沉重 zung6->cung5) at the library-data
     level — a free correctness improvement on reinstall, no code change
     needed. Existing `g2p` rows (`provenance = 'g2p_node'`) are NOT
     automatically revisited by this node's anti-join discovery; a
     corpus-wide reprocess needs a one-time `provenance` reset (out of this
     change's scope — see pending_task.md).
  2. v1.6.0 added `canto_hk_g2p.segment()`, the LSHK Jyutping Inventory
     parser — `_is_valid_token()` now uses it alongside the existing regex
     (see the function's docstring for why the regex alone isn't enough).
     This is strictly stricter than the old regex-only check, so
     `valid_fraction` can only go down for a given text, never up, on
     re-conversion.
Deliberately NOT adopted here: v1.8.0's `Pipeline(user_dict=...)` override
(no curated correction data exists yet — needs sourcing from
`calibrate.sample` QA rejects first) and v1.9.0's `convert_candidates()`
(a calibration-UI feature, not a g2p-node concern) — both tracked as
follow-ups in pending_task.md rather than spec-first-guessed here.

Library upgrade to v2.0.0 (2026-07-19, same day): BREAKING — `convert_detailed()`
gained two new trailing tuple fields (`confidence`, `source`), so the fixed-arity
`for _, jp, lang in tokens` unpack in `_convert_for_moss()` below raised
`ValueError: too many values to unpack` on every call. Fixed with a starred
catch-all (`for _, jp, lang, *_ in tokens`), per the library's own CHANGELOG
migration guide — this node still has no use for `confidence`/`source`
(same "calibration-UI feature, not a g2p-node concern" reasoning as
`convert_candidates()` above), so they're discarded rather than threaded
through. Reinstalled the editable package to pick up the matching version
metadata (`uv pip install -e ~/Documents/canto-g2p` — the compiled extension
and pyproject.toml version had already moved to 2.0.0, but the installed
dist-info was stale at 1.9.0 until reinstall).

Library upgrade to v2.1.0 (2026-07-19, now on PyPI): additive, non-breaking —
new 借音字 (phonetic-loan) alias layer corrects common sound-borrowing
miswritings (e.g. 訓覺 -> fan3 gaau3, not 訓's own fan3 gok3) via a hand-curated
canonical-word alias table resolved at build time; adds `source="variant_alias"`
to the existing tuple shape, no unpack changes needed here. Reinstalled
(`uv pip install -e ~/Documents/canto-g2p`) to refresh the dist-info version
metadata, same drift as the v2.0.0 note above. Same caveat as v1.7.0's
polyphone-mis-tie-break fix: existing `g2p` rows (`provenance = 'g2p_node'`)
are NOT automatically revisited by this node's anti-join discovery — a
corpus-wide reprocess to pick up the ~20 corrected words is a separate,
not-yet-scheduled task (see pending_task.md).

`jyutping_cs` column added (2026-07-20, T30): canto-tts (a downstream consumer
in a sibling repo) independently re-implements "text -> Jyutping" itself
rather than using this node's `jyutping` column, because `jyutping` drops
English/punctuation tokens entirely (see `_convert_for_moss()` above) and
canto-tts needs them kept inline for code-switch alignment. That duplication
meant canto-tts's own `canto-hk-g2p` install could silently drift out of sync
with this node's -- exactly what happened with the v2.1.0 upgrade above (this
node's `provenance`-tagged anti-join reprocessed the 961 affected segments
automatically; canto-tts's already-encoded training data had no such
tracking and needed a manual audit to catch it). `jyutping_cs` closes that
gap: same Cantonese-token conversion as `jyutping`, but via `_G2P.convert()`
instead of `_G2P.convert_detailed()` + `lang == "yue"` filtering, so English
words and punctuation survive verbatim in their original position. Additive
column, existing `jyutping`/`valid_fraction` gate untouched -- `jyutping_cs`
is not validated against `JYUTPING_TOKEN`/`_is_valid_token()` since it's
intentionally not pure Jyutping.
"""

import logging
import re
import time

from canto_hk_g2p import Pipeline as _Pipeline
from canto_hk_g2p import segment as _segment

log = logging.getLogger(__name__)

JYUTPING_TOKEN = re.compile(r"^[a-z]+[1-6]$")
MIN_VALID_FRACTION = 0.80
WARN_VALID_FRACTION = 0.95

# Loaded once at import time — matches scripts/07_g2p.py's module-global singleton.
_G2P = _Pipeline()


def _convert_for_moss(text: str) -> str:
    """Return space-separated Jyutping for Cantonese tokens only (no English/punct)."""
    tokens = _G2P.convert_detailed(text)
    parts = [jp for _, jp, lang, *_ in tokens if lang == "yue"]
    return " ".join(parts)


def text_to_jyutping(text: str) -> str | None:
    """Convert Cantonese text to space-separated Jyutping string.

    English tokens and punctuation are excluded from the output.
    Returns None if no Cantonese tokens were found (or on a G2P failure)."""
    try:
        jyutping = _convert_for_moss(text)
    except Exception as exc:
        log.error(f"canto-g2p failed: {exc}")
        return None
    return jyutping if jyutping else None


def _convert_codeswitch(text: str) -> str:
    """Space-separated Jyutping for Cantonese tokens; English words and
    punctuation are kept verbatim in their original position (no filtering) --
    unlike _convert_for_moss(), nothing is dropped. Uses canto_hk_g2p's
    .convert() directly, same call code-switch-aware downstream consumers
    (e.g. canto-tts's core/cantophon.py) already made independently -- see
    pending_task.md T30."""
    return _G2P.convert(text)


def text_to_jyutping_codeswitch(text: str) -> str:
    """Cantonese -> Jyutping with English/punctuation kept inline (see
    _convert_codeswitch()). Returns "" on empty input or a G2P failure --
    always a printable string, never None, matching
    canto_hk_g2p.Pipeline.convert()'s own empty-input behaviour."""
    text = (text or "").strip()
    if not text:
        return ""
    try:
        return _convert_codeswitch(text)
    except Exception as exc:
        log.error(f"canto-g2p convert (codeswitch) failed: {exc}")
        return ""


def candidate_preview(text: str) -> list[dict]:
    """Per-token polyphone-ambiguity detail for the calibrate UI's live preview
    (pipeline/nodes/calibrate.py's jyutping_preview()) -- reuses the same
    Pipeline singleton as g2p_one()/text_to_jyutping() so this can never drift
    from what the g2p DAG node actually commits (always the rank-0 reading,
    from convert_detailed()). Uses convert_candidates() (canto-hk-g2p v1.9.0+)
    purely to surface the alternates a human reviewer might want to see --
    the g2p node's own conversion path is untouched.

    Returns one {token, candidates, confidence, source} dict per Cantonese
    token that has 2+ known candidate readings; unambiguous tokens (single
    known reading, English, punctuation) are omitted -- nothing for a
    reviewer to look at. confidence is "ranked" (real context-aware lean) or
    "tied" (rime-cantonese arbitrary tie-break, no real signal -- the ones
    most worth a second look)."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        candidates = _G2P.convert_candidates(text)
    except Exception as exc:
        log.error(f"canto-g2p convert_candidates failed: {exc}")
        return []
    return [
        {"token": token, "candidates": readings, "confidence": confidence, "source": source}
        for token, readings, lang, confidence, source in candidates
        if lang == "yue" and len(readings) > 1
    ]


def _is_valid_token(token: str) -> bool:
    """Hard constraint #8's regex shape (`^[a-z]+[1-6]$`) is necessary but not
    sufficient -- it accepts syllable-shaped garbage like "zzz1" that isn't a
    real Cantonese syllable. canto_hk_g2p.segment() (v1.6.0+, LSHK Jyutping
    Inventory) resolves a token into a real onset/rime/tone combination and
    returns None if it doesn't parse -- use both checks."""
    return bool(JYUTPING_TOKEN.match(token)) and _segment(token) is not None


def validate_jyutping(jyutping: str) -> tuple[bool, float, list[str]]:
    """Returns (accept, valid_fraction, bad_tokens). Hard constraint #8's regex,
    tightened with a phonological-inventory check -- see _is_valid_token()."""
    tokens = jyutping.strip().split()
    if not tokens:
        return True, 1.0, []
    valid = [t for t in tokens if _is_valid_token(t)]
    bad = [t for t in tokens if not _is_valid_token(t)]
    frac = len(valid) / len(tokens)
    return frac >= MIN_VALID_FRACTION, round(frac, 3), bad


def g2p_one(text: str | None) -> dict:
    """Pure logic: text -> {jyutping, valid_fraction, jyutping_cs}. Empty text
    or a G2P failure yields jyutping="" + valid_fraction=0.0 (treated as
    reject by any downstream valid_fraction >= 0.80 filter, but still a
    written row). jyutping_cs (English/punctuation kept inline, see
    text_to_jyutping_codeswitch()) is computed independently and is NOT part
    of the accept/reject gate -- it can be non-empty even when
    jyutping/valid_fraction reject, e.g. pure-English text."""
    text = (text or "").strip()
    if not text:
        return {"jyutping": "", "valid_fraction": 0.0, "jyutping_cs": ""}

    jyutping_cs = text_to_jyutping_codeswitch(text)

    jyutping = text_to_jyutping(text)
    if not jyutping:
        return {"jyutping": "", "valid_fraction": 0.0, "jyutping_cs": jyutping_cs}

    accept, frac, bad = validate_jyutping(jyutping)
    if not accept:
        log.info(f"  low Jyutping validity {frac:.2f} {bad[:5]}")
    elif frac < WARN_VALID_FRACTION:
        log.info(f"  Jyutping validity warning {frac:.2f} {bad[:3]}")

    return {"jyutping": jyutping, "valid_fraction": frac, "jyutping_cs": jyutping_cs}


# ---------------------------------------------------------------------------
# Catalog discovery (supervisor side)
# ---------------------------------------------------------------------------

G2P_DISCOVER_SQL = """
    SELECT f.id, a.best_text
    FROM filters f
    JOIN asr_agreement a ON f.id = a.id
    LEFT JOIN g2p g ON f.id = g.id AND g.provenance = 'g2p_node'
    WHERE f.pass = TRUE AND g.id IS NULL
"""


def discover(conn) -> list[tuple]:
    return conn.execute(G2P_DISCOVER_SQL).fetchall()


# ---------------------------------------------------------------------------
# Supervisor: g2p — CPU-only, in-process (no worker subprocess — see module docstring).
# ---------------------------------------------------------------------------

async def run_g2p(*, conn=None, batch_size: int = 2000, limit: int | None = None) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run g2p` usage."""
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    rows = discover(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"g2p: {len(rows)} segments to convert")
    if not rows:
        return {"processed": 0, "errors": 0}

    run_id = new_run_id("g2p")
    processed = 0
    accepted = 0
    t0 = time.time()

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        out_rows = [{"id": seg_id, **g2p_one(text), "provenance": "g2p_node"} for seg_id, text in batch]
        upsert_rows(conn, "g2p", out_rows, ["id"])
        record_batch(conn, run_id, "g2p", [r["id"] for r in out_rows], "ok")
        processed += len(out_rows)
        accepted += sum(1 for r in out_rows if r["valid_fraction"] >= MIN_VALID_FRACTION)
        rate = processed / (time.time() - t0) if time.time() > t0 else 0.0
        log.info(f"{processed}/{len(rows)} converted ({rate:.1f}/s), "
                 f"accept_rate={accepted / processed:.3f}")

    elapsed = time.time() - t0
    log.info(f"DONE: {processed} converted ({accepted} accepted, {accepted / max(processed, 1):.1%}) "
             f"in {elapsed:.0f}s ({processed / elapsed if elapsed > 0 else 0:.1f}/s), run_id={run_id}")
    return {"processed": processed, "accepted": accepted, "errors": 0, "run_id": run_id}
