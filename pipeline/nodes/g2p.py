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
public `canto_hk_g2p.Pipeline` wrapper (now on PyPI, v1.5.0 vs. whatever
version originally produced manifest.jsonl's frozen `jyutping` field) that
resolves its own bundled data/ correctly regardless of install layout — used
here instead of the private API. Byte-exact parity against the legacy
snapshot's `jyutping` field is NOT expected as a result (the library gained
punctuation-normalisation and other behaviour changes between versions,
outside this pipeline's control) — same "dependency drifted, not a bug" shape
as asr.transcribe's golden-parity note (REARCHITECTURE_IMPLEMENTATION_PLAN.md
§9.1). The correctness gate for this node is per-token structural validity
(`^[a-z]+[1-6]$`, hard constraint #8), not legacy-byte-match.
"""

import logging
import re
import time

from canto_hk_g2p import Pipeline as _Pipeline

log = logging.getLogger(__name__)

JYUTPING_TOKEN = re.compile(r"^[a-z]+[1-6]$")
MIN_VALID_FRACTION = 0.80
WARN_VALID_FRACTION = 0.95

# Loaded once at import time — matches scripts/07_g2p.py's module-global singleton.
_G2P = _Pipeline()


def _convert_for_moss(text: str) -> str:
    """Return space-separated Jyutping for Cantonese tokens only (no English/punct)."""
    tokens = _G2P.convert_detailed(text)
    parts = [jp for _, jp, lang in tokens if lang == "yue"]
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


def validate_jyutping(jyutping: str) -> tuple[bool, float, list[str]]:
    """Returns (accept, valid_fraction, bad_tokens). Hard constraint #8's regex."""
    tokens = jyutping.strip().split()
    if not tokens:
        return True, 1.0, []
    valid = [t for t in tokens if JYUTPING_TOKEN.match(t)]
    bad = [t for t in tokens if not JYUTPING_TOKEN.match(t)]
    frac = len(valid) / len(tokens)
    return frac >= MIN_VALID_FRACTION, round(frac, 3), bad


def g2p_one(text: str | None) -> dict:
    """Pure logic: text -> {jyutping, valid_fraction}. Empty text or a G2P
    failure yields jyutping="" + valid_fraction=0.0 (treated as reject by any
    downstream valid_fraction >= 0.80 filter, but still a written row)."""
    text = (text or "").strip()
    if not text:
        return {"jyutping": "", "valid_fraction": 0.0}

    jyutping = text_to_jyutping(text)
    if not jyutping:
        return {"jyutping": "", "valid_fraction": 0.0}

    accept, frac, bad = validate_jyutping(jyutping)
    if not accept:
        log.info(f"  low Jyutping validity {frac:.2f} {bad[:5]}")
    elif frac < WARN_VALID_FRACTION:
        log.info(f"  Jyutping validity warning {frac:.2f} {bad[:3]}")

    return {"jyutping": jyutping, "valid_fraction": frac}


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

async def run_g2p(*, batch_size: int = 2000, limit: int | None = None) -> dict:
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = connect()
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
