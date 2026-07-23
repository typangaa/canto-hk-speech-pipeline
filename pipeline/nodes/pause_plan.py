"""
pipeline/nodes/pause_plan.py
pause.plan DAG node — P2 of docs/PAUSE_TOKEN_PUNCTUATION_PLAN.md: for every gold/
auto_gold segment with char-level timestamps (`alignments.chars`, P0), walk
`asr_agreement.best_text` character-by-character against those timestamps and, for
every mid-sentence punctuation mark encountered, compute the acoustic pause (if any)
that flanks it and bucket it into a `no_pause`/`short`/`long` verdict.

This re-implements — as a permanent, idempotent DAG node — EXACTLY the character-walk
algorithm already validated corpus-wide in `docs/PAUSE_CALIBRATION_REPORT.md` §0 (method
recap) and §1-§6 (results). It does not invent a new algorithm; it is not expected to
re-derive different numbers than that report if re-run in aggregate. See that doc for
full context, and `docs/PAUSE_TOKEN_PUNCTUATION_PLAN.md` §2 "P2" for the design intent
this node fulfils.

Algorithm (subsequence-matching pointer, matches the report's §0 method recap verbatim):
walk `best_text` left to right with a pointer `ptr` into `alignments.chars` (one entry
per non-punctuation character, in original text order — punctuation/quotes/spaces were
already stripped by the aligner, see align.py). A text character equal to
`chars[ptr][0]` is a "kept" character and advances `ptr`; a non-matching character does
not advance `ptr` and is treated as dropped by the aligner. Whenever a *mid-sentence*
punctuation mark (，。？！、；： — half-width ,?!;: normalised to full-width; half-width
`.` deliberately excluded, ambiguous with decimal points) is encountered at the current
`ptr` position, three outcomes are possible, matching the report's §0 table exactly:

  - `ptr == 0`             -> `kind="leading_tail"`  (before the very first kept char;
                              negligible, n=1 corpus-wide per the report)
  - `0 < ptr < len(chars)` -> `kind="normal"`         delta_t = chars[ptr].start -
                              chars[ptr-1].end  (genuine flanked acoustic gap)
  - `ptr >= len(chars)`    -> `kind="trailing_tail"`  delta_t measured against
                              `segments.duration_sec` instead (no following kept char --
                              vad_cut has already trimmed the true post-utterance
                              silence, see PAUSE_CALIBRATION_REPORT.md §5); informational
                              only, no verdict assigned regardless of Δt (owner_decisions.4
                              in pause_calibration.json)

A `verdict` (`no_pause`/`short`/`long`) is assigned only for `kind="normal"`, per the
FROZEN `pause_calibration.json` bucket_rule (loaded once at import time, mirroring
g2p.py's `_G2P = _Pipeline()` module-level singleton — do not reload per row).
`kind="trailing_tail"`/`"leading_tail"` get no verdict — this is an explicit owner
decision (pause_calibration.json's `owner_decisions.4`), not an oversight: don't invent
one.

**Segment-level exclusion (the 3.2% char-walk desync, PAUSE_CALIBRATION_REPORT.md §6)**:
if, after walking the entirety of `best_text`, `ptr` has not reached `len(chars)`, the
text and the alignment fell out of sync for this segment (e.g. a genuine ASR/alignment
drift, not just punctuation, broke the subsequence match). This node writes a row with
`plan=[]`, `unalignable=TRUE` for those segments — same "always write a row, even on
reject" precedent as `g2p.py`'s `g2p_one()` and `asr.py`'s skipped_ids handling, so
idempotent anti-join discovery stops resurfacing a permanently-bad segment rather than
retrying it forever.

**Calibration freeze / re-run discipline**: `pause_calibration.json` is FROZEN (see that
file's own `_status` field) — its bucket thresholds must never silently drift. This
node's discovery anti-joins on `provenance = 'pause_plan'` (row existence under that
tag), NOT on `calibration_version`, so an already-computed row is never automatically
revisited just because the calibration file's `version` field changed. This mirrors
g2p.py's documented behaviour when canto-hk-g2p itself gains a correctness fix (see that
module's "Library upgrade" notes): if `pause_calibration.json` is ever legitimately
recalibrated (a NEW owner decision, bumping `version` — the current file's values must
not change in place), a corpus-wide reprocess requires a manual one-time provenance
reset, e.g.:
    UPDATE pause_plan SET provenance = 'pause_plan_stale_<old_version>'
    WHERE calibration_version = '<old_version>';
so this node's discovery SQL (`p.id IS NULL` on `provenance = 'pause_plan'`) picks the
rows back up and recomputes them against the new file. This is a manual fallback, not
automatic version-bump detection — same acceptable-fallback shape as g2p.py's T29
borrowed-character reprocess note.

Node shape: pure CPU, in-process — no GPU, no worker subprocess. Same reasoning and same
shape as `pipeline/nodes/g2p.py` (batched loop calling `upsert_rows`/`record_batch`, no
GPUWorkerBase/spawn_worker machinery) since this is pure Python string/JSON processing
over an already-computed alignment table, not a model inference step.
"""

import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PAUSE_CALIBRATION_PATH = REPO_ROOT / "metadata" / "labels" / "pause_calibration.json"

# The 7 mid-sentence marks the frozen calibration applies to (full-width canonical
# form) -- half-width ,?!;: are normalised to these; half-width `.` is deliberately
# excluded (ambiguous with decimal points, PAUSE_CALIBRATION_REPORT.md §6).
MID_SENTENCE_MARKS = {"，", "。", "、", "；", "：", "？", "！"}
_HALFWIDTH_TO_FULLWIDTH = {",": "，", "?": "？", "!": "！", ";": "；", ":": "："}


def _normalize_mark(ch: str) -> str:
    """Half-width ,?!;: -> full-width; every other character passes through
    unchanged (including half-width `.`, deliberately not normalised -- see
    module docstring)."""
    return _HALFWIDTH_TO_FULLWIDTH.get(ch, ch)


def _is_mid_sentence_mark(ch: str) -> str | None:
    """Returns the normalised mark if *ch* is one of the 7 mid-sentence marks
    (after half/full-width normalisation), else None."""
    norm = _normalize_mark(ch)
    return norm if norm in MID_SENTENCE_MARKS else None


# ---------------------------------------------------------------------------
# Calibration — loaded once at import time (mirrors g2p.py's `_G2P = _Pipeline()`).
# ---------------------------------------------------------------------------


def _load_calibration(path: Path = PAUSE_CALIBRATION_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"pause_calibration.json not found at {path} -- the P1 human gate "
            "(docs/PAUSE_TOKEN_PUNCTUATION_PLAN.md) must complete and freeze this file "
            "before pause.plan can run."
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


_CALIBRATION = _load_calibration()
_NO_PAUSE_CUTOFF: float = _CALIBRATION["owner_decisions"]["2_no_pause_cutoff_sec"]
_LONG_CUTOFF: float = _CALIBRATION["owner_decisions"]["3_long_cutoff_sec"]
CALIBRATION_VERSION: str = _CALIBRATION["version"]


def _verdict(delta_t: float) -> str:
    """Apply the frozen bucket_rule (pause_calibration.json) -- uniform across all
    7 marks (owner_decisions.1), `kind="normal"` only. Never called for
    trailing_tail/leading_tail (see module docstring)."""
    if delta_t < _NO_PAUSE_CUTOFF:
        return "no_pause"
    if delta_t < _LONG_CUTOFF:
        return "short"
    return "long"


# ---------------------------------------------------------------------------
# Pure logic: character walk -> plan
# ---------------------------------------------------------------------------


def compute_pause_plan(
    best_text: str | None,
    chars: list | None,
    duration_sec: float | None,
    *,
    include_timestamps: bool = False,
) -> tuple[list[dict], bool]:
    """Walk *best_text* against *chars* (already `json.loads`-parsed
    `[[char, start_sec, end_sec], ...]`) and return (plan, unalignable).

    plan: list of {"offset", "mark", "kind", "delta_t", "verdict"} dicts, one per
    mid-sentence punctuation mark found, in original `best_text` order. `delta_t`/
    `verdict` are omitted (not just null) for kinds that don't get them, matching
    LABEL_FRAMEWORK_SPEC's "unreliable -> omit, don't write null" convention.

    unalignable: True if the character-walk did not fully consume `chars` by the
    end of `best_text` (text/alignment fell out of sync, ~3.2% corpus-wide per
    PAUSE_CALIBRATION_REPORT.md §6) -- caller must then write plan=[] regardless of
    whatever was collected during the (untrustworthy) walk.

    include_timestamps (P4, docs/PAUSE_TOKEN_PUNCTUATION_PLAN.md, added 2026-07-22):
    when True, each entry also gets `t_start`/`t_end` (the same flanking char
    boundaries `delta_t` is derived from -- `chars[ptr-1][2]`/`chars[ptr][1]` for
    "normal", segment-edge equivalents for the tail kinds) so a caller can seek/draw
    audio-timeline markers. Purely additive and opt-in: `pause_plan.py`'s own
    corpus-wide node (pause_plan_one -> run_pause_plan) never passes this, so the
    stored `pause_plan.plan` shape used by manifest.py/label_store.py (P3) is
    unaffected -- only `pipeline/nodes/calibrate.py`'s live pause-preview path (P4)
    uses it, computed fresh per request rather than backfilled into the catalog.
    """
    best_text = best_text or ""
    chars = chars or []
    n_chars = len(chars)
    ptr = 0
    plan: list[dict] = []

    for offset, ch in enumerate(best_text):
        if ptr < n_chars and ch == chars[ptr][0]:
            ptr += 1
            continue

        mark = _is_mid_sentence_mark(ch)
        if mark is None:
            continue  # dropped: punctuation/quote/space we don't care about

        if ptr == 0:
            entry = {"offset": offset, "mark": mark, "kind": "leading_tail"}
            if include_timestamps:
                entry["t_start"] = 0.0
                entry["t_end"] = chars[0][1] if n_chars else None
            plan.append(entry)
        elif ptr >= n_chars:
            last_end = chars[n_chars - 1][2]
            delta_t = (duration_sec - last_end) if duration_sec is not None else None
            entry = {"offset": offset, "mark": mark, "kind": "trailing_tail"}
            if delta_t is not None:
                entry["delta_t"] = round(delta_t, 4)
            if include_timestamps:
                entry["t_start"] = last_end
                entry["t_end"] = duration_sec
            plan.append(entry)
        else:
            prev_end = chars[ptr - 1][2]
            next_start = chars[ptr][1]
            delta_t = round(next_start - prev_end, 4)
            entry = {
                "offset": offset,
                "mark": mark,
                "kind": "normal",
                "delta_t": delta_t,
                "verdict": _verdict(delta_t),
            }
            if include_timestamps:
                entry["t_start"] = prev_end
                entry["t_end"] = next_start
            plan.append(entry)

    unalignable = ptr != n_chars
    if unalignable:
        return [], True
    return plan, False


def _summarize(plan: list[dict]) -> dict:
    n_no_pause = sum(1 for e in plan if e.get("verdict") == "no_pause")
    n_short = sum(1 for e in plan if e.get("verdict") == "short")
    n_long = sum(1 for e in plan if e.get("verdict") == "long")
    return {
        "n_punct": len(plan),
        "n_no_pause": n_no_pause,
        "n_short": n_short,
        "n_long": n_long,
    }


def pause_plan_one(best_text: str | None, chars_raw, duration_sec: float | None) -> dict:
    """Row-level entrypoint: raw catalog values -> the full pause_plan row dict
    (minus id/provenance, which the caller attaches). `chars_raw` is the raw
    `alignments.chars` value as fetched from DuckDB (a JSON string) -- parsed here."""
    try:
        chars = json.loads(chars_raw) if chars_raw is not None else []
    except (json.JSONDecodeError, TypeError) as exc:
        log.warning("pause_plan: could not parse alignments.chars JSON (%s) -- treating as unalignable", exc)
        chars = None

    if chars is None:
        plan, unalignable = [], True
    else:
        plan, unalignable = compute_pause_plan(best_text, chars, duration_sec)

    row = {"plan": plan, "unalignable": unalignable, "calibration_version": CALIBRATION_VERSION}
    row.update(_summarize(plan))
    return row


# ---------------------------------------------------------------------------
# Catalog discovery (supervisor side)
# ---------------------------------------------------------------------------

PAUSE_PLAN_DISCOVER_SQL = """
    SELECT a.id, a.best_text, al.chars, s.duration_sec
    FROM asr_agreement a
    JOIN tiers t ON a.id = t.id AND t.tier IN ('gold', 'auto_gold')
    JOIN alignments al ON a.id = al.id AND al.provenance = 'qwen3_aligner' AND al.chars IS NOT NULL
    JOIN segments s ON a.id = s.id
    LEFT JOIN pause_plan p ON a.id = p.id AND p.provenance = 'pause_plan'
    WHERE a.best_text IS NOT NULL AND a.best_text <> '' AND p.id IS NULL
"""


def discover(conn) -> list[tuple]:
    return conn.execute(PAUSE_PLAN_DISCOVER_SQL).fetchall()


# ---------------------------------------------------------------------------
# Supervisor: pause.plan -- CPU-only, in-process (no worker subprocess, see
# module docstring).
# ---------------------------------------------------------------------------


async def run_pause_plan(*, conn=None, batch_size: int = 5000, limit: int | None = None) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see g2p.py's run_g2p
    docstring for the rationale). Defaults to a fresh self-managed connect()
    for standalone `pipe run pause.plan` usage."""
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    rows = discover(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"pause.plan: {len(rows)} segments to plan")
    if not rows:
        return {"processed": 0, "errors": 0}

    run_id = new_run_id("pause.plan")
    processed = 0
    unalignable_count = 0
    t0 = time.time()

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        out_rows = []
        for seg_id, best_text, chars_raw, duration_sec in batch:
            row = pause_plan_one(best_text, chars_raw, duration_sec)
            row["id"] = seg_id
            row["provenance"] = "pause_plan"
            out_rows.append(row)
        upsert_rows(conn, "pause_plan", out_rows, ["id"])
        record_batch(conn, run_id, "pause.plan", [r["id"] for r in out_rows], "ok")
        processed += len(out_rows)
        unalignable_count += sum(1 for r in out_rows if r["unalignable"])
        rate = processed / (time.time() - t0) if time.time() > t0 else 0.0
        log.info(f"{processed}/{len(rows)} planned ({rate:.1f}/s), "
                 f"unalignable_rate={unalignable_count / processed:.3f}")

    elapsed = time.time() - t0
    log.info(f"DONE: {processed} planned ({unalignable_count} unalignable, "
              f"{unalignable_count / max(processed, 1):.1%}) in {elapsed:.0f}s "
              f"({processed / elapsed if elapsed > 0 else 0:.1f}/s), run_id={run_id}")
    return {
        "processed": processed,
        "unalignable": unalignable_count,
        "errors": 0,
        "run_id": run_id,
    }
