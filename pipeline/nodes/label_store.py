"""
pipeline/nodes/label_store.py

Label store node -- pass 2 of the two-pass label-store build.

Joins segments against all four populated label tables (lang / overlap /
music / prosody), applies bucket() rules using calibration constants written
by pass 1 (``pipeline.nodes.label_calibrate``), and writes one JSON line per
segment to ``metadata/labels.jsonl`` (via ``pipeline.config.LABELS_STORE_PATH``).

Two-pass design rationale (docs/LABEL_FRAMEWORK_SPEC.md §§ 7–9):
  Pass 1 (label_calibrate.py) computes the corpus-wide constants needed to
  turn raw continuous values into categorical buckets.  Pass 2 (this file)
  is a pure read-join-write with no statistics computation of its own.

Per-line output shape (spec § 7):
  {
    "id": "<segment_id>",
    "quality": {
      "lang": "<iso_code>",   # only if labels_lang row present
      "cmn_prob": <float>,    # only if labels_lang row present
      "overlap_ratio": <float>, # only if labels_overlap row present
      "music_prob": <float>   # only if labels_music row present
    },
    "control": {
      "rate":  {"raw": <float>, "bucket": "<slow|normal|fast>"},
      "pitch": {"raw_hz": <float>, "z": <float>, "bucket": "<low|normal|high>"},
      "pause": {"gaps": [[start, dur], ...], "total_sec": <float>}
    },
    "calibration_version": "<version string>",
    "provenance": {"rate": "jyutping+silero", "pitch": "parselmouth"}
  }

Keys are omitted entirely (not set to null) when the underlying detector has
not produced a result for this segment (spec § 7: "unreliable attributes --
omit the whole key, never write null"). Segments with no usable quality
labels at all (no lang/overlap/music row) are skipped and counted.

NOTE: emotion and energy labels are deliberately excluded from this pass.
  • emotion is gated behind an owner listening spot-check that has not yet
    occurred (see docs/LABEL_FRAMEWORK_SPEC.md § 6).
  • energy is a schema-only slot; the underlying detector has never been run.
"""

import json
import logging
import time
from pathlib import Path
from typing import Iterator

from pipeline.catalog.catalog import connect_ro
from pipeline.config import LABELS_CALIBRATION_PATH, LABELS_STORE_PATH

log = logging.getLogger(__name__)

# ── sigma floor (must match label_calibrate.py) ──────────────────────────────
_SIGMA_EPSILON: float = 1.0  # Hz
"""Defensive sigma floor for F0 z-score division.

label_calibrate.py already clamps sigma before writing calibration.json, but
we clamp again here in case this function is called with a hand-crafted
calibration dict or in unit tests."""


# ── pure bucket functions ─────────────────────────────────────────────────────


def bucket_rate(raw: float, p25: float, p75: float) -> str:
    """Classify a speaking-rate value into a three-way bucket.

    Parameters
    ----------
    raw:
        rate_raw value (jyutping syllables per voiced second).
    p25, p75:
        Corpus-wide 25th and 75th percentile thresholds from calibration.json.

    Returns
    -------
    ``"slow"`` if raw < p25, ``"fast"`` if raw > p75, otherwise ``"normal"``.
    """
    if raw < p25:
        return "slow"
    if raw > p75:
        return "fast"
    return "normal"


def bucket_pitch(
    raw_hz: float,
    mu: float,
    sigma: float,
) -> tuple[float, str]:
    """Compute F0 z-score and classify into a three-way bucket.

    The +/-0.5 sigma rule is taken from docs/LABEL_FRAMEWORK_SPEC.md § 3.

    Parameters
    ----------
    raw_hz:
        Per-segment median F0 in Hz (parselmouth).
    mu:
        Speaker-specific (or corpus fallback) F0 mean from calibration.json.
    sigma:
        Speaker-specific (or corpus fallback) F0 standard deviation.
        Clamped defensively to ``_SIGMA_EPSILON`` here even if calibration
        already clamped it, to protect callers using hand-crafted dicts.

    Returns
    -------
    (z_score, bucket) where z_score is rounded to 3 d.p. and bucket is one
    of ``"low"``, ``"normal"``, ``"high"``.
    """
    sigma_safe = max(sigma, _SIGMA_EPSILON)
    z = (raw_hz - mu) / sigma_safe
    z_rounded = round(z, 3)
    if z < -0.5:
        bucket = "low"
    elif z > 0.5:
        bucket = "high"
    else:
        bucket = "normal"
    return z_rounded, bucket


# ── calibration loader ────────────────────────────────────────────────────────


def load_calibration(path: Path) -> dict:
    """Load and return the calibration JSON written by label_calibrate.py.

    Parameters
    ----------
    path:
        Path to calibration.json (typically ``LABELS_CALIBRATION_PATH``).

    Raises
    ------
    FileNotFoundError
        If the calibration file does not exist.  The error message explicitly
        tells the caller to run ``pipe run label.calibrate`` first.
    json.JSONDecodeError
        If the file exists but is not valid JSON (corrupt / partial write).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {path}\n"
            "Run 'pipe run label.calibrate' (or label_calibrate.run_label_calibrate()) "
            "before building the label store.  The calibration pass must complete "
            "successfully at least once to produce this file."
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ── main SQL query ─────────────────────────────────────────────────────────────

# One query joining all relevant tables.  Everything is LEFT JOINed off
# segments so that partially-labelled segments are included; NULLs are handled
# in Python row-building logic.  filters is joined only for english_ratio,
# which gates rate-bucket eligibility.
_STORE_SQL = """
SELECT
    s.id,
    s.speaker_id,
    -- lang labels
    ll.lang,
    ll.cmn_prob,
    -- overlap labels
    lo.overlap_ratio,
    -- music labels
    lm.music_prob,
    -- prosody labels
    lp.rate_raw,
    lp.f0_median_hz,
    lp.gaps,
    -- filter column for rate eligibility
    f.english_ratio
FROM segments AS s
LEFT JOIN labels_lang    AS ll ON ll.id = s.id
LEFT JOIN labels_overlap AS lo ON lo.id = s.id
LEFT JOIN labels_music   AS lm ON lm.id = s.id
LEFT JOIN labels_prosody AS lp ON lp.id = s.id
LEFT JOIN filters        AS f  ON f.id  = s.id
ORDER BY s.id
"""

# Named column positions (must match SELECT order above)
_COL_ID            = 0
_COL_SPEAKER_ID    = 1
_COL_LANG          = 2
_COL_CMN_PROB      = 3
_COL_OVERLAP_RATIO = 4
_COL_MUSIC_PROB    = 5
_COL_RATE_RAW      = 6
_COL_F0_HZ         = 7
_COL_GAPS          = 8
_COL_ENGLISH_RATIO = 9


# Sentinel object used internally by build_label_rows to signal a skipped
# segment without breaking the generator protocol. run_label_store() checks
# identity with ``is _SKIP_SENTINEL`` rather than inspecting dict contents.
_SKIP_SENTINEL = object()


# ── row builder ───────────────────────────────────────────────────────────────


def build_label_rows(conn, calibration: dict) -> Iterator[dict]:
    """Generate one label dict per segment, ready for JSONL serialisation.

    Yields rows that have at least one usable quality label (lang/overlap/music).
    Segments with no quality label at all yield the internal ``_SKIP_SENTINEL``
    object instead of a dict; callers must filter these out (see
    :func:`run_label_store`).

    Parameters
    ----------
    conn:
        DuckDB read-only connection.
    calibration:
        Dict loaded from calibration.json by :func:`load_calibration`.

    Yields
    ------
    dict matching the per-line schema documented in the module docstring, or
    ``_SKIP_SENTINEL`` for segments with no usable quality label.
    """
    # Unpack calibration constants once (not per-row).
    rate_p25: float = calibration["rate"]["p25"]
    rate_p75: float = calibration["rate"]["p75"]
    pitch_cfg: dict = calibration["pitch"]
    per_speaker: dict = pitch_cfg["per_speaker"]
    corpus_fallback: dict = pitch_cfg["corpus_fallback"]
    calib_version: str = calibration["version"]

    log.info("label_store: executing main join query...")
    rows = conn.execute(_STORE_SQL).fetchall()
    log.info("label_store: %d segment rows fetched from catalog", len(rows))

    for row in rows:
        seg_id: str          = row[_COL_ID]
        speaker_id: str | None = row[_COL_SPEAKER_ID]
        lang: str | None       = row[_COL_LANG]
        cmn_prob: float | None = row[_COL_CMN_PROB]
        overlap_ratio: float | None = row[_COL_OVERLAP_RATIO]
        music_prob: float | None    = row[_COL_MUSIC_PROB]
        rate_raw: float | None      = row[_COL_RATE_RAW]
        f0_hz: float | None         = row[_COL_F0_HZ]
        gaps_raw                    = row[_COL_GAPS]   # JSON string or None
        english_ratio: float | None = row[_COL_ENGLISH_RATIO]

        # ── quality dict ───────────────────────────────────────────────────
        # Omit whole keys when the underlying detector has no result.
        # If NONE of the three quality labels exist, skip this segment entirely.
        quality: dict = {}
        if lang is not None and cmn_prob is not None:
            # labels_lang.lang is already the detector argmax (e.g. "yue"/"cmn");
            # pass it through directly. cmn_prob is carried as a raw quality
            # number alongside it (spec § 7).
            quality["lang"] = lang
            quality["cmn_prob"] = round(cmn_prob, 4)
        if overlap_ratio is not None:
            quality["overlap_ratio"] = round(overlap_ratio, 4)
        if music_prob is not None:
            quality["music_prob"] = round(music_prob, 4)

        if not quality:
            # No usable quality information -- caller counts these as skipped.
            yield _SKIP_SENTINEL
            continue

        # ── control dict ───────────────────────────────────────────────────
        control: dict = {}
        provenance: dict = {}

        # rate: omit if rate_raw is NULL or english_ratio > 0.5
        rate_eligible = (
            rate_raw is not None
            and (english_ratio is None or english_ratio <= 0.5)
        )
        if rate_eligible:
            bucket_r = bucket_rate(rate_raw, rate_p25, rate_p75)
            control["rate"] = {"raw": round(rate_raw, 4), "bucket": bucket_r}
            provenance["rate"] = "jyutping+silero"

        # pitch: omit if f0_median_hz is NULL
        if f0_hz is not None:
            # Use per-speaker baseline when available; fall back to corpus.
            if speaker_id is not None and speaker_id in per_speaker:
                spk_stats = per_speaker[speaker_id]
            else:
                spk_stats = corpus_fallback
            z, bucket_p = bucket_pitch(f0_hz, spk_stats["mu"], spk_stats["sigma"])
            control["pitch"] = {
                "raw_hz": round(f0_hz, 3),
                "z": z,
                "bucket": bucket_p,
            }
            provenance["pitch"] = "parselmouth"

        # pause: include if gaps column is not NULL (empty list is informative)
        if gaps_raw is not None:
            try:
                gaps: list = json.loads(gaps_raw)
            except (json.JSONDecodeError, TypeError) as exc:
                log.warning(
                    "label_store: could not parse gaps JSON for id=%s (%s) -- skipping pause key",
                    seg_id,
                    exc,
                )
                gaps = None  # treat as missing so we omit the key cleanly

            if gaps is not None:
                total_sec = round(sum(dur for _, dur in gaps), 3) if gaps else 0.0
                control["pause"] = {"gaps": gaps, "total_sec": total_sec}
                # pause provenance is not listed in spec § 7 (silero implied via
                # prosody provenance); do not add a "pause" key to provenance.

        # ── assemble record ────────────────────────────────────────────────
        record: dict = {"id": seg_id}
        if quality:
            record["quality"] = quality
        if control:
            record["control"] = control
        record["calibration_version"] = calib_version
        if provenance:
            record["provenance"] = provenance

        yield record


# ── entrypoint ────────────────────────────────────────────────────────────────


def run_label_store() -> dict:
    """Run the full label store build and write labels.jsonl.

    Opens a read-only catalog connection, loads calibration.json (must already
    exist -- see :func:`load_calibration`), iterates :func:`build_label_rows`
    writing one ``json.dumps`` line per segment, counts written vs skipped rows,
    and returns a summary dict.

    This is a synchronous, CLI-style entrypoint -- no async, no orchestrator
    involvement. Invoke directly or via ``pipe run label.store``.

    Returns
    -------
    dict with keys:
      ``path``               -- absolute path of the written JSONL file
      ``written``            -- number of segment lines written
      ``skipped_no_quality`` -- segments with no lang/overlap/music label (omitted)
      ``with_rate``          -- segments that have a rate bucket
      ``with_pitch``         -- segments that have a pitch bucket
      ``with_pause``         -- segments that have pause data
    """
    log.info("label_store: starting label store build")
    t0 = time.monotonic()

    # ── load calibration (fails loudly if pass 1 hasn't run) ──────────────
    calibration = load_calibration(LABELS_CALIBRATION_PATH)
    log.info(
        "label_store: loaded calibration version=%s from %s",
        calibration.get("version", "?"),
        LABELS_CALIBRATION_PATH,
    )

    out_path: Path = LABELS_STORE_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    n_rate = 0
    n_pitch = 0
    n_pause = 0

    with connect_ro() as conn, out_path.open("w", encoding="utf-8") as fh:
        for record in build_label_rows(conn, calibration):
            if record is _SKIP_SENTINEL:
                n_skipped += 1
                if n_skipped % 10_000 == 0:
                    log.info(
                        "label_store: ... %d written, %d skipped (no quality) so far",
                        n_written,
                        n_skipped,
                    )
                continue

            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

            # tally optional keys for summary logging
            ctrl = record.get("control", {})
            if "rate" in ctrl:
                n_rate += 1
            if "pitch" in ctrl:
                n_pitch += 1
            if "pause" in ctrl:
                n_pause += 1

            if n_written % 50_000 == 0:
                log.info(
                    "label_store: ... %d written, %d skipped so far", n_written, n_skipped
                )

    elapsed = time.monotonic() - t0
    log.info(
        "label_store: done -- wrote %d rows, skipped %d (no quality)  "
        "|  rate=%d  pitch=%d  pause=%d  |  %.1fs  ->  %s",
        n_written,
        n_skipped,
        n_rate,
        n_pitch,
        n_pause,
        elapsed,
        out_path,
    )

    return {
        "path": str(out_path),
        "written": n_written,
        "skipped_no_quality": n_skipped,
        "with_rate": n_rate,
        "with_pitch": n_pitch,
        "with_pause": n_pause,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run_label_store()
    print(json.dumps(result, indent=2))
