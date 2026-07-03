"""
pipeline/nodes/label_calibrate.py

Label calibration node -- pass 1 of the two-pass label-store build.

Scans the raw label tables once to compute corpus-wide calibration constants
(speaking-rate percentiles, per-speaker F0 z-score baselines) and records
already-decided fixed thresholds (lang / overlap / music) for provenance.
The resulting JSON is written to metadata/labels/calibration.json (via
``pipeline.config.LABELS_CALIBRATION_PATH``) and consumed by
``pipeline.nodes.label_store`` in pass 2.

Two-pass design rationale (docs/LABEL_FRAMEWORK_SPEC.md §§ 8–9):
  Pass 1 (this file) -- data-driven constants only:
    • rate  : P25/P75 over corpus speaking-rate values (syllables/voiced-sec),
              excluding English-heavy segments where the syllable counter is
              unreliable (english_ratio > 0.5).
    • pitch : per-speaker mean/σ of F0 (Hz) for z-score normalisation; speakers
              with fewer than MIN_SPEAKER_SAMPLES rows fall back to a corpus-wide
              baseline computed here.
    • fixed thresholds (lang / overlap / music) are recorded here for a single
              source-of-truth -- no computation required.
  Pass 2 (label_store.py) -- join + bucket + write labels.jsonl.

NOTE: emotion and energy labels are deliberately excluded from this pass.
  • emotion is gated behind an owner listening spot-check that has not yet
    occurred (see docs/LABEL_FRAMEWORK_SPEC.md § 6).
  • energy is a schema-only slot; the underlying detector has never been run.
"""

import datetime
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from pipeline.catalog.catalog import connect_ro
from pipeline.config import LABELS_CALIBRATION_PATH

log = logging.getLogger(__name__)

# ── module constant (spec § 8.2) ────────────────────────────────────────────
MIN_SPEAKER_SAMPLES: int = 5
"""Minimum per-speaker F0 observations required to use the per-speaker
baseline; speakers below this threshold fall back to the corpus-wide
fallback (mu, sigma) computed across all speakers."""

# ── sigma floor (spec § 8.2 footnote) ───────────────────────────────────────
_SIGMA_EPSILON: float = 1.0  # Hz
"""Minimum allowed sigma for F0 z-score division.

A speaker with literally identical F0 every utterance (or a single-sample
corpus edge case) would yield sigma = 0 and cause a divide-by-zero downstream.
We clamp to this small epsilon instead; 1 Hz is well below the 3-5 Hz JND
for pitch perception and has no practical effect on bucket assignment."""


# ── pure functions ───────────────────────────────────────────────────────────


def compute_rate_percentiles(values: list[float]) -> dict:
    """Return P25/P75 speaking-rate statistics over *values*.

    Parameters
    ----------
    values:
        List of rate_raw floats (syllables per voiced second).  Must be
        non-empty; caller is responsible for pre-filtering NULL / excluded rows.

    Returns
    -------
    dict with keys ``p25``, ``p75`` (both rounded to 3 d.p.) and
    ``n_samples`` (int).
    """
    if not values:
        raise ValueError(
            "compute_rate_percentiles: received an empty value list; "
            "cannot compute percentiles."
        )
    arr = np.asarray(values, dtype=np.float64)
    p25 = float(round(float(np.percentile(arr, 25)), 3))
    p75 = float(round(float(np.percentile(arr, 75)), 3))
    return {"p25": p25, "p75": p75, "n_samples": len(values)}


def compute_speaker_pitch_stats(
    rows: list[tuple[str, float]],
    min_samples: int = MIN_SPEAKER_SAMPLES,
) -> tuple[dict, dict, dict]:
    """Compute per-speaker and corpus-wide F0 baseline statistics.

    Parameters
    ----------
    rows:
        List of (speaker_id, f0_median_hz) tuples.  Rows where either field
        is NULL must be filtered out before calling this function.
    min_samples:
        Minimum observations per speaker to emit a per-speaker entry;
        speakers below this threshold are counted but not returned in the
        per-speaker dict -- label_store.py will use the corpus fallback for
        them.

    Returns
    -------
    A three-tuple:
      per_speaker : dict[speaker_id -> {"mu": float, "sigma": float, "n": int}]
                    Only speakers with n >= min_samples.
      corpus_fallback : {"mu": float, "sigma": float}
                    Computed over ALL f0_median_hz values regardless of speaker
                    sample count.
      counts : {"calibrated": int, "below_min": int, "corpus_total": int}
    """
    from collections import defaultdict

    buckets: dict[str, list[float]] = defaultdict(list)
    all_values: list[float] = []

    for speaker_id, f0_hz in rows:
        buckets[speaker_id].append(f0_hz)
        all_values.append(f0_hz)

    per_speaker: dict = {}
    n_calibrated = 0
    n_below_min = 0

    for spk, f0_list in buckets.items():
        n = len(f0_list)
        arr = np.asarray(f0_list, dtype=np.float64)
        mu = float(round(float(np.mean(arr)), 3))
        # population stddev (ddof=0) -- consistent with z-score baseline intent
        sigma_raw = float(np.std(arr, ddof=0))
        # clamp sigma to epsilon to guard against zero-variance edge cases
        sigma = float(round(max(sigma_raw, _SIGMA_EPSILON), 3))
        if n >= min_samples:
            per_speaker[spk] = {"mu": mu, "sigma": sigma, "n": n}
            n_calibrated += 1
        else:
            n_below_min += 1

    # corpus-wide fallback (all values, regardless of per-speaker eligibility)
    if all_values:
        ca = np.asarray(all_values, dtype=np.float64)
        c_mu = float(round(float(np.mean(ca)), 3))
        c_sigma_raw = float(np.std(ca, ddof=0))
        c_sigma = float(round(max(c_sigma_raw, _SIGMA_EPSILON), 3))
        corpus_fallback = {"mu": c_mu, "sigma": c_sigma}
    else:
        raise ValueError(
            "compute_speaker_pitch_stats: received an empty rows list; "
            "cannot compute corpus fallback F0 statistics."
        )

    counts = {
        "calibrated": n_calibrated,
        "below_min": n_below_min,
        "corpus_total": len(all_values),
    }
    return per_speaker, corpus_fallback, counts


# ── DB-driving function ──────────────────────────────────────────────────────

# SQL: speaking rate (syllables/voiced-sec), excluding English-heavy segments.
# LEFT JOIN so that rows with no matching filters row (english_ratio IS NULL)
# are kept -- NULL english_ratio is treated as "not excluded" per spec § 8.1.
_RATE_SQL = """
SELECT
    lp.rate_raw
FROM labels_prosody AS lp
LEFT JOIN filters AS f ON f.id = lp.id
WHERE
    lp.rate_raw IS NOT NULL
    AND (f.english_ratio IS NULL OR f.english_ratio <= 0.5)
"""

# SQL: F0 per segment joined to speaker_id for grouping.
_PITCH_SQL = """
SELECT
    s.speaker_id,
    lp.f0_median_hz
FROM labels_prosody AS lp
JOIN segments AS s ON s.id = lp.id
WHERE
    lp.f0_median_hz IS NOT NULL
    AND s.speaker_id IS NOT NULL
"""


def calibrate(conn) -> dict:
    """Run calibration queries and assemble the full calibration dict.

    Parameters
    ----------
    conn:
        A DuckDB read-only connection (from ``pipeline.catalog.catalog.connect_ro``).

    Returns
    -------
    dict ready for ``json.dump``.  See module docstring for the exact schema.
    """
    # ── 1. speaking rate ────────────────────────────────────────────────────
    log.info("label_calibrate: querying speaking-rate values...")
    rate_rows = conn.execute(_RATE_SQL).fetchall()
    rate_values = [row[0] for row in rate_rows]
    log.info(
        "label_calibrate: %d rate_raw values after English-heavy exclusion",
        len(rate_values),
    )
    rate_stats = compute_rate_percentiles(rate_values)

    # ── 2. per-speaker F0 baseline ──────────────────────────────────────────
    log.info("label_calibrate: querying F0 (pitch) values per speaker...")
    pitch_rows = conn.execute(_PITCH_SQL).fetchall()  # list of (speaker_id, f0_hz)
    log.info(
        "label_calibrate: %d (speaker_id, f0_median_hz) pairs fetched",
        len(pitch_rows),
    )
    per_speaker, corpus_fallback, counts = compute_speaker_pitch_stats(
        pitch_rows, min_samples=MIN_SPEAKER_SAMPLES
    )
    log.info(
        "label_calibrate: pitch -- %d speakers calibrated, %d below min (%d), "
        "%d total utterances",
        counts["calibrated"],
        counts["below_min"],
        MIN_SPEAKER_SAMPLES,
        counts["corpus_total"],
    )

    # ── 3. version stamping ─────────────────────────────────────────────────
    today = datetime.date.today().isoformat()
    git_rev = _get_git_rev()
    version = f"{today}-{git_rev}"

    # ── 4. assemble output dict ─────────────────────────────────────────────
    calibration = {
        "version": version,
        "date": today,
        "git_rev": git_rev,
        "n_samples": {
            "rate": rate_stats["n_samples"],
            "pitch_speakers_calibrated": counts["calibrated"],
            "pitch_speakers_below_min": counts["below_min"],
            "pitch_corpus_total": counts["corpus_total"],
        },
        "rate": {
            "p25": rate_stats["p25"],
            "p75": rate_stats["p75"],
        },
        "pitch": {
            "min_speaker_samples": MIN_SPEAKER_SAMPLES,
            "corpus_fallback": corpus_fallback,
            "per_speaker": per_speaker,
        },
        # Fixed thresholds from docs/LABEL_FRAMEWORK_SPEC.md label catalog
        # table -- recorded here for provenance/reproducibility so that
        # label_store.py and any future consumer read from one place.
        "thresholds": {
            "lang": {
                # segments with cmn_prob >= this value are classified "cmn";
                # otherwise the detector's argmax label (labels_lang.lang) wins.
                "cmn_threshold": 0.90,
            },
            "overlap": {
                # overlap_ratio < tier_b_max  => passes strict-tier filter
                # overlap_ratio >= tier_a_max => stage-1-fatal exclusion signal
                "tier_b_max": 0.05,
                "tier_a_max": 0.20,
            },
            "music": {
                # music_prob >= tier_b_max_prob => segment flagged as music-heavy.
                # PROVISIONAL: 0.30 is a conservative placeholder -- easy to
                # retune here without touching any bucketing logic downstream,
                # since label_store.py reads this value from the JSON at runtime.
                "tier_b_max_prob": 0.30,
            },
        },
    }
    return calibration


# ── helper ───────────────────────────────────────────────────────────────────


def _get_git_rev() -> str:
    """Return a short git revision hash, or 'unknown' on any failure.

    Uses ``pipeline.config.REPO_ROOT`` when available; falls back to three
    directories above this file.  Never raises -- a missing git binary or a
    detached/non-git context must not crash a calibration run.
    """
    try:
        from pipeline.config import REPO_ROOT
        cwd = REPO_ROOT
    except (ImportError, AttributeError):
        cwd = Path(__file__).resolve().parent.parent.parent

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        log.warning(
            "label_calibrate: git rev-parse exited %d -- using 'unknown'",
            result.returncode,
        )
    except FileNotFoundError:
        log.warning("label_calibrate: git binary not found -- using 'unknown'")
    except Exception as exc:  # noqa: BLE001
        log.warning("label_calibrate: git rev lookup failed (%s) -- using 'unknown'", exc)
    return "unknown"


# ── entrypoint ───────────────────────────────────────────────────────────────


def run_label_calibrate() -> dict:
    """Run the full label calibration pass and write calibration.json.

    Opens a read-only catalog connection, calls :func:`calibrate` to compute
    all constants, writes the result to ``LABELS_CALIBRATION_PATH`` (creating
    parent directories as needed), and returns a small summary dict.

    This is a synchronous, CLI-style entrypoint -- no async, no orchestrator
    involvement.  Invoke directly or via ``pipe run label.calibrate``.

    Returns
    -------
    dict with keys:
      ``path``                      -- absolute path of the written JSON file
      ``n_rate_samples``            -- number of rate_raw values used
      ``n_pitch_speakers_calibrated`` -- speakers with a per-speaker baseline
      ``version``                   -- version string embedded in the JSON
    """
    log.info("label_calibrate: starting calibration pass")
    t0 = time.monotonic()

    with connect_ro() as conn:
        calibration = calibrate(conn)

    out_path: Path = LABELS_CALIBRATION_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(calibration, fh, ensure_ascii=False, indent=2)
        fh.write("\n")  # POSIX trailing newline

    elapsed = time.monotonic() - t0
    n = calibration["n_samples"]
    log.info(
        "label_calibrate: wrote %s  |  rate n=%d (p25=%.3f, p75=%.3f)  "
        "|  pitch speakers=%d (fallback corpus sigma=%.3f)  |  %.1fs",
        out_path,
        n["rate"],
        calibration["rate"]["p25"],
        calibration["rate"]["p75"],
        n["pitch_speakers_calibrated"],
        calibration["pitch"]["corpus_fallback"]["sigma"],
        elapsed,
    )

    return {
        "path": str(out_path),
        "n_rate_samples": n["rate"],
        "n_pitch_speakers_calibrated": n["pitch_speakers_calibrated"],
        "version": calibration["version"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run_label_calibrate()
    print(json.dumps(result, indent=2))
