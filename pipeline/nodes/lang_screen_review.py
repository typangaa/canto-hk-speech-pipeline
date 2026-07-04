"""
pipeline/nodes/lang_screen_review.py

Interactive human-in-the-loop CLI that lets a reviewer work through the queue of
raw files flagged as ``needs_review`` by the automated ``lang_screen.auto`` node.
The reviewer listens to each file's sampled windows via ``ffplay`` and assigns a
human decision (pass / reject / mixed), which is written back to the ``lang_screen``
DuckDB table immediately so that a partial session (Ctrl-C, ``q``, unexpected exit)
never loses decisions that were already made.

Design rationale
----------------
- Single-writer, direct UPDATE per decision: this is a low-throughput, single-operator
  interactive tool, not a high-throughput DAG node.  The journal-first batching pattern
  used by other nodes (segment.py, lang_screen.py) is intentionally absent here; a plain
  direct UPDATE after each decision is simpler and equally safe.
- One read-write catalog connection is held open for the whole session and closed in a
  ``finally`` block so it is always released cleanly regardless of how the session ends.
- ``ffplay`` errors (missing binary, bad file) are logged as warnings and do not crash the
  session; the reviewer can still decide from the printed ratios alone.
- ``window_starts`` is stored as DuckDB JSON; we handle both the already-decoded list
  case and the JSON-string case defensively.
- Importing ``WINDOW_SEC`` from ``pipeline.nodes.lang_screen`` keeps the playback
  duration in sync with the window length used during automated screening without
  duplicating the constant.
"""

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from pipeline.catalog.catalog import connect
from pipeline.config import LOGS_DIR
from pipeline.nodes.lang_screen import WINDOW_SEC

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Queue discovery SQL
# ---------------------------------------------------------------------------

_QUEUE_SQL = """
    SELECT ls.raw_id,
           ls.decision,
           ls.cantonese_ratio_raw,
           ls.mandarin_ratio_raw,
           ls.n_windows,
           ls.window_starts,
           rf.wav_path,
           rf.source,
           rf.program,
           rf.duration_sec
    FROM lang_screen ls
    JOIN raw_files rf ON ls.raw_id = rf.raw_id
    WHERE ls.needs_review
      AND ls.human_decision IS NULL
    ORDER BY ls.raw_id
"""

_REMAINING_SQL = """
    SELECT COUNT(*) FROM lang_screen
    WHERE needs_review AND human_decision IS NULL
"""

_UPDATE_SQL = """
    UPDATE lang_screen
    SET human_decision = ?,
        reviewed_by    = ?,
        reviewed_at    = ?
    WHERE raw_id = ?
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ts() -> str:
    """Current UTC timestamp as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _parse_window_starts(raw) -> list[float]:
    """Return window_starts as a Python list[float] regardless of how DuckDB returned it."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def _play_window(wav_path: str, start_sec: float) -> None:
    """Play one WINDOW_SEC window from *wav_path* starting at *start_sec* via ffplay.

    Failures (missing binary, non-zero exit, permission error) are logged as warnings
    so the review loop can continue even without a working ``ffplay`` installation.
    """
    cmd = [
        "ffplay",
        "-nodisp",
        "-autoexit",
        "-loglevel", "quiet",
        "-ss", str(start_sec),
        "-t", str(WINDOW_SEC),
        wav_path,
    ]
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            log.warning(
                "ffplay exited with code %d for %s @ %.2fs",
                result.returncode, wav_path, start_sec,
            )
    except FileNotFoundError:
        log.warning("ffplay not found — cannot play audio (install ffmpeg to enable playback)")
    except Exception as exc:  # noqa: BLE001
        log.warning("ffplay error for %s @ %.2fs: %s", wav_path, start_sec, exc)


def _prompt_decision() -> str:
    """Prompt the reviewer for a decision; loop until a valid single-letter is entered.

    Returns one of: 'k', 'r', 'm', 's', 'q'.
    Raises KeyboardInterrupt transparently so the caller can handle Ctrl-C.
    """
    valid = {"k", "r", "m", "s", "q"}
    while True:
        try:
            raw = input("[k]eep(pass) / [r]eject / [m]ixed / [s]kip (leave for later) / [q]uit: ")
        except EOFError:
            # stdin closed — treat like quit
            return "q"
        choice = raw.strip().lower()
        if choice in valid:
            return choice
        print(f"  Invalid input '{raw}'. Please enter one of: k, r, m, s, q")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_lang_screen_review(
    reviewer: str | None = None,
    limit: int | None = None,
) -> dict:
    """Interactive human-in-the-loop review of the lang_screen queue.

    Presents each flagged raw file to the reviewer in order, plays back the
    sampled windows via ``ffplay``, and writes the reviewer's decision directly
    to the ``lang_screen`` table.  The session can be quit cleanly at any time
    with ``q`` or Ctrl-C; already-made decisions are always persisted.

    Parameters
    ----------
    reviewer:
        Name to record in ``reviewed_by``.  Falls back to the ``$USER``
        environment variable, then to the string ``"unknown"``.
    limit:
        Maximum number of queued items to present in this session.  ``None``
        means present all queued items.

    Returns
    -------
    dict with keys:
      ``reviewed``   -- number of raw_ids given a human decision this session
      ``skipped``    -- number of raw_ids explicitly skipped (still in queue)
      ``remaining``  -- queue depth after this session ends (re-queried from DB)
    """
    # ── logging setup ────────────────────────────────────────────────────────
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "lang_screen_review.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    log.addHandler(file_handler)
    # Ensure a console handler is present when run as a library (not via __main__).
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in log.handlers
    ):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        log.addHandler(console_handler)
    log.setLevel(logging.INFO)

    reviewer_name: str = reviewer or os.environ.get("USER", "unknown")
    log.info(
        "lang_screen_review: starting session reviewer=%s limit=%s window_sec=%.1f",
        reviewer_name, limit, WINDOW_SEC,
    )

    reviewed = 0
    skipped = 0
    conn = connect()
    try:
        # ── discover queue ───────────────────────────────────────────────────
        rows = conn.execute(_QUEUE_SQL).fetchall()
        if limit is not None:
            rows = rows[:limit]

        total = len(rows)
        log.info("lang_screen_review: %d item(s) in review queue (limit=%s)", total, limit)

        if total == 0:
            print("No items in the review queue — all caught up!")

        # ── per-item review loop ─────────────────────────────────────────────
        for idx, row in enumerate(rows, start=1):
            (
                raw_id, decision, cantonese_ratio_raw, mandarin_ratio_raw,
                n_windows, window_starts_raw, wav_path, source, program, duration_sec,
            ) = row

            window_starts = _parse_window_starts(window_starts_raw)

            # Compact header
            print()
            print(f"{'─' * 70}")
            print(f"  [{idx}/{total}] raw_id             : {raw_id}")
            print(f"          source             : {source}  |  program: {program}")
            print(f"          duration_sec       : {duration_sec}")
            print(f"          auto decision      : {decision}")
            print(f"          cantonese_ratio_raw: {cantonese_ratio_raw}")
            print(f"          mandarin_ratio_raw : {mandarin_ratio_raw}")
            print(f"          n_windows          : {n_windows}")
            print(f"{'─' * 70}")

            # Play each window in order
            for win_idx, start_sec in enumerate(window_starts, start=1):
                print(
                    f"  ▶ window {win_idx}/{len(window_starts)}"
                    f" @ {start_sec:.2f}s (dur={WINDOW_SEC:.1f}s)"
                )
                _play_window(wav_path, start_sec)

            # Prompt for decision
            try:
                choice = _prompt_decision()
            except KeyboardInterrupt:
                print(
                    "\n  [Ctrl-C] — ending session early. "
                    "Already-made decisions are saved."
                )
                log.info(
                    "lang_screen_review: KeyboardInterrupt after %d reviewed, %d skipped",
                    reviewed, skipped,
                )
                break

            if choice == "q":
                print("  [quit] — ending session. Already-made decisions are saved.")
                log.info(
                    "lang_screen_review: quit after %d reviewed, %d skipped",
                    reviewed, skipped,
                )
                break

            if choice == "s":
                skipped += 1
                log.info("lang_screen_review: SKIP %s (idx=%d)", raw_id, idx)
                continue

            # k / r / m — record decision immediately
            human_decision = {"k": "pass", "r": "reject", "m": "mixed"}[choice]
            now = _now_ts()
            conn.execute(_UPDATE_SQL, [human_decision, reviewer_name, now, raw_id])
            reviewed += 1
            log.info(
                "lang_screen_review: DECIDED %s -> human_decision=%s reviewed_by=%s",
                raw_id, human_decision, reviewer_name,
            )
            print(f"  ✓ Recorded: {human_decision}")

        # ── remaining queue depth ─────────────────────────────────────────────
        remaining = conn.execute(_REMAINING_SQL).fetchone()[0]

    finally:
        conn.close()

    summary = {"reviewed": reviewed, "skipped": skipped, "remaining": remaining}
    log.info("lang_screen_review: session complete %s", summary)
    print()
    print(
        f"Session summary — reviewed: {reviewed}  "
        f"skipped: {skipped}  remaining in queue: {remaining}"
    )
    return summary


# ---------------------------------------------------------------------------
# Standalone entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Interactive human review of the lang_screen needs_review queue."
    )
    ap.add_argument(
        "--reviewer",
        default=None,
        help="Reviewer name to record in the catalog (default: $USER env var).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of queued items to review in this session.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = run_lang_screen_review(reviewer=args.reviewer, limit=args.limit)
    print(json.dumps(result, indent=2))
