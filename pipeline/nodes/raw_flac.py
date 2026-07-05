#!/usr/bin/env python3
"""
pipeline/nodes/raw_flac.py
raw.flac DAG node (P5-B) — transcode the existing raw WAV backlog to lossless FLAC.

Storage-format decision (2026-07-04, DECISIONS.md "Storage format policy FINALIZED"):
raw backlog = FLAC, not opus (that direction was reopened and rejected the same day).
Native containers from `ingest.download` (webm/opus, m4a/AAC — 2026-07-04 policy) are
NEVER touched here: re-encoding an already-lossy source to FLAC only inflates size
(measured 2.1-3.5x bigger) with zero fidelity gain, so this node only ever targets
`raw_files.wav_path LIKE '%.wav'`.

Eligibility (see TRANSCODE_DISCOVER_SQL): a raw file must be EITHER already segmented
(`raw_segments` row exists — cut first, so a mid-transcode crash never forces
re-decoding the source for segmentation) OR permanently excluded from segmentation
(`lang_screen` decision = 'reject' — those raw_ids never enter `raw_segments` via
segment.diarize, so this is the only path that ever reclaims their WAV space).

Two independent passes, run as separate CLI invocations so each can be reviewed:
  1. transcode (default): stream-copy WAV -> FLAC, then verify PCM bit-exact via a
     block-by-block decode comparison against the original. Writes `raw_flac` rows.
     Never touches or deletes the original .wav.
  2. --delete-verified: for raw_ids with verified=true and wav_deleted_at IS NULL,
     atomically (single DB transaction) repoints raw_files.wav_path at the FLAC file
     and stamps wav_deleted_at, THEN (only after that commit succeeds) deletes the
     physical .wav — so a crash between the two leaves the catalog already
     consistent (pointing at the FLAC) with, at worst, a harmless leftover .wav on
     disk, never a catalog row pointing at a file that no longer exists.

Usage:
    python -m pipeline.cli run raw.flac [--limit N] [--batch-gb G] [--workers N]
    python -m pipeline.cli run raw.flac --delete-verified [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

# Streaming block size for both transcode-copy and bit-exact verify — small enough
# to never load a multi-hour raw file fully into RAM, large enough to keep I/O
# efficient (matches the "generous but bounded" spirit of ingest_probe.py's
# CORR_SAMPLE_SEC choice).
BLOCK_FRAMES = 1_000_000

# ---------------------------------------------------------------------------
# Catalog discovery
# ---------------------------------------------------------------------------

TRANSCODE_DISCOVER_SQL = """
    SELECT r.raw_id, r.wav_path
    FROM raw_files r
    WHERE r.wav_path LIKE '%.wav'
      AND r.raw_id NOT IN (SELECT raw_id FROM raw_flac)
      AND (
            r.raw_id IN (SELECT raw_id FROM raw_segments)
            OR r.raw_id IN (
                SELECT raw_id FROM lang_screen
                WHERE COALESCE(human_decision, decision) = 'reject'
            )
          )
    ORDER BY r.raw_id
"""

DELETE_VERIFIED_SQL = """
    SELECT rf.raw_id, rf.flac_path
    FROM raw_flac rf
    WHERE rf.verified = true AND rf.wav_deleted_at IS NULL
    ORDER BY rf.raw_id
"""


def discover_transcode(conn) -> list[tuple]:
    """Return (raw_id, wav_path) for raw files eligible for FLAC transcode."""
    return conn.execute(TRANSCODE_DISCOVER_SQL).fetchall()


def discover_delete_verified(conn) -> list[tuple]:
    """Return (raw_id, flac_path) for verified transcodes still holding the original wav."""
    return conn.execute(DELETE_VERIFIED_SQL).fetchall()


# ---------------------------------------------------------------------------
# Per-file transcode + verify (thread-pool worker, plain function — CPU/IO bound,
# no GPU, same shape as ingest_probe.py's probe_one).
# ---------------------------------------------------------------------------

def _transcode_one(raw_id: str, wav_path: str) -> dict:
    """Stream-copy *wav_path* to a sibling .flac, then verify it decodes back to
    the exact same PCM samples.

    Returns a raw_flac row dict.  On any failure, `verified=False` and
    `provenance='transcode_failed'` — never deletes or modifies the source .wav.
    """
    flac_path = str(Path(wav_path).with_suffix(".flac"))

    try:
        with sf.SoundFile(wav_path, mode="r") as src:
            samplerate = src.samplerate
            channels = src.channels
            subtype = src.subtype
            with sf.SoundFile(
                flac_path, mode="w", samplerate=samplerate, channels=channels,
                format="FLAC", subtype=subtype,
            ) as dst:
                while True:
                    block = src.read(frames=BLOCK_FRAMES, dtype="float32", always_2d=True)
                    if len(block) == 0:
                        break
                    dst.write(block)
    except Exception as exc:
        log.error(f"raw.flac: transcode failed {raw_id} ({wav_path}): {exc}")
        Path(flac_path).unlink(missing_ok=True)  # don't leave a partial file behind
        return {
            "raw_id": raw_id, "flac_path": None, "duration_sec": None,
            "verified": False, "wav_deleted_at": None, "transcoded_at": datetime.now(timezone.utc),
            "provenance": "transcode_failed",
        }

    verified, duration_sec, err = _verify_bit_exact(wav_path, flac_path)
    if not verified:
        log.error(f"raw.flac: verify failed {raw_id} ({wav_path} vs {flac_path}): {err}")
        Path(flac_path).unlink(missing_ok=True)  # never keep an unverified FLAC around
        return {
            "raw_id": raw_id, "flac_path": None, "duration_sec": None,
            "verified": False, "wav_deleted_at": None, "transcoded_at": datetime.now(timezone.utc),
            "provenance": "transcode_failed",
        }

    return {
        "raw_id": raw_id, "flac_path": flac_path, "duration_sec": duration_sec,
        "verified": True, "wav_deleted_at": None, "transcoded_at": datetime.now(timezone.utc),
        "provenance": "raw_flac",
    }


def _verify_bit_exact(wav_path: str, flac_path: str) -> tuple[bool, float | None, str | None]:
    """Decode both files block-by-block and compare for exact PCM equality.

    FLAC is lossless, so a correct transcode must reproduce the source samples
    exactly — this is a correctness check on the encode/decode round-trip
    itself, not a perceptual judgment call (unlike opus, no human listening
    is needed).
    """
    try:
        with sf.SoundFile(wav_path, mode="r") as src, sf.SoundFile(flac_path, mode="r") as dst:
            if src.frames != dst.frames or src.channels != dst.channels:
                return False, None, (
                    f"shape mismatch: src frames={src.frames} channels={src.channels} "
                    f"vs dst frames={dst.frames} channels={dst.channels}"
                )
            while True:
                a = src.read(frames=BLOCK_FRAMES, dtype="int16" if src.subtype == "PCM_16" else "float32")
                b = dst.read(frames=BLOCK_FRAMES, dtype="int16" if src.subtype == "PCM_16" else "float32")
                if len(a) == 0 and len(b) == 0:
                    break
                if not np.array_equal(a, b):
                    return False, None, "PCM sample mismatch"
            duration_sec = dst.frames / dst.samplerate
            return True, duration_sec, None
    except Exception as exc:
        return False, None, str(exc)


# ---------------------------------------------------------------------------
# Supervisor: transcode pass
# ---------------------------------------------------------------------------

async def run_raw_flac_transcode(
    *,
    workers: int = 8,
    batch_size: int = 50,
    batch_gb: float | None = None,
    limit: int | None = None,
) -> dict:
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = connect()
    rows = discover_transcode(conn)
    if limit:
        rows = rows[:limit]
    if batch_gb is not None:
        rows = _cap_by_size(rows, batch_gb)
    log.info(f"raw.flac transcode: {len(rows)} raw files eligible")
    if not rows:
        return {"processed": 0, "verified": 0, "failed": 0}

    run_id = new_run_id("raw.flac")
    processed = 0
    verified_count = 0
    failed = 0
    t0 = time.time()
    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            futures = [
                loop.run_in_executor(ex, _transcode_one, raw_id, wav_path)
                for raw_id, wav_path in batch
            ]
            results = await asyncio.gather(*futures)

            upsert_rows(conn, "raw_flac", results, ["raw_id"])
            ok_ids = [r["raw_id"] for r in results if r["verified"]]
            bad_ids = [r["raw_id"] for r in results if not r["verified"]]
            if ok_ids:
                record_batch(conn, run_id, "raw.flac", ok_ids, "ok")
            if bad_ids:
                record_batch(conn, run_id, "raw.flac", bad_ids, "error",
                             error="transcode or verify failed")

            processed += len(results)
            verified_count += len(ok_ids)
            failed += len(bad_ids)
            rate = processed / (time.time() - t0) if time.time() > t0 else 0.0
            log.info(f"raw.flac: {processed}/{len(rows)} ({rate:.2f}/s) — verified={verified_count}, failed={failed}")

    elapsed = time.time() - t0
    log.info(
        f"raw.flac transcode DONE: {processed} processed, {verified_count} verified, "
        f"{failed} failed, {elapsed:.0f}s, run_id={run_id}"
    )
    return {"processed": processed, "verified": verified_count, "failed": failed, "run_id": run_id}


def _cap_by_size(rows: list[tuple], batch_gb: float) -> list[tuple]:
    """Greedily take rows (in discovery order) until their cumulative on-disk
    byte size reaches *batch_gb* GiB.  Sizing is by the source .wav file's
    actual size, not audio duration — that's what actually bounds disk I/O
    and the eventual Drive2 space freed per batch."""
    budget = batch_gb * (1024 ** 3)
    out = []
    total = 0
    for raw_id, wav_path in rows:
        try:
            size = os.path.getsize(wav_path)
        except OSError:
            size = 0
        out.append((raw_id, wav_path))
        total += size
        if total >= budget:
            break
    return out


# ---------------------------------------------------------------------------
# Supervisor: delete-verified pass
# ---------------------------------------------------------------------------

def _delete_one_verified(conn, raw_id: str, flac_path: str) -> tuple[str, bool, str | None]:
    """Atomically repoint raw_files.wav_path at *flac_path* and stamp
    raw_flac.wav_deleted_at in ONE transaction, then (only after that commits)
    physically delete the original .wav.  A crash between commit and unlink
    leaves the catalog already consistent — worst case a harmless leftover
    .wav, never a dangling catalog reference.
    """
    row = conn.execute(
        "SELECT wav_path FROM raw_files WHERE raw_id = ?", [raw_id]
    ).fetchone()
    if row is None:
        return raw_id, False, "raw_id not found in raw_files"
    old_wav_path = row[0]

    try:
        conn.begin()
        conn.execute(
            "UPDATE raw_files SET wav_path = ? WHERE raw_id = ?", [flac_path, raw_id],
        )
        conn.execute(
            "UPDATE raw_flac SET wav_deleted_at = ? WHERE raw_id = ?",
            [datetime.now(timezone.utc), raw_id],
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        return raw_id, False, f"catalog transaction failed: {exc}"

    try:
        Path(old_wav_path).unlink(missing_ok=True)
    except OSError as exc:
        # Catalog is already correctly updated at this point — a failed unlink
        # only wastes disk space, it does not desync the catalog.
        log.warning(f"raw.flac: catalog updated but failed to unlink {old_wav_path}: {exc}")

    return raw_id, True, None


def run_raw_flac_delete_verified(*, limit: int | None = None) -> dict:
    from pipeline.catalog.catalog import connect
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = connect()
    rows = discover_delete_verified(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"raw.flac --delete-verified: {len(rows)} verified transcode(s) with wav still present")
    if not rows:
        return {"deleted": 0, "errors": 0}

    run_id = new_run_id("raw.flac.delete")
    deleted = 0
    errors = 0
    for raw_id, flac_path in rows:
        _, ok, err = _delete_one_verified(conn, raw_id, flac_path)
        if ok:
            deleted += 1
            record_batch(conn, run_id, "raw.flac.delete", [raw_id], "ok")
        else:
            errors += 1
            record_batch(conn, run_id, "raw.flac.delete", [raw_id], "error", error=err)
            log.error(f"raw.flac --delete-verified: {raw_id}: {err}")

    log.info(f"raw.flac --delete-verified DONE: {deleted} deleted, {errors} errors, run_id={run_id}")
    return {"deleted": deleted, "errors": errors, "run_id": run_id}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=50, help="items per catalog-commit batch")
    ap.add_argument("--batch-gb", type=float, default=None, help="stop after ~this many GiB of source .wav this invocation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delete-verified", action="store_true",
                     help="delete original .wav for already-verified transcodes instead of transcoding")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.delete_verified:
        result = run_raw_flac_delete_verified(limit=args.limit)
    else:
        result = asyncio.run(run_raw_flac_transcode(
            workers=args.workers, batch_size=args.batch,
            batch_gb=args.batch_gb, limit=args.limit,
        ))
    print(f"\nDone: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
