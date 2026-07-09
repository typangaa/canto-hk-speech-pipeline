#!/usr/bin/env python3
"""
pipeline/nodes/rebalance.py
rebalance.segments DAG node (P5-C) — spread the `segments` table's physical
files across the 3-way disk shard defined in config/storage_layout.yaml, so a
single drive (Drive4, 843G pre-P5-C) doesn't hold the entire working corpus.

Shard key: hash(coalesce(raw_id, id)) % n_shards (config/storage_layout.py
shard_index/shard_root — the SAME function pipeline/nodes/segment.py's
segment.vad_cut consults for every NEW segment, so this node only ever needs
to migrate the pre-existing backlog once; new writes already land correctly
sharded). raw_id is only populated for the 11,411 segments cut since P3 S4 —
the 455,299 P0 legacy-imported rows fall back to their own `id`.

Two independent passes, run as separate CLI invocations so each can be
reviewed (same shape as raw_flac.py's P5-B design):
  1. copy (default): for every segment whose current file isn't already under
     its target shard root, copy it there and verify the copy is byte-for-byte
     identical to the source (plain file copy, not a re-encode, so a checksum
     comparison is the right check — no PCM decode needed). Writes
     `segment_shard_migrations` rows. Never touches segments.audio_path or
     deletes the original file. Segments already on their correct shard are
     recorded immediately (verified=true, migrated_at=now, no I/O) so they're
     never rescanned.
  2. --delete-verified: for ids with verified=true and migrated_at IS NULL,
     atomically (single DB transaction) repoints segments.audio_path at the
     new file and stamps migrated_at, THEN (only after that commit succeeds)
     deletes the original file — a crash between the two leaves the catalog
     already consistent, at worst a harmless leftover copy, never a dangling
     reference.

Usage:
    python -m pipeline.cli run rebalance.segments [--limit N] [--batch-gb G] [--workers N]
    python -m pipeline.cli run rebalance.segments --delete-verified [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# Streaming block size for checksum verification — mirrors raw_flac.py's
# BLOCK_FRAMES rationale (never load a multi-minute segment fully into RAM).
CHUNK_BYTES = 8 * 1024 * 1024

# ---------------------------------------------------------------------------
# Catalog discovery
# ---------------------------------------------------------------------------

DISCOVER_SQL = """
    SELECT s.id, s.audio_path, s.raw_id, s.source
    FROM segments s
    WHERE s.id NOT IN (SELECT id FROM segment_shard_migrations)
    ORDER BY s.id
"""

DELETE_VERIFIED_SQL = """
    SELECT id, old_path, new_path
    FROM segment_shard_migrations
    WHERE verified = true AND migrated_at IS NULL
    ORDER BY id
"""


def _target(seg_id: str, audio_path: str, raw_id: str | None, source: str) -> tuple[int, str]:
    from config.storage_layout import shard_index, shard_root

    key = raw_id if raw_id else seg_id
    idx = shard_index(key)
    new_path = str(shard_root(key) / source / Path(audio_path).name)
    return idx, new_path


def discover_copy(conn) -> list[tuple]:
    """Return (id, old_path, new_path, target_shard) for segments not yet
    migrated, EXCLUDING already-in-place rows (those are recorded immediately
    in the copy pass, without a physical copy, so they never show up here on
    a re-run)."""
    rows = conn.execute(DISCOVER_SQL).fetchall()
    out = []
    for seg_id, audio_path, raw_id, source in rows:
        idx, new_path = _target(seg_id, audio_path, raw_id, source)
        out.append((seg_id, audio_path, new_path, idx))
    return out


def discover_delete_verified(conn) -> list[tuple]:
    return conn.execute(DELETE_VERIFIED_SQL).fetchall()


# ---------------------------------------------------------------------------
# Per-file copy + verify (thread-pool worker — I/O bound, no GPU/CPU decode).
# ---------------------------------------------------------------------------

def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _verify_copy(old_path: str, new_path: str) -> tuple[bool, str | None]:
    try:
        if os.path.getsize(old_path) != os.path.getsize(new_path):
            return False, "size mismatch"
        if _md5(old_path) != _md5(new_path):
            return False, "checksum mismatch"
        return True, None
    except OSError as exc:
        return False, str(exc)


def _copy_one(seg_id: str, old_path: str, new_path: str, target_shard: int) -> dict:
    """Copy *old_path* to *new_path* (already-in-place rows never reach this
    function — see run_rebalance_copy) and verify byte-for-byte equality.
    Returns a segment_shard_migrations row dict. Never deletes or modifies the
    source file."""
    import shutil

    now = datetime.now(timezone.utc)
    dst = Path(new_path)

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_path, new_path)
    except Exception as exc:
        log.error(f"rebalance.segments: copy failed {seg_id} ({old_path} -> {new_path}): {exc}")
        Path(new_path).unlink(missing_ok=True)
        return {
            "id": seg_id, "old_path": old_path, "new_path": None, "target_shard": target_shard,
            "verified": False, "migrated_at": None, "copied_at": now, "provenance": "copy_failed",
        }

    ok, err = _verify_copy(old_path, new_path)
    if not ok:
        log.error(f"rebalance.segments: verify failed {seg_id} ({old_path} vs {new_path}): {err}")
        Path(new_path).unlink(missing_ok=True)
        return {
            "id": seg_id, "old_path": old_path, "new_path": None, "target_shard": target_shard,
            "verified": False, "migrated_at": None, "copied_at": now, "provenance": "copy_failed",
        }

    return {
        "id": seg_id, "old_path": old_path, "new_path": new_path, "target_shard": target_shard,
        "verified": True, "migrated_at": None, "copied_at": now, "provenance": "rebalance",
    }


# ---------------------------------------------------------------------------
# Supervisor: copy pass
# ---------------------------------------------------------------------------

async def run_rebalance_copy(
    *,
    conn=None,
    workers: int = 8,
    batch_size: int = 200,
    batch_gb: float | None = None,
    limit: int | None = None,
) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run rebalance.segments` usage."""
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    rows = discover_copy(conn)

    already_in_place = [r for r in rows if r[1] == r[2]]
    to_copy = [r for r in rows if r[1] != r[2]]

    # Record already-correctly-sharded segments immediately — no I/O, so they
    # never count against --limit / --batch-gb (those budget actual copies).
    if already_in_place:
        now = datetime.now(timezone.utc)
        placed_rows = [
            {
                "id": seg_id, "old_path": old_path, "new_path": new_path, "target_shard": idx,
                "verified": True, "migrated_at": now, "copied_at": now, "provenance": "already_in_place",
            }
            for seg_id, old_path, new_path, idx in already_in_place
        ]
        upsert_rows(conn, "segment_shard_migrations", placed_rows, ["id"])
        log.info(f"rebalance.segments: {len(already_in_place)} already on their target shard, recorded")

    if limit:
        to_copy = to_copy[:limit]
    if batch_gb is not None:
        to_copy = _cap_by_size(to_copy, batch_gb)
    log.info(f"rebalance.segments copy: {len(to_copy)} segment(s) eligible this invocation")
    if not to_copy:
        return {"processed": 0, "verified": 0, "failed": 0, "already_in_place": len(already_in_place)}

    run_id = new_run_id("rebalance.segments")
    processed = 0
    verified_count = 0
    failed = 0
    t0 = time.time()
    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(to_copy), batch_size):
            batch = to_copy[i : i + batch_size]
            futures = [
                loop.run_in_executor(ex, _copy_one, seg_id, old_path, new_path, idx)
                for seg_id, old_path, new_path, idx in batch
            ]
            results = await asyncio.gather(*futures)

            upsert_rows(conn, "segment_shard_migrations", results, ["id"])
            ok_ids = [r["id"] for r in results if r["verified"]]
            bad_ids = [r["id"] for r in results if not r["verified"]]
            if ok_ids:
                record_batch(conn, run_id, "rebalance.segments", ok_ids, "ok")
            if bad_ids:
                record_batch(conn, run_id, "rebalance.segments", bad_ids, "error",
                             error="copy or verify failed")

            processed += len(results)
            verified_count += len(ok_ids)
            failed += len(bad_ids)
            rate = processed / (time.time() - t0) if time.time() > t0 else 0.0
            log.info(f"rebalance.segments: {processed}/{len(to_copy)} ({rate:.2f}/s) — verified={verified_count}, failed={failed}")

    elapsed = time.time() - t0
    log.info(
        f"rebalance.segments copy DONE: {processed} processed, {verified_count} verified, "
        f"{failed} failed, {elapsed:.0f}s, run_id={run_id}"
    )
    return {
        "processed": processed, "verified": verified_count, "failed": failed,
        "already_in_place": len(already_in_place), "run_id": run_id,
    }


def _cap_by_size(rows: list[tuple], batch_gb: float) -> list[tuple]:
    """Greedily take rows (in discovery order) until their cumulative on-disk
    byte size reaches *batch_gb* GiB — sized by the source file (what actually
    bounds this invocation's I/O), matching raw_flac.py's convention."""
    budget = batch_gb * (1024 ** 3)
    out = []
    total = 0
    for seg_id, old_path, new_path, idx in rows:
        try:
            size = os.path.getsize(old_path)
        except OSError:
            size = 0
        out.append((seg_id, old_path, new_path, idx))
        total += size
        if total >= budget:
            break
    return out


# ---------------------------------------------------------------------------
# Supervisor: delete-verified pass
# ---------------------------------------------------------------------------

def _delete_one_verified(conn, seg_id: str, new_path: str) -> tuple[str, bool, str | None]:
    """Atomically repoint segments.audio_path at *new_path* and stamp
    segment_shard_migrations.migrated_at in ONE transaction, then (only after
    that commits) physically delete the original file."""
    row = conn.execute(
        "SELECT audio_path FROM segments WHERE id = ?", [seg_id]
    ).fetchone()
    if row is None:
        return seg_id, False, "id not found in segments"
    old_path = row[0]

    try:
        conn.begin()
        conn.execute(
            "UPDATE segments SET audio_path = ? WHERE id = ?", [new_path, seg_id],
        )
        conn.execute(
            "UPDATE segment_shard_migrations SET migrated_at = ? WHERE id = ?",
            [datetime.now(timezone.utc), seg_id],
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        return seg_id, False, f"catalog transaction failed: {exc}"

    try:
        Path(old_path).unlink(missing_ok=True)
    except OSError as exc:
        log.warning(f"rebalance.segments: catalog updated but failed to unlink {old_path}: {exc}")

    return seg_id, True, None


async def run_rebalance_delete_verified(*, conn=None, limit: int | None = None) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run rebalance.segments
    --delete-verified` usage. Converted from a plain `def` to `async def`
    2026-07-07 so it can join a run-many group like every other node — the
    body itself stays synchronous/blocking, same as before."""
    from pipeline.catalog.catalog import connect
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    rows = discover_delete_verified(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"rebalance.segments --delete-verified: {len(rows)} verified migration(s) with old copy still present")
    if not rows:
        return {"deleted": 0, "errors": 0}

    run_id = new_run_id("rebalance.segments.delete")
    deleted = 0
    errors = 0
    for seg_id, old_path, new_path in rows:
        _, ok, err = _delete_one_verified(conn, seg_id, new_path)
        if ok:
            deleted += 1
            record_batch(conn, run_id, "rebalance.segments.delete", [seg_id], "ok")
        else:
            errors += 1
            record_batch(conn, run_id, "rebalance.segments.delete", [seg_id], "error", error=err)
            log.error(f"rebalance.segments --delete-verified: {seg_id}: {err}")

    log.info(f"rebalance.segments --delete-verified DONE: {deleted} deleted, {errors} errors, run_id={run_id}")
    return {"deleted": deleted, "errors": errors, "run_id": run_id}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=200, help="items per catalog-commit batch")
    ap.add_argument("--batch-gb", type=float, default=None, help="stop after ~this many GiB of source file this invocation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delete-verified", action="store_true",
                     help="delete original file for already-verified migrations instead of copying")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.delete_verified:
        result = run_rebalance_delete_verified(limit=args.limit)
    else:
        result = asyncio.run(run_rebalance_copy(
            workers=args.workers, batch_size=args.batch,
            batch_gb=args.batch_gb, limit=args.limit,
        ))
    print(f"\nDone: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
