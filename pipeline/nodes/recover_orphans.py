#!/usr/bin/env python3
"""
pipeline/nodes/recover_orphans.py
recover.orphans DAG node (one-time, 2026-07-06) -- the pre-P0 legacy pipeline
(scripts/03_segment.py) VAD-cut every candidate clip straight to disk under
Drive4/canto/segments/{source}/, but only the ones that survived its filter
stage ever made it into manifest.jsonl (and hence the P0-imported `segments`
catalog table). ~730k candidate WAVs never got cleaned up or imported --
"orphans" this node discovers and classifies using whatever sidecar metadata
survives next to each one (.pregate.json from scripts/03b_acoustic_pregate.py,
.transcript.json from the legacy ASR stage).

Two outcomes per orphan, NEVER a third silent one:
  - RECOVER: sidecar evidence already looks acceptable (acoustic pregate
    passed, or ASR cross-model agreement >= HIGH_AGREEMENT_THRESHOLD).
    Backfill segments/asr_results/asr_agreement so the segment looks
    EXACTLY like a normal freshly-cut one -- the actual accept/reject and
    gold/silver/excluded call is then made by the real, current
    filter.text / filter.acoustic / filter.decide / tier.assign nodes, not
    by this node's own judgment. This node's thresholds only gate what's
    worth re-importing at all, not the final quality bar.
  - pending_delete: recorded in `orphan_segments` with status='pending_delete'
    -- NO file is touched. A later, separately-approved cleanup node reads
    this queue to actually reclaim disk space.

Idempotent: discovery excludes any audio_path already in `segments` (already
catalog-known) or already in `orphan_segments` (already classified by a prior
run of this node).

Usage:
    python -m pipeline.cli run recover.orphans [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

HIGH_AGREEMENT_THRESHOLD = 0.80  # CLAUDE.md's stated ASR-agreement quality bar --
                                 # an import-eligibility pre-filter, NOT the final
                                 # tier cutoff (tier.assign's own SILVER_AGREE_MIN
                                 # is looser, 0.65, and still applies downstream).

BATCH_SIZE = 2000


def _segments_root() -> Path:
    from pipeline.nodes.segment import SEGMENTS_ROOT
    return SEGMENTS_ROOT


def _segment_id(path: Path) -> str:
    from pipeline.nodes.segment import _segment_id as seg_id_fn
    return seg_id_fn(path)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover(conn) -> list[tuple[str, str]]:
    """Return (source, wav_path) for every physical WAV under SEGMENTS_ROOT not
    already in `segments` (catalog-known) or `orphan_segments` (already
    classified by a prior run)."""
    kept = {row[0] for row in conn.execute("SELECT audio_path FROM segments").fetchall()}
    classified = {row[0] for row in conn.execute("SELECT audio_path FROM orphan_segments").fetchall()}
    skip = kept | classified

    root = _segments_root()
    out = []
    for source in sorted(os.listdir(root)):
        src_dir = root / source
        if not src_dir.is_dir():
            continue
        with os.scandir(src_dir) as it:
            for entry in it:
                if not entry.name.endswith(".wav"):
                    continue
                if entry.path in skip:
                    continue
                out.append((source, entry.path))
    return out


# ---------------------------------------------------------------------------
# Per-file classification -- pure sidecar-reading logic, no catalog access.
# ---------------------------------------------------------------------------

def classify_one(wav_path: str) -> dict:
    """Classify one orphan WAV using its sidecar files. Returns:
      {"bucket": str, "recover": bool, "pregate": dict|None, "transcript": dict|None}
    Never raises -- unreadable sidecars are treated as absent."""
    stem = wav_path[:-4] if wav_path.endswith(".wav") else wav_path
    pregate_path = stem + ".pregate.json"
    transcript_path = stem + ".transcript.json"

    pregate = None
    if os.path.exists(pregate_path):
        try:
            with open(pregate_path, "r", encoding="utf-8") as f:
                pregate = json.load(f)
        except Exception:
            pregate = None

    transcript = None
    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = json.load(f)
        except Exception:
            transcript = None

    if pregate is not None:
        if pregate.get("pass") is True:
            return {"bucket": "pregate_pass", "recover": True, "pregate": pregate, "transcript": transcript}
        reason = pregate.get("reason", "unknown")
        return {"bucket": f"pregate_fail_{reason}", "recover": False, "pregate": pregate, "transcript": transcript}

    if transcript is not None:
        agreement = transcript.get("asr_agreement")
        if agreement is not None and agreement >= HIGH_AGREEMENT_THRESHOLD:
            return {"bucket": "transcript_high_agreement", "recover": True, "pregate": None, "transcript": transcript}
        return {"bucket": "transcript_low_agreement", "recover": False, "pregate": None, "transcript": transcript}

    return {"bucket": "no_sidecar_at_all", "recover": False, "pregate": None, "transcript": None}


def build_recovery_rows(seg_id: str, wav_path: str, source: str, transcript: dict,
                         duration_sec: float, sample_rate: int, today: str) -> dict:
    """Build the segments / asr_results / asr_agreement rows for a RECOVER
    orphan. *transcript* may be None (pregate-pass orphans without a surviving
    transcript.json) -- asr_agreement/asr_results are then omitted, and the
    segment enters with no ASR yet (same shape asr.transcribe would produce
    pre-transcription; filter.text's discovery requires an asr_agreement row,
    so such a segment simply waits for that stage same as any fresh cut)."""
    segments_row = {
        "id": seg_id, "audio_path": wav_path, "source": source,
        "source_url": None, "program": None, "domain": None,
        "duration_sec": duration_sec, "sample_rate": sample_rate,
        "speaker_id": None, "gender": None, "style": None,
        "created_at": today, "raw_id": None,
    }

    asr_agreement_row = None
    asr_results_rows = []
    if transcript is not None:
        asr_agreement_row = {
            "id": seg_id,
            "agreement": transcript.get("asr_agreement"),
            "best_text": transcript.get("text"),
            "text_verified": bool(transcript.get("text_verified", False)),
        }
        for cand in transcript.get("asr_candidates", []):
            asr_results_rows.append({
                "id": seg_id, "model": cand.get("model"),
                "text": cand.get("text"), "confidence": cand.get("confidence"),
            })

    return {
        "segments_row": segments_row,
        "asr_agreement_row": asr_agreement_row,
        "asr_results_rows": asr_results_rows,
    }


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

async def run_recover_orphans(*, conn=None, limit: int | None = None) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run recover.orphans` usage."""
    import soundfile as sf

    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    log.info("recover.orphans: scanning physical segments dirs for catalog-unknown WAVs...")
    rows = discover(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"recover.orphans: {len(rows)} orphan(s) to classify")
    if not rows:
        return {"scanned": 0, "recovered": 0, "pending_delete": 0, "errors": 0}

    run_id = new_run_id("recover.orphans")
    today = datetime.now(timezone.utc).date().isoformat()
    now = datetime.now(timezone.utc)

    scanned = 0
    recovered = 0
    pending_delete = 0
    errors = 0
    t0 = time.time()

    seg_batch, agree_batch, asr_batch, orphan_batch = [], [], [], []

    def flush():
        nonlocal seg_batch, agree_batch, asr_batch, orphan_batch
        if seg_batch:
            upsert_rows(conn, "segments", seg_batch, ["id"])
        if agree_batch:
            upsert_rows(conn, "asr_agreement", agree_batch, ["id"])
        if asr_batch:
            upsert_rows(conn, "asr_results", asr_batch, ["id", "model"])
        if orphan_batch:
            upsert_rows(conn, "orphan_segments", orphan_batch, ["audio_path"])
        seg_batch, agree_batch, asr_batch, orphan_batch = [], [], [], []

    for source, wav_path in rows:
        scanned += 1
        try:
            size = os.path.getsize(wav_path)
        except OSError as exc:
            log.warning(f"recover.orphans: stat failed {wav_path}: {exc}")
            errors += 1
            continue

        result = classify_one(wav_path)
        bucket = result["bucket"]

        if result["recover"]:
            try:
                info = sf.info(wav_path)
            except Exception as exc:
                log.warning(f"recover.orphans: unreadable audio, queued for delete instead {wav_path}: {exc}")
                orphan_batch.append({
                    "audio_path": wav_path, "source": source, "bucket": "unreadable_audio",
                    "bytes": size, "status": "pending_delete", "recovered_id": None,
                    "classified_at": now,
                })
                pending_delete += 1
                continue

            seg_id = _segment_id(Path(wav_path))
            built = build_recovery_rows(
                seg_id, wav_path, source, result["transcript"],
                round(info.frames / info.samplerate, 3), info.samplerate, today,
            )
            seg_batch.append(built["segments_row"])
            if built["asr_agreement_row"] is not None:
                agree_batch.append(built["asr_agreement_row"])
                asr_batch.extend(built["asr_results_rows"])
            orphan_batch.append({
                "audio_path": wav_path, "source": source, "bucket": bucket,
                "bytes": size, "status": "recovered", "recovered_id": seg_id,
                "classified_at": now,
            })
            recovered += 1
        else:
            orphan_batch.append({
                "audio_path": wav_path, "source": source, "bucket": bucket,
                "bytes": size, "status": "pending_delete", "recovered_id": None,
                "classified_at": now,
            })
            pending_delete += 1

        if len(orphan_batch) >= BATCH_SIZE:
            flush()
            record_batch(conn, run_id, "recover.orphans", [], "ok")
            rate = scanned / (time.time() - t0) if time.time() > t0 else 0.0
            log.info(f"recover.orphans: {scanned}/{len(rows)} ({rate:.1f}/s) -- "
                     f"recovered={recovered}, pending_delete={pending_delete}, errors={errors}")

    flush()

    elapsed = time.time() - t0
    log.info(
        f"recover.orphans DONE: {scanned} scanned, {recovered} recovered, "
        f"{pending_delete} queued pending_delete, {errors} errors, {elapsed:.0f}s, run_id={run_id}"
    )
    return {
        "scanned": scanned, "recovered": recovered, "pending_delete": pending_delete,
        "errors": errors, "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# recover.reingest_pending -- one-time, 2026-07-12. Owner decision: the
# pending_delete bucket was classified using LEGACY ASR/pregate evidence
# (the old single/dual-model ASR and the old scripts/03b_acoustic_pregate.py
# SNR formula). The current 3-model ASR ensemble (canto_ft/qwen3_asr/
# sense_voice) measurably outperforms the legacy ASR (qwen3_asr ~0.4% CER on
# calibration samples vs 17-36% for the others -- see CLAUDE.md's "ASR
# strategy"), so a segment the legacy pipeline rejected for low ASR agreement
# may well pass under the current models. Rather than delete this bucket,
# re-admit it into `segments` (NO pre-seeded ASR text -- the whole point is a
# clean re-transcription, not reuse of the old candidates) so it flows
# through the CURRENT pregate.snr/asr.transcribe/asr.agreement/filter.*/
# tier.assign nodes exactly like a fresh cut, and let those nodes make the
# real call. `orphan_segments.status` moves pending_delete -> re_admitted
# (a third status value, deliberately distinct from `recovered`, so the
# original one-time classification's audit trail stays intact).
# ---------------------------------------------------------------------------

REINGEST_BATCH_SIZE = 5000


def discover_reingest(conn) -> list[tuple[str, str]]:
    """Return (source, wav_path) for every orphan_segments row still queued
    pending_delete. Idempotent: once reingested a row's status flips to
    're_admitted', dropping it out of this WHERE clause."""
    return conn.execute(
        "SELECT source, audio_path FROM orphan_segments WHERE status = 'pending_delete'"
    ).fetchall()


async def run_reingest_pending(*, conn=None, limit: int | None = None,
                                batch_size: int = REINGEST_BATCH_SIZE) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) — see other
    nodes' docstrings for the `pipe run-many` rationale. Defaults to a fresh
    self-managed connect() for standalone `pipe run recover.reingest_pending`
    usage."""
    import soundfile as sf

    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    log.info("recover.reingest_pending: scanning orphan_segments for pending_delete rows...")
    rows = discover_reingest(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"recover.reingest_pending: {len(rows)} orphan(s) to re-admit for a fresh pipeline pass")
    if not rows:
        return {"scanned": 0, "admitted": 0, "unreadable": 0}

    run_id = new_run_id("recover.reingest_pending")
    today = datetime.now(timezone.utc).date().isoformat()

    scanned = 0
    admitted = 0
    unreadable = 0
    t0 = time.time()

    seg_batch: list[dict] = []
    orphan_paths: list[str] = []

    def flush():
        nonlocal seg_batch, orphan_paths
        if seg_batch:
            upsert_rows(conn, "segments", seg_batch, ["id"])
        if orphan_paths:
            placeholders = ",".join(["?"] * len(orphan_paths))
            conn.execute(
                f"UPDATE orphan_segments SET status = 're_admitted' WHERE audio_path IN ({placeholders})",
                orphan_paths,
            )
        seg_batch, orphan_paths = [], []

    for source, wav_path in rows:
        scanned += 1
        try:
            info = sf.info(wav_path)
        except Exception as exc:
            log.warning(f"recover.reingest_pending: unreadable, left queued {wav_path}: {exc}")
            unreadable += 1
            continue

        seg_id = _segment_id(Path(wav_path))
        seg_batch.append({
            "id": seg_id, "audio_path": wav_path, "source": source,
            "source_url": None, "program": None, "domain": None,
            "duration_sec": round(info.frames / info.samplerate, 3),
            "sample_rate": info.samplerate,
            "speaker_id": None, "gender": None, "style": None,
            "created_at": today, "raw_id": None,
        })
        orphan_paths.append(wav_path)
        admitted += 1

        if len(orphan_paths) >= batch_size:
            flush()
            record_batch(conn, run_id, "recover.reingest_pending", [], "ok")
            rate = scanned / (time.time() - t0) if time.time() > t0 else 0.0
            log.info(f"recover.reingest_pending: {scanned}/{len(rows)} ({rate:.1f}/s) -- "
                     f"admitted={admitted}, unreadable={unreadable}")

    flush()

    elapsed = time.time() - t0
    log.info(
        f"recover.reingest_pending DONE: {scanned} scanned, {admitted} admitted, "
        f"{unreadable} unreadable (left queued), {elapsed:.0f}s, run_id={run_id}"
    )
    return {"scanned": scanned, "admitted": admitted, "unreadable": unreadable, "run_id": run_id}


def main() -> int:
    import asyncio

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_recover_orphans(limit=args.limit))
    print(f"\nDone: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
