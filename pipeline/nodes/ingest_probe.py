"""
pipeline/nodes/ingest_probe.py
ingest.probe DAG node — ffprobe metadata (duration/sample_rate/channels/codec) per raw
source file, plus an L/R correlation sample for stereo-vs-dual-mono discrimination.

Feeds the LABEL_FRAMEWORK_SPEC.md §11 / REARCHITECTURE_IMPLEMENTATION_PLAN.md §10 Q6
production-audio stereo feasibility question: is any meaningful fraction of raw source
audio genuinely stereo (correlated-but-distinct channels) vs mono duplicated into two
channels (dual-mono, in which case stereo storage buys nothing)? This node only
*measures* — it never re-encodes or deletes anything.

CPU-only, no GPU/torch — ffprobe subprocess + a short soundfile read per raw file, both
cheap enough to run fully concurrently via a thread pool without any GPU-coexistence
policy (contrast pipeline/nodes/label_music.py, which needs the orchestrator's Sampler).

Discovery: raw_files not yet in raw_probe (anti-join, same idiom as label_music.discover).
"""

import argparse
import asyncio
import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

# How much audio to read (from the start) for the L/R correlation sample — enough to be
# a meaningful statistic without reading multi-hour raw files in full.
CORR_SAMPLE_SEC = 30.0

# ---------------------------------------------------------------------------
# Catalog discovery
# ---------------------------------------------------------------------------

DISCOVER_SQL = """
    SELECT r.raw_id, r.wav_path
    FROM raw_files r
    LEFT JOIN raw_probe p ON r.raw_id = p.raw_id
    WHERE p.raw_id IS NULL
    ORDER BY r.raw_id
"""


def discover(conn) -> list[tuple]:
    return conn.execute(DISCOVER_SQL).fetchall()


# ---------------------------------------------------------------------------
# Per-file probe (worker-pool side, plain function — no subprocess-worker protocol
# needed here: ffprobe + a short sf.read both release the GIL, and there is no GPU
# to protect from a co-running training job).
# ---------------------------------------------------------------------------

def _ffprobe(path: str) -> dict | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", "-select_streams", "a:0", path],
            capture_output=True, text=True, timeout=30, check=True,
        )
    except Exception as e:
        log.warning(f"ffprobe fail {path}: {e}")
        return None
    try:
        meta = json.loads(out.stdout)
        stream = meta["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        log.warning(f"ffprobe unparseable output {path}: {e}")
        return None
    return {
        "channels": int(stream.get("channels", 0)) or None,
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        "duration_sec": float(stream["duration"]) if stream.get("duration") else None,
    }


def _lr_correlation(path: str, channels: int | None) -> float | None:
    if channels != 2:
        return None
    try:
        frames = None
        with sf.SoundFile(path) as f:
            frames = int(min(CORR_SAMPLE_SEC * f.samplerate, f.frames))
            y = f.read(frames=frames, dtype="float32", always_2d=True)
    except Exception as e:
        log.warning(f"correlation read fail {path}: {e}")
        return None
    if y.shape[1] != 2 or y.shape[0] < 2:
        return None
    left, right = y[:, 0], y[:, 1]
    if left.std() == 0 or right.std() == 0:
        return None  # silent/constant channel — correlation undefined
    r = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(r):
        return None
    return r


def probe_one(raw_id: str, path: str) -> dict | None:
    if not Path(path).exists():
        log.warning(f"missing file, skipping: {path}")
        return None
    meta = _ffprobe(path)
    if meta is None:
        return None
    meta["raw_id"] = raw_id
    meta["lr_correlation"] = _lr_correlation(path, meta["channels"])
    return meta


# ---------------------------------------------------------------------------
# Supervisor: thread-pool fan-out, batched catalog commits (journal-backed, resumable)
# ---------------------------------------------------------------------------

async def run_ingest_probe(
    *,
    workers: int = 8,
    batch_size: int = 200,
    limit: int | None = None,
) -> dict:
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = connect()
    rows = discover(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"ingest.probe: {len(rows)} raw files to probe")
    if not rows:
        return {"processed": 0, "errors": 0}

    run_id = new_run_id("ingest.probe")
    processed = 0
    errors = 0
    t0 = time.time()
    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            futures = [
                loop.run_in_executor(ex, probe_one, raw_id, path)
                for raw_id, path in batch
            ]
            results = await asyncio.gather(*futures)

            probed_at = datetime.now(timezone.utc)
            out_rows = []
            failed_ids = []
            for (raw_id, _path), res in zip(batch, results):
                if res is None:
                    failed_ids.append(raw_id)
                    continue
                out_rows.append({
                    "raw_id": raw_id,
                    "channels": res["channels"],
                    "codec": res["codec"],
                    "sample_rate": res["sample_rate"],
                    "duration_sec": res["duration_sec"],
                    "lr_correlation": res["lr_correlation"],
                    "probed_at": probed_at,
                })

            if out_rows:
                upsert_rows(conn, "raw_probe", out_rows, ["raw_id"])
                record_batch(conn, run_id, "ingest.probe",
                             [r["raw_id"] for r in out_rows], "ok")
            if failed_ids:
                record_batch(conn, run_id, "ingest.probe", failed_ids, "error",
                             error="ffprobe/read failed")
                log.warning(f"{len(failed_ids)} raw file(s) failed to probe: {failed_ids[:5]}")

            processed += len(out_rows)
            errors += len(failed_ids)
            rate = processed / (time.time() - t0) if time.time() > t0 else 0.0
            log.info(f"{processed + errors}/{len(rows)} probed ({rate:.1f}/s)")

    elapsed = time.time() - t0
    log.info(f"DONE: {processed} probed, {errors} errors in {elapsed:.0f}s, run_id={run_id}")
    return {"processed": processed, "errors": errors, "run_id": run_id}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_ingest_probe(
        workers=args.workers, batch_size=args.batch, limit=args.limit,
    ))
    print(f"\nDone: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
