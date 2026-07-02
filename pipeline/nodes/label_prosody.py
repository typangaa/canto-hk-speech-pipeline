"""
pipeline/nodes/label_prosody.py
label.prosody DAG node — CPU control-label raw detector per LABEL_FRAMEWORK_SPEC.md §8:
voiced_sec + silence gaps via Silero VAD, F0 via parselmouth (Praat), and rate_raw
(jyutping syllables per voiced second). Writes the raw-only fields of labels_prosody;
per-speaker z-scoring of pitch and the corpus-percentile rate bucket are a later
"calibrate" pass (§9, out of scope here — this node only produces raw measurements).

CPU-only (no GPU pool/Sampler needed — mirrors the coexistence guidance in
LABEL_FRAMEWORK_SPEC.md §13: cap threads per worker so a co-running training
dataloader keeps its cores). Uses the same GPUWorkerBase + JSONL worker-subprocess
protocol as pipeline/nodes/label_music.py, but always on device="cpu" and with
multiple worker processes (one per pool "cpu.0".."cpu.{n-1}") for parallelism across
cores instead of across GPUs.

Discovery: segments not yet in labels_prosody (anti-join), joined against g2p
(syllable count) and filters (english_ratio, for the rate_raw exception rule).
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from pipeline.audio.bus import decode
from pipeline.workers.gpu_base import GPUWorkerBase

log = logging.getLogger(__name__)

VAD_SR = 16000

# Hard constraint #8's Jyutping token regex — reused here to count only genuine
# syllables (canto-hk-g2p already strips English/punctuation, but this is a
# cheap safety net against any stray non-syllable token).
JYUTPING_TOKEN = re.compile(r"^[a-z]+[1-6]$")

# LABEL_FRAMEWORK_SPEC.md §8.3: only report silence gaps >= 0.2s.
MIN_GAP_SEC = 0.2
# LABEL_FRAMEWORK_SPEC.md §8.1: rate is unreliable when English dominates the segment.
ENGLISH_RATIO_OMIT_RATE = 0.5

_vad_model = None
_vad_utils = None


def get_vad_model():
    global _vad_model, _vad_utils
    if _vad_model is None:
        log.info("Loading Silero VAD ...")
        _vad_model, _vad_utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
    return _vad_model, _vad_utils


def count_syllables(jyutping: str | None) -> int:
    if not jyutping:
        return 0
    return sum(1 for tok in jyutping.split() if JYUTPING_TOKEN.match(tok))


# ---------------------------------------------------------------------------
# Catalog discovery (supervisor side)
# ---------------------------------------------------------------------------

DISCOVER_SQL = """
    SELECT s.id, s.audio_path, s.duration_sec, g.jyutping, f.english_ratio
    FROM segments s
    LEFT JOIN labels_prosody p ON s.id = p.id
    LEFT JOIN g2p g ON s.id = g.id
    LEFT JOIN filters f ON s.id = f.id
    WHERE p.id IS NULL
    ORDER BY s.duration_sec
"""


def discover(conn) -> list[tuple]:
    return conn.execute(DISCOVER_SQL).fetchall()


# ---------------------------------------------------------------------------
# Supervisor: N CPU worker processes + JSONL protocol (mirrors label_music.py,
# swapping the GPU device/pool axis for a CPU worker-count axis)
# ---------------------------------------------------------------------------

def _batches(rows: list[tuple], size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


async def run_label_prosody(
    *,
    n_workers: int = 4,
    threads_per_worker: int = 2,
    batch_size: int = 8,
    limit: int | None = None,
) -> dict:
    """Supervisor entrypoint for the label.prosody node.

    Spawns n_workers CPU worker subprocesses (each capped to threads_per_worker
    torch threads — LABEL_FRAMEWORK_SPEC.md §13's "don't starve the training
    dataloader" guidance), dispatches length-sorted batches round-robin via a
    shared queue, and commits results through the same journal + upsert_rows
    idiom as label_music.py so kill -9 resume is free.
    """
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch
    from pipeline.orchestrator.pools import PoolRegistry
    from pipeline.orchestrator.worker import spawn_worker

    conn = connect()
    rows = discover(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"label.prosody: {len(rows)} segments to process")
    if not rows:
        return {"processed": 0, "errors": 0}

    registry = PoolRegistry()
    pool_names = [f"cpu.{i}" for i in range(n_workers)]
    for name in pool_names:
        registry.register(name, target=1)

    handles = {}
    for pool_name in pool_names:
        cmd = [
            sys.executable, "-m", "pipeline.nodes.label_prosody",
            "--threads", str(threads_per_worker),
        ]
        handle = await spawn_worker(cmd)
        await handle.wait_ready(timeout=120.0)
        handles[pool_name] = handle
        log.info(f"worker ready: {pool_name} (pid={handle.pid})")

    run_id = new_run_id("label.prosody")
    queue: asyncio.Queue = asyncio.Queue()
    for batch in _batches(rows, batch_size):
        queue.put_nowait(batch)

    processed = 0
    errors = 0
    t0 = time.time()

    async def worker_loop(pool_name: str, handle) -> None:
        nonlocal processed, errors
        pool = registry.get(pool_name)
        while True:
            try:
                batch = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            meta = {r[0]: (r[3], r[4]) for r in batch}  # id -> (jyutping, english_ratio)
            items = [{"id": r[0], "path": r[1]} for r in batch]
            async with pool.acquire():
                await handle.send_task(f"{pool_name}-{processed}", items)
                try:
                    result = await handle.read_message(timeout=180.0)
                except Exception as e:
                    log.error(f"{pool_name}: batch failed: {e}")
                    errors += len(batch)
                    queue.task_done()
                    continue
            if result["type"] == "error":
                log.error(f"{pool_name}: worker error: {result['error']}")
                errors += len(batch)
                queue.task_done()
                continue

            out_rows = []
            for r in result["rows"]:
                jyutping, english_ratio = meta[r["id"]]
                syll = count_syllables(jyutping)
                voiced_sec = r["voiced_sec"]
                rate_raw = None
                if (
                    voiced_sec and voiced_sec > 0
                    and syll > 0
                    and (english_ratio is None or english_ratio <= ENGLISH_RATIO_OMIT_RATE)
                ):
                    rate_raw = round(syll / voiced_sec, 3)
                out_rows.append({
                    "id": r["id"],
                    "rate_raw": rate_raw,
                    "f0_median_hz": r["f0_median_hz"],
                    "f0_z": None,  # per-speaker calibration — later stage
                    "gaps": r["gaps"],
                    "voiced_sec": voiced_sec,
                })

            skipped_rows = [
                {
                    "id": sid, "rate_raw": None, "f0_median_hz": None,
                    "f0_z": None, "gaps": [], "voiced_sec": None,
                }
                for sid in result.get("skipped_ids", [])
            ]
            if skipped_rows:
                log.warning(f"{pool_name}: {len(skipped_rows)} unreadable segment(s): "
                            f"{[r['id'] for r in skipped_rows][:5]}")

            upsert_rows(conn, "labels_prosody", out_rows + skipped_rows, ["id"])
            record_batch(conn, run_id, "label.prosody", [r["id"] for r in out_rows], "ok",
                         metrics=result.get("metrics"))
            if skipped_rows:
                record_batch(conn, run_id, "label.prosody", [r["id"] for r in skipped_rows],
                             "error", error="unreadable audio file")

            processed += len(out_rows) + len(skipped_rows)
            errors += len(skipped_rows)
            queue.task_done()
            if processed and processed % (batch_size * 20) < batch_size:
                rate = processed / (time.time() - t0)
                log.info(f"{processed}/{len(rows)} processed ({rate:.1f}/s)")

    await asyncio.gather(*(
        worker_loop(pool_name, handles[pool_name]) for pool_name in pool_names
    ))

    for handle in handles.values():
        await handle.shutdown()

    elapsed = time.time() - t0
    log.info(f"DONE: {processed} processed, {errors} errors in {elapsed:.0f}s "
             f"({processed / elapsed if elapsed > 0 else 0:.1f}/s), run_id={run_id}")
    return {"processed": processed, "errors": errors, "run_id": run_id}


# ---------------------------------------------------------------------------
# CPU worker (subprocess side)
# ---------------------------------------------------------------------------

class ProsodyWorker(GPUWorkerBase):
    def load_model(self):
        model, utils = get_vad_model()
        self.get_speech_timestamps = utils[0]
        return model

    def forward_batch(self, items: list[np.ndarray]) -> list[dict]:
        rows = []
        for y16 in items:
            rows.append(self._prosody_one(y16))
        return rows

    def _prosody_one(self, y16: np.ndarray) -> dict:
        duration_sec = len(y16) / VAD_SR
        voiced_sec, gaps = self._vad_pass(y16, duration_sec)
        f0_median_hz = self._f0(y16)
        return {"voiced_sec": voiced_sec, "gaps": gaps, "f0_median_hz": f0_median_hz}

    def _vad_pass(self, y16: np.ndarray, duration_sec: float) -> tuple[float, list]:
        tensor = torch.from_numpy(y16).float()
        try:
            timestamps = self.get_speech_timestamps(
                tensor, self.model, sampling_rate=VAD_SR,
                threshold=0.5, min_silence_duration_ms=int(MIN_GAP_SEC * 1000),
                min_speech_duration_ms=100,
            )
        except Exception as e:
            log.warning(f"VAD failed on a clip: {e}")
            return 0.0, []

        voiced_sec = sum((t["end"] - t["start"]) / VAD_SR for t in timestamps)
        gaps = []
        for i in range(len(timestamps) - 1):
            gap_start = timestamps[i]["end"] / VAD_SR
            gap_dur = timestamps[i + 1]["start"] / VAD_SR - gap_start
            if gap_dur >= MIN_GAP_SEC:
                gaps.append([round(gap_start, 3), round(gap_dur, 3)])
        return round(voiced_sec, 3), gaps

    def _f0(self, y16: np.ndarray) -> float | None:
        import parselmouth
        try:
            snd = parselmouth.Sound(y16.astype(np.float64), sampling_frequency=VAD_SR)
            pitch = snd.to_pitch()
            f0 = pitch.selected_array["frequency"]
            voiced = f0[f0 > 0]
            if len(voiced) == 0:
                return None
            return round(float(np.median(voiced)), 2)
        except Exception as e:
            log.warning(f"F0 extraction failed on a clip: {e}")
            return None


# ---------------------------------------------------------------------------
# Worker subprocess entrypoint — JSONL over stdio
# ---------------------------------------------------------------------------

def worker_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2,
                    help="torch CPU threads cap — keep low so co-running trainer keeps cores")
    ap.add_argument("--io-workers", type=int, default=4)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                         format="%(asctime)s %(levelname)s %(message)s")

    torch.set_num_threads(args.threads)

    worker = ProsodyWorker("cpu", fp16=False)

    def emit(msg: dict) -> None:
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    emit({"type": "ready", "node": "label.prosody", "pid": __import__("os").getpid(), "proto": 1})

    ex = ThreadPoolExecutor(max_workers=args.io_workers)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg["type"] == "shutdown":
            break
        if msg["type"] != "task":
            continue

        task_id = msg["task_id"]
        items = msg["items"]
        t0 = time.time()
        try:
            paths = [it["path"] for it in items]
            wavs = list(ex.map(lambda p: decode(p, VAD_SR), paths))
            keep_idx = [i for i, w in enumerate(wavs) if w is not None and len(w) >= VAD_SR // 10]
            skipped_ids = [items[i]["id"] for i in range(len(items)) if i not in set(keep_idx)]
            if not keep_idx:
                emit({"type": "result", "task_id": task_id, "rows": [],
                      "skipped_ids": skipped_ids, "metrics": {"items_s": 0.0}})
                continue
            kept_wavs = [wavs[i] for i in keep_idx]
            results = worker.infer_with_oom_halving(kept_wavs)
            rows = []
            for i, res in zip(keep_idx, results):
                rows.append({"id": items[i]["id"], **res})
            elapsed = time.time() - t0
            emit({"type": "result", "task_id": task_id, "rows": rows, "skipped_ids": skipped_ids,
                  "metrics": {"items_s": round(len(rows) / elapsed, 2) if elapsed > 0 else 0.0}})
        except Exception as e:
            emit({"type": "error", "task_id": task_id, "error": str(e), "retryable": True})

    ex.shutdown(wait=False)


if __name__ == "__main__":
    worker_main()
