"""
pipeline/nodes/label_music.py
label.music DAG node — PANNs CNN14 music-family probability, ported from
scripts/11_audio_tag.py onto the orchestrator's GPUWorkerBase + JSONL worker protocol.
Discovery: segments not yet in labels_music (anti-join, verified against
load_done_ids-equivalent logic in P0's pipeline/catalog/verify.py).
"""

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch
from scipy.signal import firwin, upfirdn

from pipeline.workers.gpu_base import GPUWorkerBase

log = logging.getLogger(__name__)

PANNS_SR = 32000

_RS_CACHE: dict = {}

# Same music-family keyword include / exclude as scripts/11_audio_tag.py — kept
# identical so label_music's output is comparable to the s0/s1 provenance rows
# already in labels_music (golden-set parity requires the same taxonomy).
_INCLUDE_KW = (
    "music", "jingle", "singing", "song", "choir", "rapping", "melody", "tune",
    "instrument", "guitar", "piano", "drum", "orchestr", "violin", "trumpet",
    "harmonica", "accordion", "synthesizer", "bass", "cello", "flute", "saxophone",
    "organ", "banjo", "mandolin", "harp", "trombone", "brass", "wind instrument",
    "percussion", "cymbal", "gong", "string", "keyboard (musical)", "theme",
)
_EXCLUDE_EXACT = {
    "Speech synthesizer",
    "Bird vocalization, bird call, bird song",
}


def music_indices(labels: list[str]) -> list[int]:
    idx = []
    for i, lab in enumerate(labels):
        if lab in _EXCLUDE_EXACT:
            continue
        if any(k in lab.lower() for k in _INCLUDE_KW):
            idx.append(i)
    return idx


def _resample(y: np.ndarray, sr: int) -> np.ndarray:
    if sr == PANNS_SR:
        return y
    g = math.gcd(sr, PANNS_SR)
    up, down = PANNS_SR // g, sr // g
    h = _RS_CACHE.get((up, down))
    if h is None:
        maxr = max(up, down)
        h = (firwin(2 * 10 * maxr + 1, 1.0 / maxr, window=("kaiser", 5.0)) * up
             ).astype(np.float32)
        _RS_CACHE[(up, down)] = h
    return upfirdn(h, y, up, down).astype(np.float32)


def read_audio_32k(path: str) -> np.ndarray | None:
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:
        log.warning(f"read fail {path}: {e}")
        return None
    if y.ndim > 1:
        y = y.mean(axis=1)
    return _resample(y, sr)


# ---------------------------------------------------------------------------
# Catalog discovery (supervisor side)
# ---------------------------------------------------------------------------

DISCOVER_SQL = """
    SELECT s.id, s.source, s.audio_path, s.duration_sec
    FROM segments s
    LEFT JOIN labels_music m ON s.id = m.id
    WHERE m.id IS NULL
    ORDER BY s.duration_sec
"""


def discover(conn) -> list[tuple]:
    """Length-sorted list of (id, source, audio_path, duration_sec) not yet tagged."""
    return conn.execute(DISCOVER_SQL).fetchall()


# ---------------------------------------------------------------------------
# Supervisor: pool + sampler + worker-protocol wiring (P1 pilot)
# ---------------------------------------------------------------------------

def _batches(rows: list[tuple], size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


async def run_label_music(
    devices: list[str],
    *,
    conn=None,
    gpu_policy: str = "cap",
    batch_size: int = 16,
    mem_fraction: float | None = 0.15,
    limit: int | None = None,
) -> dict:
    """Supervisor entrypoint for the label.music pilot node.

    Spawns one worker subprocess per device, gates dispatch through a
    ResourcePool per device, and lets Sampler shrink/restore each GPU pool's
    target when it detects foreign (non-orchestrator) compute processes on
    that GPU — this is what makes label.music coexist with a co-running
    training job instead of OOM-racing it.

    conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run label.music` usage.
    """
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch
    from pipeline.orchestrator.pools import PoolRegistry
    from pipeline.orchestrator.resources import GpuPolicy, Sampler
    from pipeline.orchestrator.worker import spawn_worker

    conn = conn or connect()
    rows = discover(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"label.music: {len(rows)} segments to tag")
    if not rows:
        return {"processed": 0, "errors": 0}

    registry = PoolRegistry()
    pool_names = []
    for dev in devices:
        pool_name = f"gpu.{dev.split(':')[1]}" if dev.startswith("cuda") else "cpu"
        registry.register(pool_name, target=1)
        pool_names.append(pool_name)

    handles = {}
    for dev, pool_name in zip(devices, pool_names):
        cmd = [
            sys.executable, "-m", "pipeline.nodes.label_music",
            "--device", dev,
        ]
        if mem_fraction is not None and dev.startswith("cuda"):
            cmd += ["--mem-fraction", str(mem_fraction)]
        handle = await spawn_worker(cmd)
        await handle.wait_ready(timeout=120.0)
        handles[pool_name] = handle
        log.info(f"worker ready: {pool_name} -> {dev} (pid={handle.pid})")

    gpu_policies = {
        name: GpuPolicy(gpu_policy) for name in pool_names if name.startswith("gpu.")
    }
    sampler = Sampler(
        registry, gpu_policies,
        own_pids=lambda: {h.pid for h in handles.values()},
        poll_interval=2.0,
    )
    sampler_task = asyncio.create_task(sampler.run())

    run_id = new_run_id("label.music")
    queue: asyncio.Queue = asyncio.Queue()
    for batch in _batches(rows, batch_size):
        queue.put_nowait(batch)
    n_batches = queue.qsize()

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
            meta = {r[0]: (r[1], r[3]) for r in batch}
            items = [{"id": r[0], "path": r[2]} for r in batch]
            async with pool.acquire():
                await handle.send_task(f"{pool_name}-{processed}", items)
                try:
                    result = await handle.read_message(timeout=300.0)
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
            out_rows = [
                {
                    "id": r["id"],
                    "source": meta[r["id"]][0],
                    "duration_sec": meta[r["id"]][1],
                    "music_prob": r["music_prob"],
                    "music_tags": r["music_tags"],
                    "provenance": "p1_pilot",
                }
                for r in result["rows"]
            ]
            # Unreadable / corrupt audio (e.g. zero-byte segment files) gets a
            # placeholder row too — otherwise discover()'s anti-join would keep
            # resurfacing the same dead id on every future run forever.
            skipped_rows = [
                {
                    "id": sid,
                    "source": meta[sid][0],
                    "duration_sec": meta[sid][1],
                    "music_prob": None,
                    "music_tags": [],
                    "provenance": "read_failed",
                }
                for sid in result.get("skipped_ids", [])
            ]
            if skipped_rows:
                log.warning(f"{pool_name}: {len(skipped_rows)} unreadable segment(s), "
                            f"marked provenance=read_failed: "
                            f"{[r['id'] for r in skipped_rows][:5]}")
            upsert_rows(conn, "labels_music", out_rows + skipped_rows, ["id"])
            record_batch(
                conn, run_id, "label.music", [r["id"] for r in out_rows], "ok",
                metrics=result.get("metrics"),
            )
            if skipped_rows:
                record_batch(
                    conn, run_id, "label.music", [r["id"] for r in skipped_rows],
                    "error", error="unreadable audio file",
                )
            processed += len(out_rows) + len(skipped_rows)
            errors += len(skipped_rows)
            queue.task_done()
            if processed and processed % (batch_size * 20) < batch_size:
                rate = processed / (time.time() - t0)
                log.info(f"{processed}/{len(rows)} tagged ({rate:.1f}/s), "
                         f"pools={registry.snapshot()}")

    await asyncio.gather(*(
        worker_loop(pool_name, handles[pool_name]) for pool_name in pool_names
    ))

    sampler.stop()
    await asyncio.gather(sampler_task, return_exceptions=True)
    for handle in handles.values():
        await handle.shutdown()

    elapsed = time.time() - t0
    log.info(f"DONE: {processed} tagged, {errors} errors in {elapsed:.0f}s "
             f"({processed / elapsed if elapsed > 0 else 0:.1f}/s), run_id={run_id}")
    return {"processed": processed, "errors": errors, "run_id": run_id, "n_batches": n_batches}


# ---------------------------------------------------------------------------
# GPU worker (subprocess side)
# ---------------------------------------------------------------------------

class MusicWorker(GPUWorkerBase):
    def load_model(self):
        from panns_inference import AudioTagging
        from panns_inference.config import labels

        # panns_inference prints "Checkpoint path: ..." / "Using CPU." straight to
        # stdout on init, which would corrupt the JSONL worker protocol stream —
        # redirect stdout to stderr for the duration of the load only.
        # panns_inference.AudioTagging does an *exact* string match `device == 'cuda'`
        # to decide GPU vs CPU — "cuda:0"/"cuda:1" silently fall through to CPU.
        # set_device() pins the process's default GPU so unqualified "cuda" resolves
        # to the right physical card.
        panns_device = "cuda" if str(self.device).startswith("cuda") else "cpu"
        if panns_device == "cuda":
            torch.cuda.set_device(self.device)

        real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            at = AudioTagging(checkpoint_path=None, device=panns_device)
        finally:
            sys.stdout = real_stdout
        if isinstance(at.model, torch.nn.DataParallel):
            # AudioTagging always wraps in DataParallel(device_ids=None) when its
            # (now-unqualified) device is "cuda" — DataParallel's default device_ids
            # scans ALL visible GPUs and assumes the model lives on cuda:0, which
            # breaks true one-GPU-per-worker-process fan-out on non-zero devices.
            # The weights are already correctly placed via set_device() above, so
            # just unwrap — no data-parallel splitting is wanted here anyway.
            at.model = at.model.module

        self.labels = labels
        self.mus_idx = np.array(music_indices(labels))
        log.info(f"music-family labels: {len(self.mus_idx)}")
        return at

    def forward_batch(self, items: list[np.ndarray]) -> list[dict]:
        maxlen = max(len(w) for w in items)
        batch = np.zeros((len(items), maxlen), dtype=np.float32)
        for i, w in enumerate(items):
            batch[i, : len(w)] = w
        clip, _ = self.model.inference(batch)  # (N, 527)
        rows = []
        for probs in clip:
            mprob = float(probs[self.mus_idx].max())
            top3 = np.argsort(probs)[-3:][::-1]
            tags = [[self.labels[j], round(float(probs[j]), 4)] for j in top3]
            rows.append({"music_prob": round(mprob, 4), "music_tags": tags})
        return rows


# ---------------------------------------------------------------------------
# Worker subprocess entrypoint — JSONL over stdio (pipeline/orchestrator/worker.py protocol)
# ---------------------------------------------------------------------------

def worker_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mem-fraction", type=float, default=None)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--io-workers", type=int, default=6)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                         format="%(asctime)s %(levelname)s %(message)s")

    worker = MusicWorker(args.device, mem_fraction=args.mem_fraction, fp16=args.fp16)

    def emit(msg: dict) -> None:
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    emit({"type": "ready", "node": "label.music", "pid": __import__("os").getpid(), "proto": 1})

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
            wavs = list(ex.map(read_audio_32k, paths))
            keep_idx = [i for i, w in enumerate(wavs) if w is not None and len(w) >= PANNS_SR // 10]
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
