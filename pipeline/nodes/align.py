"""
pipeline/nodes/align.py
align.chars DAG node — P0 of docs/PAUSE_TOKEN_PUNCTUATION_PLAN.md: char-level forced
alignment via Qwen3-ForcedAligner-0.6B-hf, so every character of a segment's
human-verified/statistically-trusted text gets a (start_sec, end_sec) timestamp.
This is the prerequisite for pause.calibrate + pause.plan (P1/P2 of that plan) — those
need a real Δt at each in-sentence punctuation mark, and the plan's own §0 measurement
found VAD-gap-count only lines up with punctuation-count 12.2% of the time (n=11,611),
i.e. sequence-matching gaps to punctuation is not viable — forced alignment is the only
way to get a trustworthy per-character timestamp.

Scope: gold + auto_gold tiers only (owner decision, PAUSE_TOKEN_PUNCTUATION_PLAN.md
"Owner 拍板" item ④). Those are the two tiers whose `asr_agreement.best_text` is either
human-verified (gold) or statistically high-confidence (auto_gold — agreement>=0.92 AND
dnsmos>=3.5, sample-QA'd); silver/bronze text is noisier and not worth spending aligner
GPU time on for a pause-calibration signal that will anchor `canto-tts`'s vocab design.

Install — ISOLATED VENV, do not install into the shared `.venv` (fixed 2026-07-21 after
the pilot run crashed on model load, see DECISIONS.md same date): the Qwen3-ForcedAligner
architecture needs the unreleased transformers dev branch, pinned to commit
29985e67cccdddef7e336d7e53840500359d30a3 (transformers 5.15.0.dev0). That dev build is
NOT compatible with `qwen_asr` (asr.transcribe's active qwen3_asr backend) — installing
it into the shared `.venv` makes `import qwen_asr` raise
`TypeError: check_model_inputs() missing 1 required positional argument: 'func'`
(transformers changed that decorator's signature between 4.57.6 and the dev commit),
silently breaking the corpus's primary ASR model. The two backends cannot coexist in one
interpreter, so this node's worker subprocess is spawned from a second, dedicated venv
instead of `sys.executable`:
    uv venv .venv_align --python 3.12
    uv pip install --python .venv_align/bin/python torch numpy soundfile soxr accelerate
    uv pip install --python .venv_align/bin/python \
        "git+https://github.com/huggingface/transformers@29985e67cccdddef7e336d7e53840500359d30a3"
(NOT `uv sync` in either venv — CLAUDE.md hard rule, would prune the CUDA-torch install
outside lock tracking). The main `.venv` must stay on the PyPI-release transformers
(4.57.6 at verification time) for qwen_asr/funasr/speechbrain/faster_whisper — never
`uv pip install` the git transformers there. `ALIGN_VENV_PYTHON` below resolves the
worker's interpreter relative to this file; `run_align_chars` raises a clear error at
spawn time if `.venv_align` hasn't been created yet.

Model: Qwen/Qwen3-ForcedAligner-0.6B-hf (Apache-2.0, native Cantonese, word-level
timestamps — see ALIGN_SR/worker below). For CJK input, "word" in this model's output
resolves to one entry per character (verified 2026-07-21 against a real catalog
gold/auto_gold segment: `word_lists` for a 20-character Cantonese sentence produced 20
single-character entries, punctuation dropped automatically by
`prepare_forced_aligner_inputs()` — no manual strip step needed on our side for this
node; the punctuation-offset-remapping work described in the plan §3 is P2's job, not
P0's). Hence this table is genuinely char-level and is named `alignments.chars`, not
`units` (see the module's row-shape comment on the schema.sql table for the naming
rationale mandated by the calling task).

Known edge case (verified 2026-07-21, not auto-fixed here — flagged for the P1
calibration pass to characterise, not silently patched): a small number of trailing/
low-energy characters can come back with a zero-width span (start_sec == end_sec),
apparently a word-boundary rounding artifact rather than a real zero-duration sound.
Stored as-is; do not assume every (start, end) pair has end > start downstream.

Node shape: same GPUWorkerBase + JSONL worker-subprocess protocol as every other GPU
node (see asr.py's Qwen3ASRWorker for the closest analog — also a transformers-backend
HF model). Unlike asr.transcribe (N models × devices), this is ONE model across a
device list, so the supervisor loop mirrors label_suite.py/label_prosody.py's shape
(shared asyncio.Queue, N workers self-balance by pulling batches until it's empty)
rather than asr.py's per-(model,device) assignment shape.

Audio: 16 kHz mono transient resample via pipeline.audio.bus.decode() (this repo's
shared decode-once bus) — never touches the 48 kHz FLAC/WAV master.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from pipeline.audio.bus import decode
from pipeline.workers.gpu_base import GPUWorkerBase

log = logging.getLogger(__name__)

ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
ALIGN_SR = 16000
ALIGN_LANGUAGE = "yue"  # Cantonese — verified working against FORCED_ALIGNER_LANGUAGES

# Isolated venv for the worker subprocess — see module docstring "Install" section for
# why this can't be the shared `.venv` (dev-transformers vs. qwen_asr conflict).
# pipeline/nodes/align.py -> parents[2] == repo root.
ALIGN_VENV_PYTHON = Path(__file__).resolve().parents[2] / ".venv_align" / "bin" / "python"


# ---------------------------------------------------------------------------
# Catalog discovery (supervisor side)
# ---------------------------------------------------------------------------

DISCOVER_SQL = """
    SELECT a.id, a.best_text, s.audio_path
    FROM asr_agreement a
    JOIN tiers t ON a.id = t.id AND t.tier IN ('gold', 'auto_gold')
    JOIN segments s ON a.id = s.id
    LEFT JOIN alignments al ON a.id = al.id AND al.provenance = 'qwen3_aligner'
    WHERE a.best_text IS NOT NULL AND a.best_text <> '' AND al.id IS NULL
"""


def discover(conn) -> list[tuple]:
    return conn.execute(DISCOVER_SQL).fetchall()


# ---------------------------------------------------------------------------
# Supervisor: align.chars — one model across a device list, shared-queue
# self-balancing dispatch (mirrors label_suite.py / label_prosody.py).
# ---------------------------------------------------------------------------

def _batches(rows: list[tuple], size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


async def run_align_chars(
    devices: list[str],
    *,
    conn=None,
    gpu_policy: str = "cap",
    batch_size: int = 64,
    mem_fraction: float | None = None,
    limit: int | None = None,
    prefetch: int = 2,
    io_workers: int = 16,
) -> dict:
    """Supervisor entrypoint for the align.chars node.

    devices: e.g. ["cuda:0", "cuda:1"] — one worker subprocess per device, all
    running the same model (unlike asr.transcribe's per-model-key assignment
    list — see this module's docstring). Rows are dispatched via one shared
    asyncio.Queue so devices self-balance (no static round-robin split needed).

    conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run align.chars` usage.

    batch_size (2026-07-21, raised 16 -> 64): AlignerWorker.forward_batch() now
    runs a genuine single-call batched forward (see that class's docstring) instead
    of looping per item, so batch_size directly controls GPU utilization/VRAM, not
    just dispatch granularity. A single synthetic worst-case forward (all-~20s-
    duration real catalog segments, the corpus's segment duration ceiling) peaked at
    ~12.1 GB allocated at bs=64 (~15.2 GB at bs=96, ~18.5 GB at bs=128, bs=160
    OOM'd) on a 24 GB 4090 -- but a REAL sustained production run (--limit 4000
    against the live backlog, 2026-07-21) showed nvidia-smi's reserved figure climb
    substantially higher than that single-call estimate, to ~19-21 GB at both
    bs=64 and bs=96, because PyTorch's caching allocator never shrinks its
    reservation back down once a batch happens to skew long, and over thousands of
    randomly-composed batches (duration 3-20s per segment, no ORDER BY in
    DISCOVER_SQL) that eventually happens regardless of nominal batch_size. Both
    bs=64 and bs=96 hit a handful of transient CUDA OOM warnings early in that same
    test run (recovered cleanly via `infer_with_oom_halving`, gpu_base.py -- final
    processed=4000/errors=0 both times, throughput ~92-94 items/s combined across
    both GPUs either way) -- bs=64 hit fewer such events for the same throughput,
    so it's the safer default of the two rather than 96. `infer_with_oom_halving`
    remains the real safety net for the rare long-tail batch composition; this
    default is chosen to need it less often, not to eliminate it.

    prefetch (2026-07-21, ported from asr.py's run_asr_transcribe — see that
    function's docstring for the general rationale): number of tasks kept in
    flight per worker so the worker's preprocess stage (decode+resample) overlaps
    the GPU forward of the previous task. The device pool is acquired around the
    SEND only (dispatch gating); in-flight tasks are never preempted. 1 restores
    the old strictly-sequential behaviour.

    io_workers: decode+resample thread-pool size inside each worker subprocess.
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
    log.info(f"align.chars: {len(rows)} segments to align")
    if not rows:
        return {"processed": 0, "errors": 0}

    if not ALIGN_VENV_PYTHON.exists():
        raise RuntimeError(
            f"align.chars worker venv not found at {ALIGN_VENV_PYTHON} — the "
            "Qwen3-ForcedAligner dev-transformers build must not be installed into the "
            "shared .venv (breaks qwen_asr, see module docstring). Create it with:\n"
            "  uv venv .venv_align --python 3.12\n"
            "  uv pip install --python .venv_align/bin/python torch numpy soundfile "
            "soxr accelerate\n"
            "  uv pip install --python .venv_align/bin/python "
            '"git+https://github.com/huggingface/transformers'
            '@29985e67cccdddef7e336d7e53840500359d30a3"'
        )

    registry = PoolRegistry()
    pool_names = []
    for dev in devices:
        pool_name = f"gpu.{dev.split(':')[1]}" if dev.startswith("cuda") else "cpu"
        registry.register(pool_name, target=1)
        pool_names.append(pool_name)

    handles = {}
    for dev, pool_name in zip(devices, pool_names):
        # Isolated venv, NOT sys.executable — see ALIGN_VENV_PYTHON comment above.
        cmd = [str(ALIGN_VENV_PYTHON), "-m", "pipeline.nodes.align", "--device", dev,
               "--io-workers", str(io_workers)]
        if mem_fraction is not None and dev.startswith("cuda"):
            cmd += ["--mem-fraction", str(mem_fraction)]
        handle = await spawn_worker(cmd)
        await handle.wait_ready(timeout=180.0)
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

    run_id = new_run_id("align.chars")
    queue: asyncio.Queue = asyncio.Queue()
    for batch in _batches(rows, batch_size):
        queue.put_nowait(batch)

    processed = 0
    errors = 0
    t0 = time.time()

    async def worker_loop(pool_name: str, handle) -> None:
        nonlocal processed, errors
        pool = registry.get(pool_name)

        # Double-buffered dispatch (2026-07-21, ported from asr.py's
        # run_asr_transcribe — see run_align_chars docstring): keep up to
        # `prefetch` tasks in flight so the worker's preprocess stage decodes
        # task N+1 while its GPU runs task N. Pool acquired around the SEND
        # only; results matched by task_id (worker emits FIFO, but matching
        # by id keeps a late straggler after a timeout from being attributed
        # to the wrong batch).
        inflight: dict[str, list[tuple]] = {}
        seq = 0

        async def dispatch_next() -> bool:
            nonlocal seq
            try:
                batch = queue.get_nowait()
            except asyncio.QueueEmpty:
                return False
            task_id = f"{pool_name}-{seq}"
            seq += 1
            items = [{"id": r[0], "text": r[1], "path": r[2]} for r in batch]
            async with pool.acquire():
                await handle.send_task(task_id, items)
            inflight[task_id] = batch
            return True

        while True:
            while len(inflight) < max(1, prefetch) and await dispatch_next():
                pass
            if not inflight:
                return
            try:
                result = await handle.read_message(timeout=300.0)
            except Exception as e:
                # A timeout/death can't be attributed to one specific task —
                # fail everything currently in flight for this worker (same
                # accounting as the old per-batch failure path, scaled to the
                # window). A late straggler result is dropped by the
                # unknown-task_id guard below.
                n_failed = sum(len(b) for b in inflight.values())
                log.error(f"{pool_name}: read failed with {len(inflight)} task(s) "
                          f"in flight ({n_failed} segments): {e}")
                errors += n_failed
                for batch in inflight.values():
                    queue.task_done()
                inflight.clear()
                continue

            batch = inflight.pop(result.get("task_id"), None)
            if batch is None:
                log.warning(f"{pool_name}: result for unknown/expired task "
                            f"{result.get('task_id')!r} — dropped")
                continue
            if result["type"] == "error":
                log.error(f"{pool_name}: worker error: {result['error']}")
                errors += len(batch)
                queue.task_done()
                continue

            out_rows = [
                {"id": r["id"], "chars": r["chars"], "model": ALIGNER_MODEL_ID,
                 "provenance": "qwen3_aligner"}
                for r in result["rows"]
            ]
            skipped_rows = [
                {"id": sid, "chars": None, "model": ALIGNER_MODEL_ID,
                 "provenance": "qwen3_aligner"}
                for sid in result.get("skipped_ids", [])
            ]
            if skipped_rows:
                log.warning(f"{pool_name}: {len(skipped_rows)} unreadable/unalignable "
                            f"segment(s): {[r['id'] for r in skipped_rows][:5]}")

            upsert_rows(conn, "alignments", out_rows + skipped_rows, ["id"])
            record_batch(conn, run_id, "align.chars", [r["id"] for r in out_rows], "ok",
                         metrics=result.get("metrics"))
            if skipped_rows:
                record_batch(conn, run_id, "align.chars", [r["id"] for r in skipped_rows],
                             "error", error="unreadable audio or alignment failure")

            processed += len(out_rows) + len(skipped_rows)
            errors += len(skipped_rows)
            queue.task_done()
            if processed and processed % (batch_size * 20) < batch_size:
                rate = processed / (time.time() - t0)
                log.info(f"{processed}/{len(rows)} aligned ({rate:.1f}/s), "
                         f"pools={registry.snapshot()}")

    await asyncio.gather(*(
        worker_loop(pool_name, handles[pool_name]) for pool_name in pool_names
    ))

    sampler.stop()
    await asyncio.gather(sampler_task, return_exceptions=True)
    for handle in handles.values():
        await handle.shutdown()

    elapsed = time.time() - t0
    log.info(f"DONE: {processed} processed, {errors} errors in {elapsed:.0f}s "
             f"({processed / elapsed if elapsed > 0 else 0:.1f}/s), run_id={run_id}")
    return {"processed": processed, "errors": errors, "run_id": run_id}


# ---------------------------------------------------------------------------
# GPU worker (subprocess side)
# ---------------------------------------------------------------------------

class AlignerWorker(GPUWorkerBase):
    """Worker for Qwen3-ForcedAligner-0.6B-hf.

    Real batched-tensor path CONFIRMED (2026-07-21, measured against real catalog
    gold/auto_gold segments in .venv_align — see PROGRESS.md/pending_task.md for the
    repro): `prepare_forced_aligner_inputs(audio=[...], transcript=[...])` accepts
    lists and returns one padded batch (`input_ids`/`attention_mask`/`input_features`/
    `input_features_mask`, shape `[B, ...]`) plus one `word_lists` entry per item; a
    SINGLE `self.model(**aligner_inputs)` forward processes the whole batch, and
    `decode_forced_alignment(...)` returns one timestamp list per batch item (index
    0..N-1, not always [0]). Verified byte-for-byte identical predicted characters
    between batched and single-item calls across 8 real segments (17-105 chars,
    3.3-11.1s audio); start/end timestamps matched exactly in most cases with rare
    sub-300ms drift on a handful of interior characters (padding-attention numerical
    noise, not a text/ordering bug — the decode grid itself only has ~0.08s
    resolution). This drift is far below the noise floor already accepted for the
    pause-calibration signal this table feeds (P1 of PAUSE_TOKEN_PUNCTUATION_PLAN.md)
    and is not worth chasing further here.

    Old code (kept only in git history) looped `_align_one()` per item — i.e. a
    "batch" of 16 was 16 sequential tiny forward passes, the actual root cause of the
    ~41 rows/s combined-GPU throughput measured on the killed full run. Superseded by
    the real single-call batched forward below.
    """

    def load_model(self):
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor

        log.info(f"Loading forced aligner: {ALIGNER_MODEL_ID} on {self.device} [bfloat16]")
        self.processor = AutoProcessor.from_pretrained(ALIGNER_MODEL_ID)
        model = AutoModelForTokenClassification.from_pretrained(
            ALIGNER_MODEL_ID, dtype=torch.bfloat16, device_map=self.device,
        )
        return model

    def forward_batch(self, items: list[dict]) -> list[dict]:
        import torch

        wavs = [it["y16"] for it in items]
        texts = [it["text"] for it in items]
        aligner_inputs, word_lists = self.processor.prepare_forced_aligner_inputs(
            audio=wavs, transcript=texts, language=ALIGN_LANGUAGE,
        )
        aligner_inputs = aligner_inputs.to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            outputs = self.model(**aligner_inputs)
        per_item_timestamps = self.processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=aligner_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=self.model.config.timestamp_token_id,
        )  # one list per batch item, index 0..N-1 (not always [0] — see class docstring)
        return [
            {"chars": [[t["text"], t["start_time"], t["end_time"]] for t in timestamps]}
            for timestamps in per_item_timestamps
        ]


# ---------------------------------------------------------------------------
# Worker subprocess entrypoint — JSONL over stdio
# ---------------------------------------------------------------------------

def worker_main() -> None:
    """Worker-subprocess entrypoint — 3-stage producer-consumer pipeline (2026-07-21),
    ported from asr.py's worker_main() (see that function's docstring for the general
    rationale: decode-then-compute serially idles the GPU during every batch's decode
    phase). Same shape here:

        [stdin reader thread] -> raw_q -> [preprocess thread + IO pool] -> ready_q -> [main thread: GPU + emit]

    so decode of task N+1 overlaps GPU forward of task N. The supervisor
    (run_align_chars) keeps `prefetch` tasks in flight per worker to keep this queue
    fed. Only the main thread writes to stdout.
    """
    import queue as queue_mod
    import threading

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mem-fraction", type=float, default=None)
    ap.add_argument("--io-workers", type=int, default=16)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                         format="%(asctime)s %(levelname)s %(message)s")

    worker = AlignerWorker(args.device, mem_fraction=args.mem_fraction)

    def emit(msg: dict) -> None:
        # Called from the main thread only — single writer, no interleaving.
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    emit({"type": "ready", "node": "align.chars", "pid": __import__("os").getpid(), "proto": 1})

    ex = ThreadPoolExecutor(max_workers=args.io_workers)
    raw_q: queue_mod.Queue = queue_mod.Queue(maxsize=2)
    ready_q: queue_mod.Queue = queue_mod.Queue(maxsize=2)

    def reader_loop() -> None:
        """stdin -> raw_q. A None sentinel (shutdown message or stdin EOF) flows
        through both queues so each stage drains in-flight work before exiting."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg["type"] == "shutdown":
                break
            if msg["type"] != "task":
                continue
            raw_q.put(msg)
        raw_q.put(None)

    def preprocess_loop() -> None:
        """raw_q -> decode+resample (GIL-releasing sf/soxr on the IO pool) -> ready_q."""
        while True:
            msg = raw_q.get()
            if msg is None:
                ready_q.put(None)
                return
            task_id, items = msg["task_id"], msg["items"]
            t0 = time.time()
            try:
                paths = [it["path"] for it in items]
                wavs = list(ex.map(lambda p: decode(p, ALIGN_SR), paths))
                keep_idx = [i for i, w in enumerate(wavs) if w is not None and len(w) >= ALIGN_SR // 10]
                skipped_ids = [items[i]["id"] for i in range(len(items)) if i not in set(keep_idx)]
                kept_items = [
                    {"id": items[i]["id"], "text": items[i]["text"], "y16": wavs[i]}
                    for i in keep_idx
                ]
                ready_q.put(("task", task_id, kept_items, skipped_ids, t0))
            except Exception as e:
                ready_q.put(("error", task_id, str(e)))

    threading.Thread(target=reader_loop, daemon=True, name="stdin-reader").start()
    threading.Thread(target=preprocess_loop, daemon=True, name="preprocess").start()

    # Main thread: GPU inference + stdout emit.
    while True:
        entry = ready_q.get()
        if entry is None:
            break
        if entry[0] == "error":
            _, task_id, err = entry
            emit({"type": "error", "task_id": task_id, "error": err, "retryable": True})
            continue
        _, task_id, kept_items, skipped_ids, t0 = entry
        try:
            if not kept_items:
                emit({"type": "result", "task_id": task_id, "rows": [],
                      "skipped_ids": skipped_ids, "metrics": {"items_s": 0.0}})
                continue
            results = worker.infer_with_oom_halving(kept_items)
            rows = [{"id": it["id"], **res} for it, res in zip(kept_items, results)]
            elapsed = time.time() - t0
            emit({"type": "result", "task_id": task_id, "rows": rows, "skipped_ids": skipped_ids,
                  "metrics": {"items_s": round(len(rows) / elapsed, 2) if elapsed > 0 else 0.0}})
        except Exception as e:
            emit({"type": "error", "task_id": task_id, "error": str(e), "retryable": True})

    ex.shutdown(wait=False)


if __name__ == "__main__":
    worker_main()
