"""
pipeline/nodes/lang_screen.py
lang_screen.auto DAG node — raw-level Mandarin-vs-Cantonese pre-filter, run BEFORE
segment.diarize (and before P5's planned raw->opus transcode). Added 2026-07-04 in
response to a recurring problem: some downloaded raw files turn out to be
Mandarin-dominant, and previously this was only ever caught by the per-SEGMENT
lang-id in label.suite/labels_lang — i.e. after the most expensive pipeline step
(pyannote diarization) had already run on a doomed file.

This node is a coarse, sampled, whole-file screen — NOT a replacement for the
existing fine-grained per-segment lang-id, which remains the final gate for
intra-episode code-switching (e.g. a Cantonese talk show with a Mandarin-speaking
guest clip). See pipeline/nodes/label_suite.py.

Design (mirrors the illustrated plan discussed with the owner):
  1. Discovery: raw_files not yet in lang_screen (plain anti-join, one row per
     raw file, never re-screened once a row exists).
  2. Sample N_WINDOWS evenly-spaced WINDOW_SEC windows from each raw file,
     skipping a MARGIN_FRAC margin at the start/end (intros/outros/music beds).
     Windows are read via partial soundfile reads (start=/frames=) so a
     multi-hour raw file is never fully decoded just to sample ~4-5 minutes
     of audio out of it.
  3. Run facebook/mms-lid-126 (same model already used by label.suite) per
     window; aggregate the top-1-language vote across windows into
     cantonese_ratio_raw / mandarin_ratio_raw.
  4. Decide pass / reject / mixed with conservative thresholds — biased toward
     false-pass over false-reject, since reject leads to the raw file being
     quarantined (and, per the two-gate deletion safety model, only ever
     physically deleted after a SEPARATE human-confirmed batch step; this node
     never deletes anything itself).
  5. Flag needs_review=true for 100% of reject/mixed decisions, plus a random
     (but reproducible — hashed on raw_id) audit sample of AUDIT_SAMPLE_FRAC of
     pass decisions, for the separate pipeline.nodes.lang_screen_review human
     review CLI to pick up.

segment.diarize's discovery query has been updated (pipeline/nodes/segment.py,
DIARIZE_DISCOVER_SQL) to skip any raw_id whose EFFECTIVE decision
(COALESCE(human_decision, decision, 'pass')) is 'reject'. A raw_id with no
lang_screen row at all (not yet screened, or predates this node) is treated as
'pass' by that COALESCE — this node is purely additive and never retroactively
blocks already-segmented or not-yet-screened raw files.
"""

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from pipeline.workers.gpu_base import GPUWorkerBase

log = logging.getLogger(__name__)

LID_MODEL_ID = "facebook/mms-lid-126"
SHARED_SR = 16000

# Sampling
N_WINDOWS = 10          # target number of windows per raw file
WINDOW_SEC = 25.0        # length of each sampled window
MARGIN_FRAC = 0.05       # skip this fraction of duration at start AND end
MIN_USABLE_SEC = WINDOW_SEC * 2  # below this usable span, take fewer/1 window

# GPU forward-pass size cap (independent of orchestrator batch_size — see
# LangScreenWorker.forward_batch's docstring for why this must be small even
# though batch_size itself can be large).
INFER_CHUNK_WINDOWS = 8

# Decision thresholds — conservative on purpose (see module docstring §4)
PASS_CANTONESE_MIN = 0.60
REJECT_MANDARIN_MIN = 0.70
REJECT_CANTONESE_MAX = 0.20

# Bounded human review triggers
AUDIT_SAMPLE_FRAC = 0.07   # ~7% random audit of automated 'pass' decisions


# ---------------------------------------------------------------------------
# Windowing (pure function — no I/O, no model — easy to unit test)
# ---------------------------------------------------------------------------

def compute_window_starts(duration_sec: float) -> list[float]:
    """Evenly-spaced window start times (seconds) for one raw file.

    Skips MARGIN_FRAC of duration at both ends. If the remaining usable span
    is too short for N_WINDOWS non-overlapping windows, falls back to fewer
    windows (down to a single centered window for very short raw files).
    """
    if duration_sec is None or duration_sec <= 0:
        return []

    margin = duration_sec * MARGIN_FRAC
    usable_start = margin
    usable_end = duration_sec - margin
    usable_span = usable_end - usable_start

    if usable_span <= WINDOW_SEC:
        start = max(0.0, (duration_sec - WINDOW_SEC) / 2)
        return [round(start, 2)]

    n = N_WINDOWS if usable_span >= MIN_USABLE_SEC else max(1, int(usable_span // WINDOW_SEC))
    starts = []
    for i in range(n):
        frac = (i + 0.5) / n
        center = usable_start + frac * usable_span
        start = center - WINDOW_SEC / 2
        start = max(usable_start, min(start, usable_end - WINDOW_SEC))
        starts.append(round(start, 2))
    return starts


# ---------------------------------------------------------------------------
# Decision logic (pure function — no I/O, no model — easy to unit test)
# ---------------------------------------------------------------------------

def aggregate_decision(top_labels: list[str]) -> dict | None:
    """Aggregate per-window top-1 language labels into ratios + a decision.

    Returns None if top_labels is empty (caller writes a read_failed row instead).
    """
    n = len(top_labels)
    if n == 0:
        return None

    cantonese_ratio = round(sum(1 for l in top_labels if l == "yue") / n, 4)
    mandarin_ratio = round(sum(1 for l in top_labels if l == "cmn") / n, 4)

    if mandarin_ratio >= REJECT_MANDARIN_MIN and cantonese_ratio <= REJECT_CANTONESE_MAX:
        decision = "reject"
    elif cantonese_ratio >= PASS_CANTONESE_MIN:
        decision = "pass"
    else:
        decision = "mixed"

    return {
        "decision": decision,
        "cantonese_ratio_raw": cantonese_ratio,
        "mandarin_ratio_raw": mandarin_ratio,
        "n_windows": n,
    }


def needs_review(raw_id: str, decision: str) -> bool:
    """100% of non-pass decisions + a reproducible (hash-based, not random.random())
    AUDIT_SAMPLE_FRAC sample of pass decisions, so re-running lang_screen.auto on
    the same raw_id always makes the same audit-sample call (idempotent)."""
    if decision != "pass":
        return True
    h = int(hashlib.md5(raw_id.encode()).hexdigest(), 16)
    return (h % 10000) < int(AUDIT_SAMPLE_FRAC * 10000)


def _now_ts() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Catalog discovery (supervisor side)
# ---------------------------------------------------------------------------

DISCOVER_SQL = """
    SELECT rf.raw_id, rf.wav_path, rf.duration_sec, rf.sample_rate
    FROM raw_files rf
    LEFT JOIN lang_screen ls ON rf.raw_id = ls.raw_id
    WHERE ls.raw_id IS NULL
    ORDER BY rf.raw_id
"""


def discover_screen(conn) -> list[tuple]:
    """Return (raw_id, wav_path, duration_sec, sample_rate) for raw_files not
    yet screened. A raw_id is never re-screened once a lang_screen row exists —
    the human_decision override path (pipeline.nodes.lang_screen_review) is the
    only way to change an already-screened raw_id's effective decision."""
    return conn.execute(DISCOVER_SQL).fetchall()


# ---------------------------------------------------------------------------
# Supervisor: pool + sampler + worker-protocol wiring (same shape as
# pipeline/nodes/label_music.py's run_label_music, generalised to raw files
# with multiple sampled windows per item instead of one array per item)
# ---------------------------------------------------------------------------

def _batches(rows: list[tuple], size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


async def run_lang_screen_auto(
    devices: list[str],
    *,
    gpu_policy: str = "cap",
    batch_size: int = 16,
    mem_fraction: float | None = 0.25,
    limit: int | None = None,
) -> dict:
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch
    from pipeline.orchestrator.pools import PoolRegistry
    from pipeline.orchestrator.resources import GpuPolicy, Sampler
    from pipeline.orchestrator.worker import spawn_worker

    conn = connect()
    rows = discover_screen(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"lang_screen.auto: {len(rows)} raw files to screen")
    if not rows:
        return {"processed": 0, "pass": 0, "reject": 0, "mixed": 0, "errors": 0}

    registry = PoolRegistry()
    pool_names = []
    for dev in devices:
        pool_name = f"gpu.{dev.split(':')[1]}" if dev.startswith("cuda") else "cpu"
        registry.register(pool_name, target=1)
        pool_names.append(pool_name)

    handles = {}
    for dev, pool_name in zip(devices, pool_names):
        cmd = [sys.executable, "-m", "pipeline.nodes.lang_screen", "--device", dev]
        if mem_fraction is not None and dev.startswith("cuda"):
            cmd += ["--mem-fraction", str(mem_fraction)]
        handle = await spawn_worker(cmd)
        await handle.wait_ready(timeout=120.0)
        handles[pool_name] = handle
        log.info(f"lang_screen.auto worker ready: {pool_name} -> {dev} (pid={handle.pid})")

    gpu_policies = {
        name: GpuPolicy(gpu_policy) for name in pool_names if name.startswith("gpu.")
    }
    sampler = Sampler(
        registry, gpu_policies,
        own_pids=lambda: {h.pid for h in handles.values()},
        poll_interval=2.0,
    )
    sampler_task = asyncio.create_task(sampler.run())

    run_id = new_run_id("lang_screen.auto")
    queue: asyncio.Queue = asyncio.Queue()
    for batch in _batches(rows, batch_size):
        queue.put_nowait(batch)

    processed = 0
    n_pass = 0
    n_reject = 0
    n_mixed = 0
    errors = 0
    t0 = time.time()

    async def worker_loop(pool_name: str, handle) -> None:
        nonlocal processed, n_pass, n_reject, n_mixed, errors
        pool = registry.get(pool_name)
        while True:
            try:
                batch = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            window_starts_by_id: dict[str, list[float]] = {}
            items = []
            for raw_id, wav_path, duration_sec, sample_rate in batch:
                starts = compute_window_starts(duration_sec or 0.0)
                window_starts_by_id[raw_id] = starts
                items.append({
                    "raw_id": raw_id, "path": wav_path,
                    "sample_rate": int(sample_rate or 48000),
                    "window_starts": starts,
                })

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

            out_rows = []
            for r in result["rows"]:
                raw_id = r["raw_id"]
                agg = aggregate_decision(r.get("top_labels", []))
                if agg is None:
                    # Every window failed to decode/read — flag for human review
                    # rather than silently guessing pass or reject.
                    out_rows.append({
                        "raw_id": raw_id, "decision": None,
                        "cantonese_ratio_raw": None, "mandarin_ratio_raw": None,
                        "n_windows": 0, "window_starts": window_starts_by_id.get(raw_id, []),
                        "needs_review": True, "human_decision": None,
                        "reviewed_by": None, "reviewed_at": None,
                        "screened_at": _now_ts(), "provenance": "read_failed",
                    })
                    continue
                out_rows.append({
                    "raw_id": raw_id, "decision": agg["decision"],
                    "cantonese_ratio_raw": agg["cantonese_ratio_raw"],
                    "mandarin_ratio_raw": agg["mandarin_ratio_raw"],
                    "n_windows": agg["n_windows"],
                    "window_starts": window_starts_by_id.get(raw_id, []),
                    "needs_review": needs_review(raw_id, agg["decision"]),
                    "human_decision": None, "reviewed_by": None, "reviewed_at": None,
                    "screened_at": _now_ts(), "provenance": "lang_screen_auto",
                })
                if agg["decision"] == "pass":
                    n_pass += 1
                elif agg["decision"] == "reject":
                    n_reject += 1
                else:
                    n_mixed += 1

            # Raw files the worker couldn't even open at all (skipped_ids) —
            # same read_failed treatment so discover() never resurfaces them.
            skipped = result.get("skipped_ids", [])
            for raw_id in skipped:
                out_rows.append({
                    "raw_id": raw_id, "decision": None,
                    "cantonese_ratio_raw": None, "mandarin_ratio_raw": None,
                    "n_windows": 0, "window_starts": window_starts_by_id.get(raw_id, []),
                    "needs_review": True, "human_decision": None,
                    "reviewed_by": None, "reviewed_at": None,
                    "screened_at": _now_ts(), "provenance": "read_failed",
                })

            if out_rows:
                upsert_rows(conn, "lang_screen", out_rows, ["raw_id"])
                ok_ids = [r["raw_id"] for r in out_rows if r["provenance"] == "lang_screen_auto"]
                fail_ids = [r["raw_id"] for r in out_rows if r["provenance"] == "read_failed"]
                if ok_ids:
                    record_batch(conn, run_id, "lang_screen.auto", ok_ids, "ok",
                                 metrics=result.get("metrics"))
                if fail_ids:
                    record_batch(conn, run_id, "lang_screen.auto", fail_ids, "error",
                                 error="unreadable raw audio file")
                    log.warning(f"{pool_name}: {len(fail_ids)} unreadable/undecodable raw file(s): "
                                f"{fail_ids[:5]}")

            processed += len(out_rows)
            errors += len([r for r in out_rows if r["provenance"] == "read_failed"])
            queue.task_done()
            if processed and processed % (batch_size * 20) < batch_size:
                rate = processed / (time.time() - t0)
                log.info(f"{processed}/{len(rows)} screened ({rate:.1f}/s) — "
                         f"pass={n_pass} reject={n_reject} mixed={n_mixed} errors={errors}, "
                         f"pools={registry.snapshot()}")

    await asyncio.gather(*(
        worker_loop(pool_name, handles[pool_name]) for pool_name in pool_names
    ))

    sampler.stop()
    await asyncio.gather(sampler_task, return_exceptions=True)
    for handle in handles.values():
        await handle.shutdown()

    elapsed = time.time() - t0
    log.info(f"lang_screen.auto DONE: {processed} screened "
             f"(pass={n_pass}, reject={n_reject}, mixed={n_mixed}, errors={errors}) "
             f"in {elapsed:.0f}s, run_id={run_id}")
    return {
        "processed": processed, "pass": n_pass, "reject": n_reject,
        "mixed": n_mixed, "errors": errors, "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# GPU worker (subprocess side)
# ---------------------------------------------------------------------------

class LangScreenWorker(GPUWorkerBase):
    def load_model(self):
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        log.info(f"loading {LID_MODEL_ID} on {self.device} ...")
        self.fe = AutoFeatureExtractor.from_pretrained(LID_MODEL_ID)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(LID_MODEL_ID)
        model = model.to(self.device).eval()
        if self.use_fp16:
            model = model.half()
        self.id2label = model.config.id2label
        return model

    def _infer_windows(self, wavs: list[np.ndarray]) -> np.ndarray:
        """Run mms-lid-126 on a small chunk of windows in one forward pass.

        Callers must chunk *wavs* themselves (see INFER_CHUNK_WINDOWS in
        forward_batch) — each window here is WINDOW_SEC (25s) of 16 kHz audio,
        far longer than the ~3-20s clips label.suite's own _lid_infer() batches
        directly, so this must NOT receive an unbounded (batch_size * n_windows)
        list in one call the way that per-segment path does.
        """
        inp = self.fe(wavs, sampling_rate=SHARED_SR, return_tensors="pt", padding=True)
        inp = {k: (v.half() if (self.use_fp16 and v.is_floating_point()) else v).to(self.device)
               for k, v in inp.items()}
        with torch.no_grad():
            logits = self.model(**inp).logits
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()

    def forward_batch(self, items: list[dict]) -> list[dict]:
        """items: [{"raw_id": str, "windows": [np.ndarray, ...]}, ...] (already
        decoded to 16 kHz by the worker's I/O thread pool — see prep() below).
        Returns one {"raw_id", "top_labels": [str, ...]} per item, same order.

        A batch of raw files can contribute up to N_WINDOWS 25s windows EACH —
        e.g. batch_size=16 raw files x 10 windows = 160 windows, each far longer
        than a typical 3-20s segment clip. Feeding all of that through mms-lid-126
        in one forward pass (as label.suite's per-segment _lid_infer does, safely,
        for much shorter single clips) OOMs. INFER_CHUNK_WINDOWS caps the actual
        GPU forward-pass size independently of how many raw files/windows the
        orchestrator batched into this call — measured 2026-07-04: 0.25 mem_fraction
        OOM'd on an unchunked ~20-window batch even with ~6-13 GiB physically free
        (co-running canto-tts training job on the same GPU).
        """
        flat_wavs: list[np.ndarray] = []
        owner: list[int] = []
        for i, it in enumerate(items):
            for w in it["windows"]:
                flat_wavs.append(w)
                owner.append(i)

        top_labels_per_item: list[list[str]] = [[] for _ in items]
        for chunk_start in range(0, len(flat_wavs), INFER_CHUNK_WINDOWS):
            chunk = flat_wavs[chunk_start : chunk_start + INFER_CHUNK_WINDOWS]
            chunk_owner = owner[chunk_start : chunk_start + INFER_CHUNK_WINDOWS]
            probs = self._infer_windows(chunk)
            for k, p in enumerate(probs):
                arg = int(np.argmax(p))
                top_labels_per_item[chunk_owner[k]].append(self.id2label[arg])

        return [
            {"raw_id": items[i]["raw_id"], "top_labels": top_labels_per_item[i]}
            for i in range(len(items))
        ]


def _read_window(path: str, start_sec: float, dur_sec: float, native_sr: int) -> np.ndarray | None:
    """Partial-read one window from a raw file and resample to 16 kHz.

    Uses soundfile's start=/frames= to seek directly rather than decoding the
    whole (possibly multi-hour) raw file — the entire point of sampling a few
    windows instead of a full-file decode. Fail-soft: returns None on any error
    (short/truncated windows near EOF are kept as long as they're non-empty).
    """
    import soundfile as sf
    import soxr

    try:
        start_frame = int(start_sec * native_sr)
        n_frames = int(dur_sec * native_sr)
        y, sr = sf.read(path, start=start_frame, frames=n_frames,
                         dtype="float32", always_2d=False)
    except Exception as e:
        log.warning(f"lang_screen window read fail {path} @ {start_sec}s: {e}")
        return None

    if y is None or len(y) == 0:
        return None
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SHARED_SR:
        y = soxr.resample(y, sr, SHARED_SR, quality="HQ").astype(np.float32)
    return y


# ---------------------------------------------------------------------------
# Worker subprocess entrypoint — JSONL over stdio
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

    worker = LangScreenWorker(args.device, mem_fraction=args.mem_fraction, fp16=args.fp16)

    def emit(msg: dict) -> None:
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    emit({"type": "ready", "node": "lang_screen.auto", "pid": __import__("os").getpid(), "proto": 1})

    ex = ThreadPoolExecutor(max_workers=args.io_workers)

    def prep(it: dict) -> tuple[dict, list[np.ndarray]]:
        wavs = []
        for start in it["window_starts"]:
            w = _read_window(it["path"], start, WINDOW_SEC, it["sample_rate"])
            if w is not None and len(w) > 0:
                wavs.append(w)
        return it, wavs

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
            preps = list(ex.map(prep, items))
            kept_items = [{"raw_id": it["raw_id"], "windows": wavs}
                          for it, wavs in preps if wavs]
            skipped_ids = [it["raw_id"] for it, wavs in preps if not wavs]
            if not kept_items:
                emit({"type": "result", "task_id": task_id, "rows": [],
                      "skipped_ids": skipped_ids, "metrics": {"items_s": 0.0}})
                continue
            rows = worker.infer_with_oom_halving(kept_items)
            elapsed = time.time() - t0
            emit({"type": "result", "task_id": task_id, "rows": rows, "skipped_ids": skipped_ids,
                  "metrics": {"items_s": round(len(rows) / elapsed, 2) if elapsed > 0 else 0.0}})
        except Exception as e:
            emit({"type": "error", "task_id": task_id, "error": str(e), "retryable": True})

    ex.shutdown(wait=False)


if __name__ == "__main__":
    worker_main()
