"""
pipeline/nodes/asr.py
asr.transcribe DAG node — dual faster-whisper models (Cantonese fine-tune + large-v3
zh+prompt), ported from scripts/04_transcribe.py onto the orchestrator's GPUWorkerBase
+ JSONL worker protocol. asr.agreement is a separate CPU-only node that computes
cross-model char-overlap agreement once both models' asr_results rows exist for a
segment (item-level dependency — an id becomes eligible the moment its second model
result lands, no stage barrier).

Both models run split across GPUs from ONE supervisor process (run_asr_transcribe),
never as two separate `pipe run` invocations — DuckDB is single-writer (P2 backlog
finding, 2026-07-03): a second concurrent process's RW connect() would just block on
the first, so both devices are dispatched from inside the same asyncio run sharing one
catalog connection.

Hard constraint #7 (CLAUDE.md / KNOWN_ISSUES.md §9): NEVER language="yue" — both
models pass language="zh" with a Cantonese written-form prompt. large-v3 decoder
collapses into repetition loops under language="yue".

Golden-set parity note (REARCHITECTURE_IMPLEMENTATION_PLAN.md §9.1, updated 2026-07-03):
the 16 kHz downsample uses scipy.signal.resample_poly(wav, 1, 3) — identical to
scripts/04_transcribe.py's _load_and_resample — instead of audio/bus.py's soxr
resampler. 48000/16000 is an exact 3:1 ratio (lossless polyphase, no approximation),
and this node is the only reader of its audio (no decode-once fan-out benefit to
sharing bus.py here), so there is no reason to introduce a different resampler that
would risk perturbing greedy-decoded (beam_size=1) ASR text at the token boundary.

ASR text is NOT expected to match the legacy snapshot byte-for-byte, even with
identical uv.lock-pinned ctranslate2/faster-whisper versions and identical audio —
confirmed both are unchanged since before the corpus was originally transcribed, yet
output still differs, stably (not randomly: repeat calls, same or fresh CUDA context,
are 100% reproducible). The likely remaining variable is GPU-driver-level numerical
drift, outside this package's control. Golden parity for this node is therefore
similarity-tolerance (char_agreement), not exact-match — see
tests/golden/check_asr_parity.py and REARCHITECTURE_IMPLEMENTATION_PLAN.md §9.1.
"""

import argparse
import asyncio
import difflib
import itertools
import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

from pipeline.config import REPO_ROOT
from pipeline.workers.gpu_base import GPUWorkerBase

log = logging.getLogger(__name__)

TARGET_SR = 48000
ASR_SR = 16000

# Cantonese written-form initial prompt (helps large-v3 produce 粵語白話文) — verbatim
# copy of scripts/04_transcribe.py's CANTO_PROMPT for golden-set parity.
CANTO_PROMPT = (
    "以下係廣東話口語，請用粵語白話文書寫，"
    "例如：係、唔係、冇、喺、佢哋、嘅、嗰、嚟。"
)

_LOCAL_CANTO = str(REPO_ROOT / "data" / "ct2_models" / "whisper-large-v2-cantonese")

# model_key (CLI/internal selector) -> cfg. cfg["id"] must stay byte-identical to
# scripts/04_transcribe.py's ASR_MODELS so the stored `asr_results.model` field
# (id + "+" + lang) matches the 910,598 rows already imported from the legacy
# manifest — new rows land in the same (id, model) primary-key space.
ASR_MODELS = {
    "canto_ft": {
        "id": _LOCAL_CANTO,
        "lang": "zh",
        "prompt": CANTO_PROMPT,
        "description": "Cantonese fine-tuned Whisper large-v2 (local ct2)",
    },
    "whisper_v3": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "zh",
        "prompt": CANTO_PROMPT,
        "description": "Whisper large-v3 with zh + Cantonese prompt",
    },
}


def model_field(model_key: str) -> str:
    """The exact string stored in asr_results.model — id + '+' + lang, matching
    scripts/04_transcribe.py's transcribe_one()."""
    cfg = ASR_MODELS[model_key]
    return cfg["id"] + (f"+{cfg['lang']}" if cfg["lang"] else "")


# ---------------------------------------------------------------------------
# Audio loading (parity-preserving — see module docstring)
# ---------------------------------------------------------------------------

def _load_and_resample(wav_path: str) -> np.ndarray | None:
    """Load a 48 kHz WAV and downsample to 16 kHz for ASR. Thread-safe (scipy only,
    no torch/CUDA) — identical to scripts/04_transcribe.py's _load_and_resample."""
    from scipy.signal import resample_poly
    try:
        wav48, _ = sf.read(wav_path, dtype="float32", always_2d=False)
    except Exception as e:
        log.warning(f"read fail {wav_path}: {e}")
        return None
    if wav48.ndim > 1:
        wav48 = wav48.mean(axis=1)
    # 48000 -> 16000 is exactly x(1/3); resample_poly is lossless for integer ratios.
    return resample_poly(wav48, 1, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Catalog discovery (supervisor side)
# ---------------------------------------------------------------------------

TRANSCRIBE_DISCOVER_SQL = """
    SELECT s.id, s.audio_path, s.duration_sec
    FROM segments s
    LEFT JOIN asr_results a ON s.id = a.id AND a.model = ?
    WHERE a.id IS NULL
    ORDER BY s.duration_sec
"""


def discover_transcribe(conn, model_key: str) -> list[tuple]:
    return conn.execute(TRANSCRIBE_DISCOVER_SQL, [model_field(model_key)]).fetchall()


# Assumes exactly 2 ASR models feed asr_results (ASR_MODELS today). The self-join
# on a1.model < a2.model pairs each id with its one counterpart; if a 3rd model is
# ever added, each id would yield 3 pairs here (non-deterministic last-write-wins
# on the asr_agreement upsert) — revisit with a GROUP BY id + list_agg() instead.
AGREEMENT_DISCOVER_SQL = """
    SELECT a1.id, a1.text, a1.confidence, a2.text, a2.confidence
    FROM asr_results a1
    JOIN asr_results a2 ON a1.id = a2.id AND a1.model < a2.model
    LEFT JOIN asr_agreement ag ON a1.id = ag.id
    WHERE ag.id IS NULL
"""


def discover_agreement(conn) -> list[tuple]:
    return conn.execute(AGREEMENT_DISCOVER_SQL).fetchall()


# ---------------------------------------------------------------------------
# Agreement computation (pure logic — verbatim port of scripts/04_transcribe.py's
# char_agreement() + write_transcripts()'s best-candidate / agreement-gating rules)
# ---------------------------------------------------------------------------

def char_agreement(texts: list[str]) -> float:
    if len(texts) < 2:
        return 1.0
    ratios = [
        difflib.SequenceMatcher(None, a, b).ratio()
        for a, b in itertools.combinations(texts, 2)
    ]
    return round(sum(ratios) / len(ratios), 3)


def compute_agreement_row(seg_id: str, t1: str, c1: float, t2: str, c2: float) -> dict:
    """Mirrors write_transcripts(): agreement is 0.0 unless >=2 non-empty candidate
    texts exist; best_text is the non-empty candidate with the higher confidence."""
    candidates = [{"text": t1 or "", "confidence": c1 or 0.0},
                  {"text": t2 or "", "confidence": c2 or 0.0}]
    texts = [c["text"] for c in candidates if c["text"]]
    agreement = char_agreement(texts) if len(texts) >= 2 else 0.0
    best = max(candidates, key=lambda c: c["confidence"] if c["text"] else -1)
    return {
        "id": seg_id,
        "agreement": agreement,
        "best_text": best["text"],
        "text_verified": False,
    }


# ---------------------------------------------------------------------------
# Supervisor: asr.transcribe — one worker per (model_key, device) assignment,
# dispatched concurrently within a single asyncio run / single DB connection.
# ---------------------------------------------------------------------------

def _batches(rows: list[tuple], size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


async def run_asr_transcribe(
    assignments: list[tuple[str, str]],
    *,
    gpu_policy: str = "cap",
    batch_size: int = 8,
    mem_fraction: float | None = None,
    limit: int | None = None,
) -> dict:
    """Supervisor entrypoint for the asr.transcribe node.

    assignments: list of (model_key, device), e.g.
        [("canto_ft", "cuda:0"), ("whisper_v3", "cuda:1")]
    Each assignment gets its own discover() query (different model = different
    anti-join target), own dispatch queue, and own worker subprocess — but all
    run under ONE asyncio.gather() sharing a single DuckDB connection, so this
    is the "two models split across GPUs, running in parallel" node the plan
    calls for, without ever opening two competing RW connections to the catalog.
    """
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch
    from pipeline.orchestrator.pools import PoolRegistry
    from pipeline.orchestrator.resources import GpuPolicy, Sampler
    from pipeline.orchestrator.worker import spawn_worker

    conn = connect()

    per_assignment_rows: dict[tuple[str, str], list[tuple]] = {}
    total = 0
    for model_key, device in assignments:
        rows = discover_transcribe(conn, model_key)
        if limit:
            rows = rows[:limit]
        per_assignment_rows[(model_key, device)] = rows
        total += len(rows)
        log.info(f"asr.transcribe[{model_key}]: {len(rows)} segments to transcribe on {device}")

    if total == 0:
        return {"processed": 0, "errors": 0}

    registry = PoolRegistry()
    registered: set[str] = set()
    pool_names: dict[tuple[str, str], str] = {}
    for model_key, device in assignments:
        pool_name = f"gpu.{device.split(':')[1]}" if device.startswith("cuda") else "cpu"
        if pool_name not in registered:
            registry.register(pool_name, target=1)
            registered.add(pool_name)
        pool_names[(model_key, device)] = pool_name

    handles = {}
    for model_key, device in assignments:
        cmd = [
            sys.executable, "-m", "pipeline.nodes.asr",
            "--device", device, "--model-key", model_key,
        ]
        if mem_fraction is not None and device.startswith("cuda"):
            cmd += ["--mem-fraction", str(mem_fraction)]
        handle = await spawn_worker(cmd)
        await handle.wait_ready(timeout=180.0)
        handles[(model_key, device)] = handle
        log.info(f"worker ready: {model_key} -> {device} (pid={handle.pid})")

    gpu_policies = {
        name: GpuPolicy(gpu_policy)
        for name in set(pool_names.values()) if name.startswith("gpu.")
    }
    sampler = Sampler(
        registry, gpu_policies,
        own_pids=lambda: {h.pid for h in handles.values()},
        poll_interval=2.0,
    )
    sampler_task = asyncio.create_task(sampler.run())

    run_id = new_run_id("asr.transcribe")
    processed = 0
    errors = 0
    t0 = time.time()

    async def worker_loop(model_key: str, device: str) -> None:
        nonlocal processed, errors
        handle = handles[(model_key, device)]
        pool = registry.get(pool_names[(model_key, device)])
        mf = model_field(model_key)
        queue: asyncio.Queue = asyncio.Queue()
        for batch in _batches(per_assignment_rows[(model_key, device)], batch_size):
            queue.put_nowait(batch)

        while True:
            try:
                batch = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            meta = {r[0]: r[2] for r in batch}  # id -> duration_sec (unused, kept for parity/logs)
            items = [{"id": r[0], "path": r[1]} for r in batch]
            async with pool.acquire():
                await handle.send_task(f"{model_key}-{processed}", items)
                try:
                    result = await handle.read_message(timeout=600.0)
                except Exception as e:
                    log.error(f"{model_key}@{device}: batch failed: {e}")
                    errors += len(batch)
                    queue.task_done()
                    continue
            if result["type"] == "error":
                log.error(f"{model_key}@{device}: worker error: {result['error']}")
                errors += len(batch)
                queue.task_done()
                continue

            out_rows = [
                {"id": r["id"], "model": mf, "text": r["text"], "confidence": r["confidence"]}
                for r in result["rows"]
            ]
            # Unreadable audio still gets a placeholder row (empty text, confidence
            # 0.0) so discover()'s anti-join stops resurfacing the same dead id —
            # mirrors label_music.py's skipped_ids handling.
            skipped_rows = [
                {"id": sid, "model": mf, "text": "", "confidence": 0.0}
                for sid in result.get("skipped_ids", [])
            ]
            if skipped_rows:
                log.warning(f"{model_key}@{device}: {len(skipped_rows)} unreadable segment(s): "
                            f"{[r['id'] for r in skipped_rows][:5]}")

            upsert_rows(conn, "asr_results", out_rows + skipped_rows, ["id", "model"])
            record_batch(conn, run_id, "asr.transcribe", [r["id"] for r in out_rows], "ok",
                         metrics=result.get("metrics"))
            if skipped_rows:
                record_batch(conn, run_id, "asr.transcribe", [r["id"] for r in skipped_rows],
                             "error", error="unreadable audio file")

            processed += len(out_rows) + len(skipped_rows)
            errors += len(skipped_rows)
            queue.task_done()
            if processed and processed % (batch_size * 20) < batch_size:
                rate = processed / (time.time() - t0)
                log.info(f"{processed}/{total} processed ({rate:.1f}/s), pools={registry.snapshot()}")

    await asyncio.gather(*(
        worker_loop(model_key, device) for model_key, device in assignments
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
# Supervisor: asr.agreement — CPU-only, pure Python (difflib), no GPU/subprocess
# workers needed. Runs directly in the supervisor process.
# ---------------------------------------------------------------------------

async def run_asr_agreement(*, batch_size: int = 2000, limit: int | None = None) -> dict:
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = connect()
    rows = discover_agreement(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"asr.agreement: {len(rows)} segments to score")
    if not rows:
        return {"processed": 0, "errors": 0}

    run_id = new_run_id("asr.agreement")
    processed = 0
    t0 = time.time()

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        out_rows = [compute_agreement_row(*r) for r in batch]
        upsert_rows(conn, "asr_agreement", out_rows, ["id"])
        record_batch(conn, run_id, "asr.agreement", [r["id"] for r in out_rows], "ok")
        processed += len(out_rows)
        rate = processed / (time.time() - t0) if time.time() > t0 else 0.0
        log.info(f"{processed}/{len(rows)} scored ({rate:.1f}/s)")

    elapsed = time.time() - t0
    log.info(f"DONE: {processed} scored in {elapsed:.0f}s "
             f"({processed / elapsed if elapsed > 0 else 0:.1f}/s), run_id={run_id}")
    return {"processed": processed, "errors": 0, "run_id": run_id}


# ---------------------------------------------------------------------------
# GPU worker (subprocess side) — asr.transcribe only; asr.agreement has no worker.
# ---------------------------------------------------------------------------

class TranscribeWorker(GPUWorkerBase):
    # NOTE: GPUWorkerBase's mem_fraction cap calls torch.cuda.set_per_process_memory_
    # fraction(), which only constrains PyTorch's own CUDA caching allocator.
    # faster-whisper's ctranslate2 backend allocates CUDA memory through its own
    # independent pool, entirely outside torch's allocator — so --mem-fraction is a
    # silent no-op for this node. Kept for CLI/interface consistency with the other
    # GPU nodes (label.music/label.suite, which ARE torch-backed); real VRAM control
    # for this node is ctranslate2's own compute_type choice (int8_float16 below).
    def __init__(self, device: str, model_key: str, *, mem_fraction: float | None = None,
                 fp16: bool = True) -> None:
        self.model_key = model_key
        self.cfg = ASR_MODELS[model_key]
        super().__init__(device, mem_fraction=mem_fraction, fp16=fp16)

    def load_model(self):
        from faster_whisper import WhisperModel
        is_cuda = str(self.device).startswith("cuda")
        compute_type = "int8_float16" if is_cuda else "int8"
        gpu_index = int(self.device.split(":")[1]) if is_cuda and ":" in str(self.device) else 0
        device_kind = "cuda" if is_cuda else "cpu"
        log.info(f"Loading ASR model: {self.cfg['description']} on {self.device} [{compute_type}]")
        return WhisperModel(
            self.cfg["id"],
            device=device_kind,
            device_index=gpu_index,
            compute_type=compute_type,
            cpu_threads=4,
        )

    def forward_batch(self, items: list[np.ndarray]) -> list[dict]:
        # faster-whisper has no native batched-tensor API here (BatchedInferencePipeline
        # is a separate opt-in path not used by the legacy script) — sequential per-item
        # decode, matching scripts/04_transcribe.py's transcribe_one() exactly.
        return [self._transcribe_one(y16) for y16 in items]

    def _transcribe_one(self, y16: np.ndarray) -> dict:
        kwargs: dict = {
            "beam_size": 1,   # greedy — matches legacy exactly for golden parity
            "vad_filter": False,
            "temperature": 0.0,
        }
        if self.cfg["lang"]:
            kwargs["language"] = self.cfg["lang"]  # NEVER "yue" — hard constraint #7
        if self.cfg["prompt"]:
            kwargs["initial_prompt"] = self.cfg["prompt"]

        segments, info = self.model.transcribe(y16, **kwargs)
        segs_list = list(segments)
        text = "".join(s.text for s in segs_list).strip()
        if segs_list:
            conf = float(np.mean([s.avg_logprob for s in segs_list if hasattr(s, "avg_logprob")]))
            conf = round(max(0.0, min(1.0, math.exp(conf))), 3)
        else:
            conf = 0.0
        return {"text": text, "confidence": conf}


def worker_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-key", required=True, choices=list(ASR_MODELS.keys()))
    ap.add_argument("--mem-fraction", type=float, default=None)
    ap.add_argument("--io-workers", type=int, default=8)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                         format="%(asctime)s %(levelname)s %(message)s")

    worker = TranscribeWorker(args.device, args.model_key, mem_fraction=args.mem_fraction)

    def emit(msg: dict) -> None:
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    emit({"type": "ready", "node": "asr.transcribe", "pid": __import__("os").getpid(), "proto": 1})

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
            wavs = list(ex.map(_load_and_resample, paths))
            keep_idx = [i for i, w in enumerate(wavs) if w is not None and len(w) >= ASR_SR // 10]
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
