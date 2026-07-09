"""
pipeline/nodes/segment.py
=========================
Three DAG nodes for the Cantonese speech-corpus pipeline:

  segment.diarize  — ensures every ``raw_files`` row that needs segmentation
                     either (a) reuses an existing legacy sidecar
                     ``{segments_root}/{source}/{stem}_segments.jsonl``, or (b)
                     runs pyannote/speaker-diarization-3.1 and writes the
                     resulting speaker turns into ``diarization_turns``.

  segment.vad_cut  — for every raw_id that has ``diarization_turns`` rows but
                     no ``raw_segments`` row yet, runs Silero VAD within each
                     turn, cuts 48 kHz clips to disk, and writes both
                     ``segments`` rows and a ``raw_segments`` completion record.

  pregate.snr      — fast SNR + DNSMOS pre-gate on every pipeline-cut segment
                     (i.e. segments with ``raw_id IS NOT NULL``) not yet in
                     ``pregate``.  Writes one ``pregate`` row per segment.

-------------------------------------------------------------------------------
Design decisions and rationale
-------------------------------------------------------------------------------

(a) Hybrid reuse-first design for segment.diarize
    A survey of the live catalog (6 272 raw_files rows) found that every single
    row already has a matching legacy sidecar:
      ``{segments_root}/{source}/{stem}_segments.jsonl``
    written by scripts/03_segment.py's ``segment_file()`` as its own
    idempotency marker (one JSON line per cut clip).  Running a full GPU
    diarization pass over all 6 272 files would waste 10-20 GPU-hours AND
    produce duplicate WAV files under new segment ids.

    Instead we use the same three-phase pattern as speaker.embed
    (pipeline/nodes/speaker.py design decision (a)):

      1. Discovery (SQL anti-join): raw_files not yet in raw_segments AND not
         yet in diarization_turns (LEFT JOIN both, WHERE both IS NULL).
      2. Cheap reuse pass (CPU, thread pool): for every discovered raw file,
         check whether the legacy sidecar exists and has >0 lines.  On a hit,
         upsert immediately into raw_segments with provenance='legacy_reused',
         n_segments=<line count>, segmented_at=now.  No diarization_turns rows
         are written for these — the raw file is fully done; its clips already
         exist wherever they exist.
      3. GPU fallback (DiarizeWorker subprocess pool): only for the genuine
         cache-miss remainder.  If the missing list is empty after Phase 2, all
         subprocess spawning is skipped entirely.

    Discovery is scoped to the (raw_segments × diarization_turns) double
    anti-join so that a file that partially succeeded (turns written but no
    raw_segments row yet) is correctly not re-diarized — it will be picked up
    by segment.vad_cut's own discovery instead.

(b) Two separate tables: diarization_turns and raw_segments
    Splitting turn storage from the completion record follows the same
    upsert-clobbering rationale as speaker.py's design decision (b): if both
    lived in one table, re-running segment.diarize would clobber the
    n_segments/provenance columns written by segment.vad_cut, and vice-versa.
    segment.diarize is the sole writer of diarization_turns; segment.vad_cut is
    the sole writer of raw_segments for provenance='segment_vad_cut' rows
    (segment.diarize writes raw_segments only for the 'legacy_reused' and
    'diarize_failed' provenance values — those three provenances are mutually
    exclusive by construction, so INSERT OR REPLACE never clobbers a column the
    other node owns).

    This split also enables item-level pipelining between the two DAG nodes
    with no barrier: the moment segment.diarize finishes turns for one raw_id,
    segment.vad_cut can start cutting clips for that file while diarization
    continues on others — exactly the same no-barrier design filter.py uses
    between filter.text and filter.acoustic.

(c) segment.vad_cut runs in-supervisor via asyncio.to_thread / ThreadPoolExecutor
    rather than spawning subprocess workers (contrast filter.acoustic which does
    spawn subprocesses).

    Rationale: Silero VAD runs on CPU, is fast (well under 1× realtime even on
    one thread), and the bottleneck is I/O (reading 48 kHz WAV masters from
    Drive2) not compute.  A ThreadPoolExecutor of per-file tasks provides all
    the parallelism needed without the subprocess-JSONL-protocol overhead.
    There is no GPU to protect and no need for OOM-halving logic.

    filter.acoustic spawns subprocess workers because (a) ONNX RT DNSMOS is
    computationally expensive relative to the audio read, and (b) the
    intra-op thread-pool cap trick requires separate OS processes to be
    effective (monkeypatching ort.InferenceSession across threads is not safe).
    Neither concern applies here: Silero VAD uses TorchScript under the hood
    and the GIL is released during inference, so a plain thread pool is
    both correct and sufficient.

    CORRECTION (found via a real production run, 2026-07-03): a shared
    module-level singleton model, called concurrently from multiple worker
    threads, segfaulted — Silero VAD's TorchScript module carries internal
    mutable state that is NOT safe for concurrent inference calls, contrary
    to what an earlier draft of this docstring claimed. The GIL is released
    during the C++/TorchScript inference call itself, which is exactly what
    allows two threads to be inside ``get_speech_timestamps()`` on the same
    model object at once and corrupt its internal state. Fix: each worker
    thread gets its OWN lazily-loaded model+utils via ``threading.local()``
    (see ``_get_vad_model()``) — no two threads ever touch the same model
    instance. The model is tiny (a few MB), so one instance per thread is
    cheap; this is not the same tradeoff as GPU model loading.

(d) pregate.snr's SNR formula deliberately does NOT match filter.py's compute_snr
    filter.py's ``compute_snr()`` (ported verbatim from scripts/06_filter.py)
    uses non-overlapping 25 ms frames, sorts them, and computes
    ``10 * log10(mean(top-10%) / mean(bottom-10%))``.

    03b_acoustic_pregate.py's ``compute_snr()`` (ported here as
    ``compute_pregate_snr()``) uses overlapping frames with a 10 ms hop,
    computes per-frame RMS energy, and uses numpy.percentile(90) /
    numpy.percentile(10) — a fundamentally different algorithm with different
    numerical output.

    These are deliberately two different gates:
      • pregate.snr  — fast early-reject before ASR (cheap, loose, acceptable
                       false-positive rate; purpose is to skip obvious noise
                       before paying for GPU transcription).
      • filter.acoustic — authoritative final gate (tight, held to the
                          ±1e-4 golden-parity tolerance).

    They are not required to agree numerically and must not be merged.  Using
    filter.py's formula in pregate.snr would change the effective threshold
    semantics and violate the "port core logic close to verbatim" requirement
    in the task brief.

(e) Resampler consistency — same reasoning as speaker.py and filter.py
    The DiarizeWorker resamples from 48 kHz to 16 kHz using
    ``torchaudio.transforms.Resample`` — the same method scripts/03_segment.py
    used in ``audio_to_16k()`` — rather than soxr (bus.py's resampler).
    segment.vad_cut applies the same torchaudio Resample for the transient 16 kHz
    VAD copy, matching the legacy script exactly.  Using a different resampler
    (soxr) for the pyannote / Silero VAD path would produce numerically different
    diarization and VAD boundary decisions and violate the golden-parity
    reasoning documented in the other GPU nodes' docstrings.

    For the 48 kHz master read (segment.vad_cut and pregate.snr), bus.py's
    ``decode(path, 48000)`` is used — this hits the zero-cost passthrough branch
    (``orig_sr == target_sr``) for the already-48 kHz masters, matching
    filter.py's documented rationale identically.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from pipeline.workers.gpu_base import GPUWorkerBase

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root

# ---------------------------------------------------------------------------
# Constants — copied verbatim from scripts/03_segment.py and
# scripts/03b_acoustic_pregate.py so thresholds are byte-identical.
# ---------------------------------------------------------------------------

TARGET_SR = 48000     # master WAV sample rate (48 kHz)
VAD_SR = 16000        # Silero VAD + pyannote operate at 16 kHz
MIN_DUR = 3.0         # minimum valid segment duration (seconds)
MAX_DUR = 20.0        # maximum valid segment duration (seconds)

DEFAULT_MIN_SNR = 25.0     # 03b pregate SNR threshold (dB)
DEFAULT_MIN_DNSMOS = 3.0   # 03b pregate DNSMOS ovrl_mos threshold

# Segments root — from config/storage_layout.yaml: /mnt/Drive4/canto/segments
# Per-source dirs: {SEGMENTS_ROOT}/{source}/
import yaml as _yaml
_STORAGE = _yaml.safe_load(
    (ROOT / "config" / "storage_layout.yaml").read_text(encoding="utf-8")
)["storage"]
SEGMENTS_ROOT = Path(_STORAGE["segments_root"])

# P5-C (2026-07-06): when sharding is enabled, new segments are written straight
# onto their final shard (hash(raw_id) % n_shards — raw_id is always present
# here, this function's own parameter) instead of always landing on Drive4 and
# needing a later rebalance pass. See config/storage_layout.py shard_root() /
# pipeline/nodes/rebalance.py (one-time migration of pre-P5-C segments).
from config.storage_layout import SHARDING as _SHARDING
from config.storage_layout import shard_root as _shard_root


def _segments_out_dir(raw_id: str, source: str) -> Path:
    root = _shard_root(raw_id) if _SHARDING.get("enabled") else SEGMENTS_ROOT
    return root / source

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _batches(rows: list, size: int):
    """Yield successive fixed-size slices of *rows*."""
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _now_ts() -> str:
    """ISO-8601 timestamp for segmented_at / created_at fields."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _segment_id(seg_path: Path) -> str:
    """Stable 12-hex segment id — verbatim convention from docs/MANIFEST_SCHEMA.md."""
    return hashlib.md5(str(seg_path.resolve()).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Silero VAD model — one instance PER THREAD (thread-local), not a shared
# singleton.
#
# scripts/03_segment.py used a true global singleton, but that script never
# actually exercised concurrent VAD calls (--workers defaults to 1, one file
# processed at a time). segment.vad_cut runs a ThreadPoolExecutor with
# multiple worker threads, and a real production run (2026-07-03) segfaulted
# with 3 threads simultaneously inside the same loaded TorchScript module's
# get_speech_timestamps() (confirmed via faulthandler traceback — all 3
# crashing threads shared one frame calling the same model instance). Silero
# VAD's TorchScript module carries internal mutable state that is NOT safe
# for concurrent calls from multiple threads, despite the model being
# advertised as CPU-friendly/fast — that claim is about single-threaded
# throughput, not thread-safety. Fix: each worker thread lazily loads and
# keeps its own model+utils in threading.local() storage, so no two threads
# ever call into the same model object. The one-time main-thread pre-load in
# run_segment_vad_cut() still runs first to warm the local torch.hub repo
# cache before any threads start, avoiding a first-load race on the cache dir
# itself; each thread's own first _get_vad_model() call is then a cheap local
# re-load from that already-warm cache, not a fresh download/verify.
# ---------------------------------------------------------------------------

import threading as _threading

_vad_tls = _threading.local()


def _get_vad_model():
    if not hasattr(_vad_tls, "model"):
        log.info(f"Loading Silero VAD (thread {_threading.get_ident()}) ...")
        model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        # Keep VAD on CPU — Silero VAD is fast enough on CPU and avoids
        # device mismatch issues when passing numpy-derived tensors.
        _vad_tls.model = model
        _vad_tls.utils = utils
    return _vad_tls.model, _vad_tls.utils


# ---------------------------------------------------------------------------
# Audio helpers — verbatim ports from scripts/03_segment.py
# ---------------------------------------------------------------------------

def _audio_to_16k(wav48: np.ndarray) -> np.ndarray:
    """Downsample from 48 kHz to 16 kHz (in-memory, transient).

    Uses torchaudio.transforms.Resample — intentionally NOT soxr — to match the
    exact resampler used by scripts/03_segment.py's ``audio_to_16k()``.  See
    design decision (e) in the module docstring.
    """
    import torchaudio
    t = torch.from_numpy(wav48).float().unsqueeze(0)
    resampler = torchaudio.transforms.Resample(TARGET_SR, VAD_SR)
    return resampler(t).squeeze(0).numpy()


def _get_vad_segments_in_window(
    wav16: np.ndarray,
    window_start: float,
    window_end: float,
    chunk_sec: float = 60.0,
) -> list[tuple[float, float]]:
    """Run Silero VAD on a mono 16 kHz array; return (start, end) in seconds.

    Ported verbatim from scripts/03_segment.py:get_vad_segments_in_window().
    Processes in chunks of ``chunk_sec`` to avoid TorchScript memory issues
    with very long audio (Silero VAD can fail on tensors > a few minutes).
    """
    model, utils = _get_vad_model()
    get_speech_timestamps = utils[0]

    start_sample = int(window_start * VAD_SR)
    end_sample = int(window_end * VAD_SR)
    chunk_samples = int(chunk_sec * VAD_SR)

    all_timestamps: list[tuple[float, float]] = []
    cursor = start_sample

    while cursor < end_sample:
        seg_end = min(cursor + chunk_samples, end_sample)
        chunk = wav16[cursor:seg_end]

        if len(chunk) < int(MIN_DUR * VAD_SR):
            cursor = seg_end
            continue

        tensor = torch.from_numpy(chunk).float()
        try:
            timestamps = get_speech_timestamps(
                tensor, model, sampling_rate=VAD_SR,
                threshold=0.5, min_silence_duration_ms=300,
                min_speech_duration_ms=int(MIN_DUR * 1000),
            )
        except Exception as exc:
            log.warning(f"VAD chunk failed (offset {cursor / VAD_SR:.1f}s): {exc}")
            cursor = seg_end
            continue

        chunk_offset = cursor / VAD_SR
        for t in timestamps:
            all_timestamps.append((
                chunk_offset + t["start"] / VAD_SR,
                chunk_offset + t["end"] / VAD_SR,
            ))
        cursor = seg_end

    return all_timestamps


# ===========================================================================
# segment.diarize
# ===========================================================================

DIARIZE_DISCOVER_SQL = """
    SELECT rf.raw_id, rf.wav_path, rf.source
    FROM raw_files rf
    LEFT JOIN raw_segments rs ON rf.raw_id = rs.raw_id
    LEFT JOIN diarization_turns dt ON rf.raw_id = dt.raw_id
    LEFT JOIN lang_screen ls ON rf.raw_id = ls.raw_id
    WHERE rs.raw_id IS NULL
      AND dt.raw_id IS NULL
      AND COALESCE(ls.human_decision, ls.decision, 'pass') != 'reject'
    ORDER BY rf.raw_id
"""


def discover_diarize(conn) -> list[tuple]:
    """Return (raw_id, wav_path, source) for raw_files not yet in raw_segments
    AND not yet in diarization_turns.

    Also excludes any raw_id whose EFFECTIVE pipeline.nodes.lang_screen decision
    (COALESCE(human_decision, decision, 'pass')) is 'reject' — a raw file with no
    lang_screen row at all (never screened, or predates that node) defaults to
    'pass' here, so this join is purely additive and never retroactively blocks
    already-in-flight or legacy raw files."""
    return conn.execute(DIARIZE_DISCOVER_SQL).fetchall()


def _check_legacy_sidecar(row: tuple) -> tuple[str, str, str, int]:
    """I/O task: check whether the legacy _segments.jsonl sidecar exists and
    has >0 lines.

    Returns (raw_id, wav_path, source, n_lines) where n_lines=0 means no
    sidecar or empty sidecar (cache miss).

    Designed to be called inside a ThreadPoolExecutor (GIL-released Path.stat
    + file open).
    """
    raw_id, wav_path, source = row
    stem = Path(wav_path).stem
    sidecar = SEGMENTS_ROOT / source / f"{stem}_segments.jsonl"
    if not sidecar.exists():
        return (raw_id, wav_path, source, 0)
    try:
        n = sum(1 for _ in open(sidecar, "r", encoding="utf-8"))
    except Exception:
        n = 0
    return (raw_id, wav_path, source, n)


async def run_segment_diarize(
    devices: list[str],
    *,
    conn=None,
    gpu_policy: str = "cap",
    batch_size: int = 32,
    mem_fraction: float | None = 0.5,
    io_workers: int | None = None,
    limit: int | None = None,
) -> dict:
    """Supervisor coroutine for the segment.diarize DAG node.

    Phase 1 — discovery (SQL anti-join on raw_segments × diarization_turns).
    Phase 2 — cheap reuse pass: parallel sidecar existence checks for every
               discovered raw file; on hit (sidecar exists, >0 lines), upsert
               immediately into raw_segments with provenance='legacy_reused'.
               No diarization_turns rows are written for reused files.
    Phase 3 — GPU fallback: only for the genuine cache-miss remainder, spawn
               one DiarizeWorker subprocess per device.  Skipped entirely when
               the missing list is empty after Phase 2.

    conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the same rationale). Defaults to a
    fresh self-managed connect() for standalone `pipe run segment.diarize`.
    """
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch
    from pipeline.orchestrator.pools import PoolRegistry
    from pipeline.orchestrator.resources import GpuPolicy, Sampler
    from pipeline.orchestrator.worker import spawn_worker

    conn = conn or connect()
    rows = discover_diarize(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"segment.diarize: {len(rows)} raw_files need diarization or reuse check")
    if not rows:
        return {"reused": 0, "gpu_computed": 0, "errors": 0}

    run_id = new_run_id("segment.diarize")
    t0 = time.time()

    # -----------------------------------------------------------------------
    # Phase 2 — cheap reuse pass (thread pool, I/O-bound)
    # -----------------------------------------------------------------------
    n_io = io_workers or min(32, (os.cpu_count() or 4) * 4)
    log.info(
        f"segment.diarize: checking {len(rows)} legacy _segments.jsonl sidecars "
        f"with {n_io} I/O threads ..."
    )

    reuse_rows: list[dict] = []
    missing: list[tuple] = []  # (raw_id, wav_path, source) with no sidecar

    with ThreadPoolExecutor(max_workers=n_io) as pool:
        for batch in _batches(rows, batch_size * 100):
            results = list(pool.map(_check_legacy_sidecar, batch))
            hits = [(raw_id, wav_path, source, n) for raw_id, wav_path, source, n in results if n > 0]
            misses = [
                (raw_id, wav_path, source)
                for raw_id, wav_path, source, n in results
                if n == 0
            ]
            missing.extend(misses)
            if hits:
                upsert_batch = [
                    {
                        "raw_id": raw_id,
                        "n_segments": n,
                        "provenance": "legacy_reused",
                        "segmented_at": _now_ts(),
                    }
                    for raw_id, wav_path, source, n in hits
                ]
                upsert_rows(conn, "raw_segments", upsert_batch, ["raw_id"])
                record_batch(
                    conn, run_id, "segment.diarize",
                    [r["raw_id"] for r in upsert_batch], "ok",
                )
                reuse_rows.extend(upsert_batch)
            log.info(
                f"segment.diarize reuse pass: {len(reuse_rows)}/{len(rows)} reused, "
                f"{len(missing)} still missing"
            )

    log.info(
        f"segment.diarize: reuse pass complete — "
        f"{len(reuse_rows)} legacy-reused, {len(missing)} need GPU diarization"
    )

    # -----------------------------------------------------------------------
    # Phase 3 — GPU fallback (only if there are remaining rows)
    # -----------------------------------------------------------------------
    gpu_computed = 0
    errors = 0

    if missing:
        log.info(f"segment.diarize: spawning GPU worker(s) for {len(missing)} files ...")

        registry = PoolRegistry()
        pool_names = []
        for dev in devices:
            pool_name = f"gpu.{dev.split(':')[1]}" if dev.startswith("cuda") else "cpu"
            registry.register(pool_name, target=1)
            pool_names.append(pool_name)

        handles = {}
        for dev, pool_name in zip(devices, pool_names):
            cmd = [
                sys.executable, "-m", "pipeline.nodes.segment",
                "--node", "diarize",
                "--device", dev,
            ]
            if mem_fraction is not None and dev.startswith("cuda"):
                cmd += ["--mem-fraction", str(mem_fraction)]
            handle = await spawn_worker(cmd)
            await handle.wait_ready(timeout=300.0)
            handles[pool_name] = handle
            log.info(f"segment.diarize worker ready: {pool_name} -> {dev} (pid={handle.pid})")

        gpu_policies = {
            name: GpuPolicy(gpu_policy) for name in pool_names if name.startswith("gpu.")
        }
        sampler = Sampler(
            registry, gpu_policies,
            own_pids=lambda: {h.pid for h in handles.values()},
            poll_interval=2.0,
        )
        sampler_task = asyncio.create_task(sampler.run())

        queue: asyncio.Queue = asyncio.Queue()
        for batch in _batches(missing, batch_size):
            queue.put_nowait(batch)

        async def worker_loop(pool_name: str, handle) -> None:
            nonlocal gpu_computed, errors
            res_pool = registry.get(pool_name)
            while True:
                try:
                    batch = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                # Build items list: [{raw_id, wav_path, source}]
                items = [
                    {"raw_id": raw_id, "wav_path": wav_path, "source": source}
                    for raw_id, wav_path, source in batch
                ]

                async with res_pool.acquire():
                    await handle.send_task(f"{pool_name}-{gpu_computed}", items)
                    try:
                        result = await handle.read_message(timeout=900.0)
                    except Exception as e:
                        log.error(f"{pool_name}: batch read failed: {e}")
                        errors += len(batch)
                        queue.task_done()
                        continue

                if result["type"] == "error":
                    log.error(f"{pool_name}: worker error: {result['error']}")
                    errors += len(batch)
                    queue.task_done()
                    continue

                # Write diarization_turns for successful files
                turn_rows: list[dict] = []
                for file_result in result.get("rows", []):
                    raw_id = file_result["raw_id"]
                    for turn in file_result.get("turns", []):
                        turn_rows.append({
                            "raw_id": raw_id,
                            "turn_idx": turn["turn_idx"],
                            "start_sec": turn["start_sec"],
                            "end_sec": turn["end_sec"],
                            "speaker_tag": turn["speaker_tag"],
                        })

                if turn_rows:
                    upsert_rows(conn, "diarization_turns", turn_rows, ["raw_id", "turn_idx"])

                # Record successful ids (those that produced turns)
                ok_ids = [fr["raw_id"] for fr in result.get("rows", [])]
                if ok_ids:
                    record_batch(
                        conn, run_id, "segment.diarize", ok_ids, "ok",
                        metrics=result.get("metrics"),
                    )

                # Write raw_segments rows for failed files (diarize_failed) —
                # mirrors speaker.py's read_failed pattern so discovery never
                # retries a broken file.
                failed_ids = result.get("failed_ids", [])
                if failed_ids:
                    fail_batch = [
                        {
                            "raw_id": raw_id,
                            "n_segments": 0,
                            "provenance": "diarize_failed",
                            "segmented_at": _now_ts(),
                        }
                        for raw_id in failed_ids
                    ]
                    upsert_rows(conn, "raw_segments", fail_batch, ["raw_id"])
                    record_batch(
                        conn, run_id, "segment.diarize", failed_ids,
                        "error", error="diarization failed or zero turns",
                    )
                    log.warning(
                        f"{pool_name}: {len(failed_ids)} diarization failure(s), "
                        f"marked provenance=diarize_failed: {failed_ids[:5]}"
                    )

                gpu_computed += len(ok_ids)
                errors += len(failed_ids)
                queue.task_done()

                total_done = len(reuse_rows) + gpu_computed + errors
                if total_done and total_done % (batch_size * 20) < batch_size:
                    rate = (gpu_computed + errors) / (time.time() - t0)
                    log.info(
                        f"segment.diarize GPU: {gpu_computed} computed, "
                        f"{errors} errors ({rate:.1f}/s), "
                        f"pools={registry.snapshot()}"
                    )

        await asyncio.gather(*(
            worker_loop(pool_name, handles[pool_name]) for pool_name in pool_names
        ))

        sampler.stop()
        await asyncio.gather(sampler_task, return_exceptions=True)
        for handle in handles.values():
            await handle.shutdown()

    elapsed = time.time() - t0
    total = len(reuse_rows) + gpu_computed
    log.info(
        f"segment.diarize DONE: {len(reuse_rows)} legacy-reused + "
        f"{gpu_computed} GPU-diarized = {total} total, "
        f"{errors} errors, {elapsed:.0f}s, run_id={run_id}"
    )
    return {
        "reused": len(reuse_rows),
        "gpu_computed": gpu_computed,
        "errors": errors,
        "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# segment.diarize GPU worker (subprocess, JSONL stdio protocol)
# ---------------------------------------------------------------------------

class DiarizeWorker(GPUWorkerBase):
    """Pyannote speaker-diarization-3.1 worker.

    Loaded once per subprocess invocation.  ``forward_batch`` decodes each raw
    WAV to 16 kHz with ``torchaudio.transforms.Resample`` (intentionally NOT
    soxr) to match the resampler used by scripts/03_segment.py's
    ``audio_to_16k()`` — diarization boundary consistency requires identical
    preprocessing.  See design decision (e).

    VAD-only fallback: only when pyannote genuinely cannot authenticate (no
    cached HF login AND no HUGGING_FACE_HUB_TOKEN, or a 401/403 gated-model
    error), pyannote is not loaded and each file is treated as one turn
    spanning the full duration with speaker_tag='SPEAKER_UNKNOWN'.  This
    mirrors the legacy script's VAD-only fallback path exactly.
    """

    def load_model(self):
        """Load pyannote speaker-diarization-3.1, or return None for VAD-only mode.

        2026-07-05 fix: don't require HUGGING_FACE_HUB_TOKEN explicitly — the
        prior version short-circuited to VAD-only whenever that one env var
        was unset, even when a working `huggingface-cli login` cache token
        existed (as it does on this machine), silently degrading a real
        backlog run to no-diarization mode.  huggingface_hub's own
        from_pretrained() already resolves the cached login token when no
        explicit token is passed — same default-cache reliance label_suite.py
        already uses for pyannote/segmentation-3.0.  An explicit
        HUGGING_FACE_HUB_TOKEN env var, if set, still overrides.
        """
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

        from pyannote.audio import Pipeline as PyannotePipeline
        log.info(f"DiarizeWorker: loading pyannote speaker-diarization-3.1 on {self.device} ...")
        try:
            pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=token,  # None -> huggingface_hub falls back to its own cached login
            )
        except Exception as exc:
            msg = str(exc)
            if "403" in msg or "401" in msg or "gated" in msg.lower():
                log.error(
                    "pyannote/speaker-diarization-3.1 requires accepting model terms "
                    "and a valid HF login (cached or HUGGING_FACE_HUB_TOKEN).\n"
                    "  1. Visit: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                    "  2. Accept user conditions with your HuggingFace account.\n"
                    "  3. Run `huggingface-cli login` (or set HUGGING_FACE_HUB_TOKEN) and re-run.\n"
                    "Falling back to VAD-only mode (no diarization — multi-speaker risk)."
                )
                return None  # VAD-only mode
            raise
        pipeline.to(torch.device(self.device))
        log.info(f"DiarizeWorker: pyannote pipeline loaded on {self.device}")
        return pipeline

    def forward_batch(self, items: list[dict]) -> list[dict]:
        """Diarize each item ({raw_id, wav_path, source}).

        Returns a list of result dicts:
          {raw_id, turns: [{turn_idx, start_sec, end_sec, speaker_tag}]}
        or {raw_id, _failed: True} on error / zero turns.

        Resamples from 48 kHz to 16 kHz using torchaudio.transforms.Resample
        (not soxr) to match scripts/03_segment.py's ``audio_to_16k()``.
        """
        import soundfile as sf
        import torchaudio

        results = []
        for item in items:
            raw_id = item["raw_id"]
            wav_path = item["wav_path"]

            try:
                # Read 48 kHz master
                wav48, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
                if wav48.ndim > 1:
                    wav48 = wav48.mean(axis=1)
                if sr != TARGET_SR:
                    log.warning(
                        f"DiarizeWorker: expected {TARGET_SR} Hz, got {sr} Hz: {wav_path}; "
                        f"resampling with torchaudio"
                    )
                    t = torch.from_numpy(wav48).float().unsqueeze(0)
                    rs = torchaudio.transforms.Resample(sr, TARGET_SR)
                    wav48 = rs(t).squeeze(0).numpy()

                full_duration = len(wav48) / TARGET_SR

                # Transient 16 kHz copy for pyannote / VAD-only mode
                t16 = torch.from_numpy(wav48).float().unsqueeze(0)
                resampler16 = torchaudio.transforms.Resample(TARGET_SR, VAD_SR)
                wav16 = resampler16(t16).squeeze(0)

                if self.model is None:
                    # VAD-only mode: emit one turn spanning the full file
                    turns = [(0.0, full_duration, "SPEAKER_UNKNOWN")]
                else:
                    wav16_tensor = wav16.unsqueeze(0)  # (1, T) for pyannote
                    output = self.model(
                        {"waveform": wav16_tensor, "sample_rate": VAD_SR}
                    )
                    if hasattr(output, "exclusive_speaker_diarization"):
                        annotation = output.exclusive_speaker_diarization
                    else:
                        annotation = output
                    turns = [
                        (seg.start, seg.end, speaker)
                        for seg, _, speaker in annotation.itertracks(yield_label=True)
                    ]

                if not turns:
                    log.warning(
                        f"DiarizeWorker: zero turns from pyannote for {raw_id} "
                        f"({wav_path}); treating as diarize_failed"
                    )
                    results.append({"raw_id": raw_id, "_failed": True})
                    continue

                log.info(
                    f"DiarizeWorker: {raw_id} — {len(turns)} turns, "
                    f"{len(set(s for _, _, s in turns))} speakers"
                )
                turn_list = [
                    {
                        "turn_idx": idx,
                        "start_sec": round(start, 3),
                        "end_sec": round(end, 3),
                        "speaker_tag": speaker,
                    }
                    for idx, (start, end, speaker) in enumerate(turns)
                ]
                results.append({"raw_id": raw_id, "turns": turn_list})

            except Exception as e:
                log.error(f"DiarizeWorker: failed {raw_id} ({wav_path}): {e}")
                results.append({"raw_id": raw_id, "_failed": True})

        return results


def _worker_main_diarize() -> None:
    """Subprocess entry point for the segment.diarize GPU worker.

    Reads JSONL task messages from stdin, writes JSONL results to stdout.
    Protocol mirrors speaker.py's EmbedWorker exactly.
    """
    ap = argparse.ArgumentParser(description="segment.diarize GPU worker")
    ap.add_argument("--node", default="diarize", help=argparse.SUPPRESS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mem-fraction", type=float, default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # fp16=False: pyannote models keep parts in float32; forcing fp16 would
    # corrupt diarization output (same reasoning as EmbedWorker in speaker.py).
    worker = DiarizeWorker(args.device, mem_fraction=args.mem_fraction, fp16=False)

    def emit(msg: dict) -> None:
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    emit({
        "type": "ready",
        "node": "segment.diarize",
        "pid": os.getpid(),
        "proto": 1,
    })

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
            raw_results = worker.infer_with_oom_halving(items)
            rows = [r for r in raw_results if not r.get("_failed")]
            failed_ids = [r["raw_id"] for r in raw_results if r.get("_failed")]
            elapsed = time.time() - t0
            emit({
                "type": "result",
                "task_id": task_id,
                "rows": rows,
                "failed_ids": failed_ids,
                "metrics": {
                    "items_s": round(len(rows) / elapsed, 2) if elapsed > 0 else 0.0
                },
            })
        except Exception as e:
            emit({"type": "error", "task_id": task_id, "error": str(e), "retryable": True})


# ===========================================================================
# segment.vad_cut  (in-supervisor, ThreadPoolExecutor, CPU+I/O)
# ===========================================================================

VAD_CUT_DISCOVER_SQL = """
    SELECT DISTINCT dt.raw_id
    FROM diarization_turns dt
    LEFT JOIN raw_segments rs ON dt.raw_id = rs.raw_id
    WHERE rs.raw_id IS NULL
    ORDER BY dt.raw_id
"""


def discover_vad_cut(conn) -> list[str]:
    """Return raw_ids that have diarization_turns but no raw_segments row yet."""
    return [row[0] for row in conn.execute(VAD_CUT_DISCOVER_SQL).fetchall()]


def _vad_cut_one(
    raw_id: str,
    wav_path: str,
    source: str,
    source_url: str,
    program: str,
    domain: str,
    style: str,
    turns: list[tuple],  # (turn_idx, start_sec, end_sec, speaker_tag) ordered
) -> dict:
    """Per-raw-file VAD-cut task.  Runs in a thread-pool worker.

    Reads the 48 kHz master via bus.decode (zero-cost passthrough for already-48
    kHz files — matches filter.py's documented rationale).  Makes a transient
    16 kHz copy with torchaudio.Resample for Silero VAD.  Cuts valid clips
    [MIN_DUR, MAX_DUR] and writes lossless 48 kHz mono FLAC files (2026-07-05
    P5-A decision — see DECISIONS.md 2026-07-04 storage-format entry; libsndfile
    writes FLAC natively, no new dependency).

    Returns a dict:
      {raw_id, segment_rows: [...], n_segments: int, error: str|None}
    """
    import soundfile as sf
    from pipeline.audio.bus import decode

    out_dir = _segments_out_dir(raw_id, source)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(wav_path).stem

    try:
        wav48 = decode(str(wav_path), TARGET_SR)
        if wav48 is None or len(wav48) == 0:
            return {
                "raw_id": raw_id,
                "segment_rows": [],
                "n_segments": 0,
                "error": f"decode returned None/empty for {wav_path}",
            }

        # Transient 16 kHz copy for Silero VAD
        wav16 = _audio_to_16k(wav48)

        segment_rows: list[dict] = []
        n_seg = 0
        today = datetime.date.today().isoformat()

        for turn_idx, t_start, t_end, speaker_tag in turns:
            if t_end - t_start < MIN_DUR:
                continue
            vad_windows = _get_vad_segments_in_window(wav16, t_start, t_end)
            for v_start, v_end in vad_windows:
                dur = v_end - v_start
                if dur < MIN_DUR or dur > MAX_DUR:
                    continue

                # Cut clip from 48 kHz master
                s = int(v_start * TARGET_SR)
                e = int(v_end * TARGET_SR)
                clip = wav48[s:e]
                if len(clip) < MIN_DUR * TARGET_SR:
                    continue

                seg_name = f"{stem}_seg{n_seg:05d}.flac"
                seg_path = out_dir / seg_name
                seg_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(seg_path), clip, TARGET_SR, format="FLAC", subtype="PCM_16")

                seg_id = _segment_id(seg_path)
                segment_rows.append({
                    "id": seg_id,
                    "audio_path": str(seg_path),
                    "source": source,
                    "source_url": source_url,
                    "program": program,
                    "domain": domain,
                    "duration_sec": round(dur, 3),
                    "sample_rate": TARGET_SR,
                    "speaker_id": None,
                    "gender": None,
                    "style": style,
                    "created_at": today,
                    "raw_id": raw_id,
                })
                n_seg += 1

        del wav16  # free transient copy

        return {
            "raw_id": raw_id,
            "segment_rows": segment_rows,
            "n_segments": n_seg,
            "error": None,
        }

    except Exception as exc:
        return {
            "raw_id": raw_id,
            "segment_rows": [],
            "n_segments": 0,
            "error": str(exc),
        }


async def run_segment_vad_cut(
    *,
    conn=None,
    n_threads: int | None = None,
    limit: int | None = None,
) -> dict:
    """Supervisor coroutine for the segment.vad_cut DAG node.

    Runs in-supervisor with a ThreadPoolExecutor (one thread per raw file).
    See design decision (c) in the module docstring for the choice of
    in-supervisor vs subprocess pool.

    For every raw_id discovered (diarization_turns exists, raw_segments does
    not): loads its wav_path/source from raw_files, loads its diarization_turns,
    runs _vad_cut_one() in a thread, then upserts the resulting segment rows
    into ``segments`` and writes one ``raw_segments`` completion record.

    Even if a raw_id produces 0 valid clips (e.g. all VAD windows out of range),
    we still write a raw_segments row with n_segments=0 so the discovery query
    never retries it.

    conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run segment.vad_cut` usage.
    """
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    raw_ids = discover_vad_cut(conn)
    if limit:
        raw_ids = raw_ids[:limit]
    log.info(f"segment.vad_cut: {len(raw_ids)} raw files to VAD-cut")
    if not raw_ids:
        return {"processed": 0, "total_segments": 0, "errors": 0}

    run_id = new_run_id("segment.vad_cut")
    t0 = time.time()
    n_threads_actual = n_threads or min(16, (os.cpu_count() or 4) * 2)

    # Pre-load Silero VAD model on the main thread before spawning threads
    # (torch.hub.load is not thread-safe on first call).
    _get_vad_model()

    # Fetch all metadata we need for the discovered raw_ids in one query
    placeholders = ", ".join("?" * len(raw_ids))
    meta_rows = conn.execute(
        f"""
        SELECT raw_id, wav_path, source, source_url, program, domain, style
        FROM raw_files
        WHERE raw_id IN ({placeholders})
        """,
        raw_ids,
    ).fetchall()
    meta_by_id = {
        row[0]: {
            "wav_path": row[1],
            "source": row[2],
            "source_url": row[3] or "",
            "program": row[4] or "",
            "domain": row[5] or "",
            "style": row[6] or "",
        }
        for row in meta_rows
    }

    # Fetch all diarization_turns for the discovered raw_ids in one query
    turns_rows = conn.execute(
        f"""
        SELECT raw_id, turn_idx, start_sec, end_sec, speaker_tag
        FROM diarization_turns
        WHERE raw_id IN ({placeholders})
        ORDER BY raw_id, turn_idx
        """,
        raw_ids,
    ).fetchall()
    turns_by_id: dict[str, list[tuple]] = {}
    for raw_id, turn_idx, start_sec, end_sec, speaker_tag in turns_rows:
        turns_by_id.setdefault(raw_id, []).append(
            (turn_idx, start_sec, end_sec, speaker_tag)
        )

    processed = 0
    total_segments = 0
    errors = 0

    def _make_task(raw_id: str):
        m = meta_by_id.get(raw_id, {})
        turns = turns_by_id.get(raw_id, [])
        return _vad_cut_one(
            raw_id=raw_id,
            wav_path=m.get("wav_path", ""),
            source=m.get("source", ""),
            source_url=m.get("source_url", ""),
            program=m.get("program", ""),
            domain=m.get("domain", ""),
            style=m.get("style", ""),
            turns=turns,
        )

    with ThreadPoolExecutor(max_workers=n_threads_actual) as pool:
        futures = {pool.submit(_make_task, raw_id): raw_id for raw_id in raw_ids}
        for fut in futures:
            raw_id = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                log.error(f"segment.vad_cut: unexpected exception for {raw_id}: {exc}")
                upsert_rows(conn, "raw_segments", [{
                    "raw_id": raw_id,
                    "n_segments": 0,
                    "provenance": "segment_vad_cut",
                    "segmented_at": _now_ts(),
                }], ["raw_id"])
                record_batch(
                    conn, run_id, "segment.vad_cut", [raw_id],
                    "error", error=str(exc),
                )
                errors += 1
                processed += 1
                continue

            if result["error"]:
                log.error(
                    f"segment.vad_cut: error for {raw_id}: {result['error']}"
                )

            seg_rows = result["segment_rows"]
            if seg_rows:
                upsert_rows(conn, "segments", seg_rows, ["id"])
                record_batch(
                    conn, run_id, "segment.vad_cut",
                    [r["id"] for r in seg_rows], "ok",
                )

            # Always write raw_segments row — even for n_segments=0 — to mark
            # this raw_id as done so discovery never retries it.
            upsert_rows(conn, "raw_segments", [{
                "raw_id": raw_id,
                "n_segments": result["n_segments"],
                "provenance": "segment_vad_cut",
                "segmented_at": _now_ts(),
            }], ["raw_id"])

            if result["error"]:
                record_batch(
                    conn, run_id, "segment.vad_cut", [raw_id],
                    "error", error=result["error"],
                )
                errors += 1
            else:
                if not seg_rows:
                    # Zero valid clips is not an error per se, but log it.
                    log.warning(
                        f"segment.vad_cut: {raw_id} produced 0 valid clips "
                        f"(all VAD windows out of [{MIN_DUR}, {MAX_DUR}] s range)"
                    )

            processed += 1
            total_segments += result["n_segments"]

            if processed % 50 == 0 or processed == len(raw_ids):
                rate = processed / (time.time() - t0)
                log.info(
                    f"segment.vad_cut: {processed}/{len(raw_ids)} files done "
                    f"({rate:.1f}/s), {total_segments} segments written, "
                    f"{errors} errors"
                )

    elapsed = time.time() - t0
    log.info(
        f"segment.vad_cut DONE: {processed} raw files, "
        f"{total_segments} segments written, {errors} errors, "
        f"{elapsed:.0f}s, run_id={run_id}"
    )
    return {
        "processed": processed,
        "total_segments": total_segments,
        "errors": errors,
        "run_id": run_id,
    }


# ===========================================================================
# pregate.snr  (cpu+io, in-supervisor ThreadPoolExecutor)
# ===========================================================================

# 03b_acoustic_pregate.py's SNR formula — sliding hop, percentile-based.
# See design decision (d) in the module docstring: this is deliberately a
# DIFFERENT formula from filter.py's compute_snr() (non-overlapping frames,
# sorted-list percentile).  Do NOT merge or replace one with the other.

def compute_pregate_snr(wav48: np.ndarray, sr: int = TARGET_SR) -> float:
    """Energy-based SNR estimate — verbatim port of scripts/03b_acoustic_pregate.py.

    Uses overlapping 25 ms frames with 10 ms hop, computes per-frame mean RMS
    energy, then returns 10 * log10(90th-percentile / 10th-percentile).

    This is deliberately NOT the same as filter.py's compute_snr() which uses
    non-overlapping frames sorted into a list.  See design decision (d).
    """
    frame_len = int(sr * 0.025)
    hop = int(sr * 0.010)
    frames = [wav48[i : i + frame_len] for i in range(0, len(wav48) - frame_len, hop)]
    if not frames:
        return 0.0
    energies = np.array([np.mean(f ** 2) + 1e-10 for f in frames])
    noise_floor = float(np.percentile(energies, 10))
    signal_peak = float(np.percentile(energies, 90))
    if noise_floor <= 0:
        return 0.0
    return round(10 * np.log10(signal_peak / noise_floor), 2)


PREGATE_DISCOVER_SQL = """
    SELECT s.id, s.audio_path
    FROM segments s
    LEFT JOIN pregate p ON s.id = p.id
    WHERE p.id IS NULL
      AND s.raw_id IS NOT NULL
    ORDER BY s.id
"""


def discover_pregate(conn) -> list[tuple]:
    """Return (id, audio_path) for pipeline-cut segments not yet in pregate.

    Only segments with raw_id IS NOT NULL are considered — legacy-imported
    segments (raw_id IS NULL) are never passed through this gate.
    """
    return conn.execute(PREGATE_DISCOVER_SQL).fetchall()


def _pregate_one(
    seg_id: str,
    audio_path: str,
    min_snr: float,
    min_dnsmos: float,
) -> dict:
    """Compute SNR + DNSMOS for one segment.  Runs in a thread-pool worker.

    Ports the per-segment logic from scripts/03b_acoustic_pregate.py's main()
    loop verbatim:
      1. Read WAV, compute SNR with compute_pregate_snr().
      2. If SNR < threshold: fail('snr'), skip DNSMOS.
      3. Else if min_dnsmos > 0: compute DNSMOS via speechmos.dnsmos.run()
         (NOT filter.py's capped-ORT-session trick — see design decision (d)).
      4. On any exception: fail-open (pass=True, both metrics=None) — matches
         03b's own exception handler which lets errors through rather than
         blocking the pipeline on a corrupt-file edge case.

    Returns a pregate table row dict.
    """
    try:
        import soundfile as sf

        wav48, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if wav48.ndim > 1:
            wav48 = wav48.mean(axis=1)

        snr = compute_pregate_snr(wav48, sr)
        dns_score: Optional[float] = None
        reason: Optional[str] = None

        if snr < min_snr:
            reason = "snr"
        elif min_dnsmos > 0:
            # DNSMOS via speechmos.dnsmos.run() — simple, exactly like 03b.
            # This is a fast early-reject pass; no need for the capped-ORT-session
            # trick that filter.acoustic uses for authoritative scoring.
            import torch
            import torchaudio
            from speechmos import dnsmos as _dnsmos_mod

            t = torch.from_numpy(wav48).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav16 = resampler(t).squeeze(0).numpy()
            # torchaudio's sinc-based resampler can overshoot slightly past
            # [-1, 1] (Gibbs phenomenon) on audio whose peaks already sit at
            # full scale — found 2026-07-05 running this at production scale
            # for the first time (409/11266 segments hit it, all from
            # already-clipped source audio). speechmos.dnsmos.run() validates
            # its input range strictly and raises on the tiniest overshoot;
            # clamping here is standard practice and doesn't materially
            # change the perceptual score.
            wav16 = np.clip(wav16, -1.0, 1.0)

            result = _dnsmos_mod.run(wav16, sr=16000)
            dns_score = round(float(result["ovrl_mos"]), 3)
            if dns_score < min_dnsmos:
                reason = "dnsmos"

        return {
            "id": seg_id,
            "snr_db": snr,
            "dnsmos": dns_score,
            "pass": reason is None,
            "fail_reason": reason,
            "provenance": "pregate_snr",
        }

    except Exception as exc:
        log.warning(f"pregate.snr: error on {seg_id} ({audio_path}): {exc}")
        # Fail-open: let the segment through rather than blocking the pipeline
        # on a corrupt-file edge case — verbatim 03b behaviour.
        return {
            "id": seg_id,
            "snr_db": None,
            "dnsmos": None,
            "pass": True,
            "fail_reason": None,
            "provenance": "pregate_snr",
        }


async def run_pregate_snr(
    *,
    conn=None,
    min_snr: float = DEFAULT_MIN_SNR,
    min_dnsmos: float = DEFAULT_MIN_DNSMOS,
    n_threads: int | None = None,
    batch_size: int = 500,
    limit: int | None = None,
) -> dict:
    """Supervisor coroutine for the pregate.snr DAG node.

    Runs in-supervisor with a ThreadPoolExecutor — same architecture as
    segment.vad_cut.  SNR computation is CPU-bound-ish but fast; DNSMOS via
    speechmos.dnsmos.run() is fast on CPU for short clips and does not require
    the subprocess / capped-ORT-session machinery that filter.acoustic uses for
    the authoritative gate.

    For each discovered segment, calls _pregate_one() in a thread and writes
    one ``pregate`` row.  On any read/compute error, writes a fail-open row
    (pass=True) matching 03b's own behaviour.

    conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run pregate.snr` usage.
    """
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    rows = discover_pregate(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"pregate.snr: {len(rows)} segments to gate (min_snr={min_snr}, min_dnsmos={min_dnsmos})")
    if not rows:
        return {"processed": 0, "passed": 0, "failed_snr": 0, "failed_dnsmos": 0, "errors": 0}

    run_id = new_run_id("pregate.snr")
    t0 = time.time()
    n_threads_actual = n_threads or min(16, (os.cpu_count() or 4) * 2)

    processed = 0
    passed = 0
    failed_snr = 0
    failed_dnsmos = 0
    errors = 0  # count of exception-based fail-open rows

    def _task(row: tuple) -> dict:
        seg_id, audio_path = row
        return _pregate_one(seg_id, audio_path, min_snr, min_dnsmos)

    with ThreadPoolExecutor(max_workers=n_threads_actual) as pool:
        for batch in _batches(rows, batch_size):
            out_rows = list(pool.map(_task, batch))
            upsert_rows(conn, "pregate", out_rows, ["id"])
            ok_ids = [r["id"] for r in out_rows if r["pass"]]
            err_ids = [r["id"] for r in out_rows if not r["pass"]]
            if ok_ids:
                record_batch(conn, run_id, "pregate.snr", ok_ids, "ok")
            if err_ids:
                record_batch(conn, run_id, "pregate.snr", err_ids, "error",
                             error="snr or dnsmos below threshold")

            for r in out_rows:
                processed += 1
                if r["pass"]:
                    if r["snr_db"] is None:  # fail-open from exception
                        errors += 1
                    else:
                        passed += 1
                else:
                    if r["fail_reason"] == "snr":
                        failed_snr += 1
                    else:
                        failed_dnsmos += 1

            if processed % (batch_size * 10) < batch_size or processed == len(rows):
                rate = processed / (time.time() - t0)
                log.info(
                    f"pregate.snr: {processed}/{len(rows)} ({rate:.1f}/s) — "
                    f"passed={passed}, fail_snr={failed_snr}, "
                    f"fail_dnsmos={failed_dnsmos}, errors={errors}"
                )

    elapsed = time.time() - t0
    log.info(
        f"pregate.snr DONE: {processed} segments, {passed} passed "
        f"(snr_fail={failed_snr}, dnsmos_fail={failed_dnsmos}, "
        f"errors/fail-open={errors}), {elapsed:.0f}s, run_id={run_id}"
    )
    return {
        "processed": processed,
        "passed": passed,
        "failed_snr": failed_snr,
        "failed_dnsmos": failed_dnsmos,
        "errors": errors,
        "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# Subprocess entry point dispatcher
# (only segment.diarize needs a GPU subprocess worker;
# segment.vad_cut and pregate.snr run in-supervisor)
# ---------------------------------------------------------------------------

def worker_main() -> None:
    """Dispatch to the correct subprocess entry point based on --node argument.

    Currently only 'diarize' is supported (the only node in this module that
    needs a GPU subprocess worker with the JSONL stdio protocol).
    segment.vad_cut and pregate.snr are in-supervisor and have no subprocess
    entry point.
    """
    # Peek at --node before delegating so we can route without argparse conflict
    # between the dispatcher and the node-specific parsers.
    if "--node" in sys.argv:
        idx = sys.argv.index("--node")
        if idx + 1 < len(sys.argv):
            node = sys.argv[idx + 1]
        else:
            node = "diarize"
    else:
        node = "diarize"

    if node == "diarize":
        _worker_main_diarize()
    else:
        print(f"Unknown --node value: {node!r}. Valid: diarize", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    worker_main()
