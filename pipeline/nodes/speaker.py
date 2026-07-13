"""
pipeline/nodes/speaker.py
=========================
Two DAG nodes for the Cantonese speech-corpus pipeline:

  speaker.embed   — ensures every ``segments`` row has a matching row in
                    ``speaker_embeddings`` (one ECAPA-TDNN d-vector .npy file
                    reference per segment).

  speaker.cluster — cross-file speaker clustering: loads all embeddings for
                    each source, runs agglomerative clustering, and writes one
                    row per segment into ``speakers``.

-------------------------------------------------------------------------------
Design decisions and rationale
-------------------------------------------------------------------------------

(a) Hybrid reuse-first design for speaker.embed
    A random 2 000-sample check against the live catalog (455 299 rows) found
    that 100 % of ``segments.audio_path`` values already have a matching
    ``<path>.embed.npy`` sidecar on disk, written by the legacy
    ``scripts/08_speaker_id.py``.  Running a full ECAPA-TDNN GPU pass over
    455 k files would waste 10-20 GPU-hours for no gain.  Instead we use a
    three-phase approach:

      1. Discovery (SQL anti-join): find segments not yet in
         ``speaker_embeddings``.
      2. Cheap reuse pass (CPU, thread pool): for every discovered segment,
         check whether the sidecar ``.embed.npy`` file exists on disk.  On a
         hit, upsert immediately with ``provenance='legacy_reused'`` — no
         file content validation (too expensive at this scale; trust the file
         exists = valid, same spirit as bus.py's zero-cost passthrough decode).
      3. GPU fallback: only for the (expected tiny or zero) remainder that has
         no cached sidecar, spawn one ECAPA-TDNN worker per device.  If the
         missing list is empty we skip all subprocess spawning entirely.

    Discovery is scoped off ``segments`` directly (not gated on
    ``filters.pass = TRUE`` the way g2p is) because ``segments`` already IS
    the legacy already-filtered-passing corpus (imported wholesale from
    manifest.jsonl in P0) — unlike g2p's text-domain gate, which specifically
    waits for a segment to be re-decided by the new filter.decide node.
    Speaker embedding is an audio-only operation independent of that text
    re-filtering status.

(b) Two separate tables: speaker_embeddings and speakers
    Splitting the embedding reference from the cluster assignment follows the
    upsert-clobbering precedent established elsewhere in the pipeline: if both
    lived in one table, re-running speaker.embed would clobber speaker_id
    columns written by speaker.cluster, and vice-versa. speaker.embed is the
    sole writer of speaker_embeddings; speaker.cluster is the sole writer of
    speakers — each always writes its full row, so INSERT OR REPLACE never
    clobbers a column the other node owns.

(c) speaker.cluster recomputes the whole source every run
    Agglomerative clustering is a global, order-dependent algorithm. Adding
    even one new embedding can shift every cluster boundary. Incremental
    per-item discovery makes no sense here: the only correct strategy is to
    reload all embeddings for a source and recluster from scratch. This
    mirrors the legacy script's behaviour (which always reclusters on every
    invocation) and is safe because ``upsert_rows`` does ``INSERT OR REPLACE``,
    so stale rows are simply overwritten.

(d) Golden-parity note
    Because speaker.embed writes exactly the same sidecar files (same path
    convention, same ECAPA-TDNN model and weights, same 16 kHz resampling) as
    the legacy script, and speaker.cluster ports ``cluster_embeddings()``
    verbatim (same threshold, same sklearn back-end), the *co-clustering*
    results should closely match the legacy ``segments.speaker_id`` column.
    However, the integer cluster IDs themselves are arbitrary 0-based labels
    whose numbering is sensitive to data order and sklearn's internal sort.
    Do NOT compare speaker_id *strings* for parity — compare which segments
    end up in the same cluster (confusion-matrix / co-clustering approach).

(e) Resampler consistency
    The GPU fallback worker resamples to 16 kHz using
    ``torchaudio.transforms.Resample`` — the same method the legacy script
    used in ``extract_embedding()`` — rather than soxr (used by bus.py).
    Matching the legacy resampler exactly keeps freshly-computed embeddings
    comparable to legacy-reused ones in the same clustering pass.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from pipeline.workers.gpu_base import GPUWorkerBase

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _batches(rows: list, size: int):
    """Yield successive fixed-size slices of *rows*."""
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


# ===========================================================================
# speaker.embed
# ===========================================================================

EMBED_DISCOVER_SQL = """
    SELECT s.id, s.source, s.audio_path
    FROM segments s
    LEFT JOIN speaker_embeddings se ON s.id = se.id
    WHERE se.id IS NULL
    ORDER BY s.source, s.id
"""


def discover_embed(conn) -> list[tuple]:
    """Return (id, source, audio_path) for segments not yet in speaker_embeddings."""
    return conn.execute(EMBED_DISCOVER_SQL).fetchall()


EXISTING_EMBED_SQL = """
    SELECT se.id, se.source, s.audio_path, se.embedding_ref
    FROM speaker_embeddings se
    JOIN segments s ON s.id = se.id
    WHERE se.embedding_ref IS NOT NULL
    ORDER BY se.source, se.id
"""


def discover_stale_embed(conn) -> list[tuple]:
    """Return (id, source, audio_path, embedding_ref) for every existing
    speaker_embeddings row with a non-null ref — candidates for the on-disk
    existence check in run_speaker_embed(verify_existing=True).

    Why this exists: unlike asr_agreement.model_count or filters/tiers'
    provenance-tagged anti-join, there is no cheap catalog-only signal for
    "this row's sidecar file got deleted out from under it" — the 2026-07
    `filtered/` tree retirement (§7.3) silently orphaned 455k+ embedding_ref
    values that still point inside that now-deleted tree. Checking file
    existence needs a disk stat per row, so it is opt-in (verify_existing
    flag) rather than run on every invocation — normal runs only need the
    cheap discover_embed() anti-join.
    """
    return conn.execute(EXISTING_EMBED_SQL).fetchall()


def _check_sidecar(row: tuple) -> tuple[str, str, str | None]:
    """I/O task: check whether <audio_path>.embed.npy exists.

    Returns (id, source, sidecar_path_str_or_None).
    Designed to be called inside a ThreadPoolExecutor (GIL-released Path.exists).
    """
    seg_id, source, audio_path = row
    sidecar = Path(audio_path).with_suffix(".embed.npy")
    return (seg_id, source, str(sidecar) if sidecar.exists() else None)


async def run_speaker_embed(
    devices: list[str],
    *,
    conn=None,
    gpu_policy: str = "cap",
    batch_size: int = 5000,
    mem_fraction: float | None = 0.15,
    limit: int | None = None,
    verify_existing: bool = False,
) -> dict:
    """Supervisor coroutine for the speaker.embed DAG node.

    Phase 0 (opt-in, verify_existing=True) — stale-ref repair: stat every
               existing speaker_embeddings.embedding_ref on disk; any row
               whose file is missing (e.g. orphaned by the 2026-07
               filtered/ tree retirement) is re-queued using the segment's
               CURRENT audio_path, so it flows through the normal Phase 2/3
               pass below exactly like a brand-new segment.
    Phase 1 — discovery (SQL anti-join).
    Phase 2 — cheap reuse pass: parallel disk-existence checks for legacy
               sidecar .embed.npy files; on hit, upsert with
               provenance='legacy_reused' without touching the file contents.
    Phase 3 — GPU fallback: only for the remainder without a cached sidecar,
               spawn one ECAPA-TDNN worker subprocess per device.  Skipped
               entirely when the remainder is empty.

    conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run speaker.embed` usage.
    """
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch
    from pipeline.orchestrator.pools import PoolRegistry
    from pipeline.orchestrator.resources import GpuPolicy, Sampler
    from pipeline.orchestrator.worker import spawn_worker

    conn = conn or connect()
    rows = discover_embed(conn)

    if verify_existing:
        existing = discover_stale_embed(conn)
        io_workers = min(32, (os.cpu_count() or 4) * 4)
        log.info(
            f"speaker.embed: verify_existing - stat'ing {len(existing)} "
            f"existing embedding_ref files with {io_workers} I/O threads ..."
        )
        with ThreadPoolExecutor(max_workers=io_workers) as pool:
            exists_flags = list(pool.map(lambda r: Path(r[3]).exists(), existing))
        stale = [
            (seg_id, source, audio_path)
            for (seg_id, source, audio_path, _ref), ok in zip(existing, exists_flags)
            if not ok
        ]
        log.info(
            f"speaker.embed: verify_existing found {len(stale)}/{len(existing)} "
            f"rows with a missing embedding_ref file, re-queuing for recompute"
        )
        rows = rows + stale

    if limit:
        rows = rows[:limit]
    log.info(f"speaker.embed: {len(rows)} segments need speaker_embeddings rows")
    if not rows:
        return {"reused": 0, "gpu_computed": 0, "errors": 0}

    run_id = new_run_id("speaker.embed")
    t0 = time.time()

    # -----------------------------------------------------------------------
    # Phase 2 — cheap reuse pass (thread pool, I/O-bound)
    # -----------------------------------------------------------------------
    reuse_rows: list[dict] = []
    missing: list[tuple] = []  # (id, source, audio_path) with no sidecar

    io_workers = min(32, (os.cpu_count() or 4) * 4)
    log.info(f"speaker.embed: checking {len(rows)} sidecar .embed.npy files "
             f"with {io_workers} I/O threads ...")

    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        for batch in _batches(rows, batch_size):
            results = list(pool.map(_check_sidecar, batch))
            hits = [(seg_id, source, sidecar) for seg_id, source, sidecar in results if sidecar]
            misses = [
                (seg_id, source, audio_path)
                for (seg_id, source, audio_path), (_, _, sidecar) in zip(batch, results)
                if sidecar is None
            ]
            missing.extend(misses)
            if hits:
                upsert_batch = [
                    {
                        "id": seg_id,
                        "source": source,
                        "embedding_ref": sidecar,
                        "provenance": "legacy_reused",
                    }
                    for seg_id, source, sidecar in hits
                ]
                upsert_rows(conn, "speaker_embeddings", upsert_batch, ["id"])
                record_batch(
                    conn, run_id, "speaker.embed",
                    [r["id"] for r in upsert_batch], "ok",
                )
                reuse_rows.extend(upsert_batch)
            log.info(
                f"speaker.embed reuse pass: {len(reuse_rows)}/{len(rows)} reused, "
                f"{len(missing)} still missing"
            )

    log.info(
        f"speaker.embed: reuse pass complete - "
        f"{len(reuse_rows)} legacy-reused, {len(missing)} need GPU"
    )

    # -----------------------------------------------------------------------
    # Phase 3 — GPU fallback (only if there are remaining rows)
    # -----------------------------------------------------------------------
    gpu_computed = 0
    errors = 0

    if missing:
        log.info(f"speaker.embed: spawning GPU worker(s) for {len(missing)} segments ...")

        registry = PoolRegistry()
        pool_names = []
        for dev in devices:
            pool_name = f"gpu.{dev.split(':')[1]}" if dev.startswith("cuda") else "cpu"
            registry.register(pool_name, target=1)
            pool_names.append(pool_name)

        handles = {}
        for dev, pool_name in zip(devices, pool_names):
            cmd = [
                sys.executable, "-m", "pipeline.nodes.speaker",
                "--device", dev,
            ]
            if mem_fraction is not None and dev.startswith("cuda"):
                cmd += ["--mem-fraction", str(mem_fraction)]
            handle = await spawn_worker(cmd)
            await handle.wait_ready(timeout=180.0)
            handles[pool_name] = handle
            log.info(f"speaker.embed worker ready: {pool_name} -> {dev} (pid={handle.pid})")

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
            pool = registry.get(pool_name)
            while True:
                try:
                    batch = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                # Map id -> source for upsert construction
                meta = {r[0]: r[1] for r in batch}  # id -> source
                items = [{"id": r[0], "path": r[2]} for r in batch]

                async with pool.acquire():
                    await handle.send_task(f"{pool_name}-{gpu_computed}", items)
                    try:
                        result = await handle.read_message(timeout=600.0)
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

                out_rows = [
                    {
                        "id": r["id"],
                        "source": meta[r["id"]],
                        "embedding_ref": r["embedding_ref"],
                        "embedding": r["embedding"],
                        "provenance": "speaker_embed_node",
                    }
                    for r in result["rows"]
                ]
                # Always-write-a-row for failures: provenance='read_failed',
                # embedding_ref=None so that discovery never resurfaces them.
                skipped_rows = [
                    {
                        "id": sid,
                        "source": meta[sid],
                        "embedding_ref": None,
                        "embedding": None,
                        "provenance": "read_failed",
                    }
                    for sid in result.get("skipped_ids", [])
                ]
                if skipped_rows:
                    log.warning(
                        f"{pool_name}: {len(skipped_rows)} unreadable segment(s), "
                        f"marked provenance=read_failed: "
                        f"{[r['id'] for r in skipped_rows][:5]}"
                    )
                all_rows = out_rows + skipped_rows
                if all_rows:
                    upsert_rows(conn, "speaker_embeddings", all_rows, ["id"])
                    record_batch(
                        conn, run_id, "speaker.embed",
                        [r["id"] for r in out_rows], "ok",
                        metrics=result.get("metrics"),
                    )
                    if skipped_rows:
                        record_batch(
                            conn, run_id, "speaker.embed",
                            [r["id"] for r in skipped_rows],
                            "error", error="unreadable audio file",
                        )

                gpu_computed += len(out_rows)
                errors += len(skipped_rows)
                queue.task_done()

                total_done = len(reuse_rows) + gpu_computed + errors
                if total_done and total_done % (batch_size * 5) < batch_size:
                    rate = (gpu_computed + errors) / (time.time() - t0)
                    log.info(
                        f"speaker.embed GPU: {gpu_computed} computed, "
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
        f"speaker.embed DONE: {len(reuse_rows)} legacy-reused + "
        f"{gpu_computed} GPU-computed = {total} total, "
        f"{errors} errors, {elapsed:.0f}s, run_id={run_id}"
    )
    return {
        "reused": len(reuse_rows),
        "gpu_computed": gpu_computed,
        "errors": errors,
        "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# speaker.embed GPU worker (subprocess, JSONL stdio protocol)
# ---------------------------------------------------------------------------

def _load_and_resample_for_embed(item: dict) -> tuple[str, "torch.Tensor", str] | None:
    """I/O + CPU resample only (no GPU/model work) — designed to run inside a
    ThreadPoolExecutor (torchaudio.load releases the GIL during disk I/O), so
    many segments load/resample concurrently while the GPU worker's single
    Python thread would otherwise sit idle waiting on disk one item at a time.
    This is exactly the preprocessing forward_batch used to do inline; moving
    it out doesn't change any per-item numerical result, only when/where it
    runs, so embedding-space parity with the legacy script is unaffected.

    Returns (id, wav, sidecar_path) or None on read failure.
    """
    import torchaudio

    seg_id = item["id"]
    audio_path = item["path"]
    sidecar = str(Path(audio_path).with_suffix(".embed.npy"))
    try:
        wav, sr = torchaudio.load(str(audio_path))
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        return (seg_id, wav, sidecar)
    except Exception as e:
        log.warning(f"EmbedWorker: read/resample fail {audio_path}: {e}")
        return None


class EmbedWorker(GPUWorkerBase):
    """ECAPA-TDNN embedding worker.

    Loaded once per subprocess invocation. Audio I/O + 16 kHz resample
    (``torchaudio.transforms.Resample`` — intentionally NOT soxr, to match the
    resampler used by the legacy ``scripts/08_speaker_id.py:extract_embedding()``
    for embedding-space consistency) happens in worker_main()'s parallel
    ThreadPoolExecutor pass, BEFORE forward_batch() — see
    _load_and_resample_for_embed(). forward_batch() itself only does the
    GPU-bound encode_batch() call plus the .npy sidecar write, so the GPU is
    never left waiting on serialized disk I/O between items.
    """

    def load_model(self):
        from speechbrain.inference.speaker import EncoderClassifier

        if str(self.device).startswith("cuda"):
            torch.cuda.set_device(self.device)

        log.info(f"EmbedWorker: loading ECAPA-TDNN on {self.device} ...")
        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)},
            savedir=str(ROOT / ".cache" / "speechbrain"),
        )
        log.info(f"EmbedWorker: encoder loaded on {self.device}")
        return encoder

    def forward_batch(self, items: list[tuple]) -> list[dict]:
        """Compute embeddings for *items* (list of (id, wav, sidecar_path)
        pre-loaded by _load_and_resample_for_embed via worker_main's I/O pool).

        The embedding is saved to <path>.embed.npy as a side-effect (same
        sidecar convention as the legacy script).
        Returns list of {id, embedding_ref} dicts (or {id, _failed: True}).
        """
        results = []
        for seg_id, wav, sidecar in items:
            try:
                with torch.no_grad():
                    emb = self.model.encode_batch(wav)  # (1, 1, D)
                emb_np = emb.squeeze().cpu().numpy().astype(np.float32)
                np.save(sidecar, emb_np)
                results.append({
                    "id": seg_id,
                    "embedding_ref": sidecar,
                    "embedding": emb_np.tolist(),
                })
            except Exception as e:
                log.error(f"EmbedWorker: encode failed for {seg_id}: {e}")
                # Signal skip — supervisor will write a read_failed row.
                results.append({"id": seg_id, "_failed": True})
        return results


def worker_main() -> None:
    """Subprocess entry point for the speaker.embed GPU worker.

    Reads JSONL task messages from stdin, writes JSONL results to stdout.
    Protocol mirrors label_music.py exactly.
    """
    ap = argparse.ArgumentParser(description="speaker.embed GPU worker")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mem-fraction", type=float, default=None)
    ap.add_argument("--io-workers", type=int, default=min(32, (os.cpu_count() or 4) * 4),
                     help="I/O threads for parallel torchaudio.load+resample "
                          "ahead of the (serialized) GPU encode step")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Cap torch's own intra-op thread pool to 1. torch.set_num_threads() is a
    # process-global setting (not per-Python-thread) — without this, every
    # torchaudio.transforms.Resample call inside the io_pool below spins up
    # its own OpenMP/MKL thread pool sized to all visible cores, and with
    # --io-workers concurrent Python threads each doing that, the process
    # oversubscribes to ~armies of native threads (observed: 890 OS threads
    # for a 32-worker pool) fighting each other for the same physical cores
    # instead of genuinely parallelizing. One torch thread per Python thread
    # + --io-workers Python threads = real, bounded parallelism.
    torch.set_num_threads(1)

    # fp16=False: SpeechBrain's encode_batch is float32 internally; forcing
    # fp16 here would corrupt the embeddings.
    worker = EmbedWorker(args.device, mem_fraction=args.mem_fraction, fp16=False)
    io_pool = ThreadPoolExecutor(max_workers=args.io_workers)
    log.info(f"speaker.embed worker: {args.io_workers} I/O threads for audio load+resample")

    def emit(msg: dict) -> None:
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    emit({
        "type": "ready",
        "node": "speaker.embed",
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
            # Parallel I/O pass (disk read + CPU resample) across all items in
            # this batch, BEFORE the GPU-bound encode step — this is the fix
            # for the GPU sitting idle while torchaudio.load() ran serially
            # one item at a time (see _load_and_resample_for_embed docstring).
            loaded = list(io_pool.map(_load_and_resample_for_embed, items))
            read_failed_ids = [items[i]["id"] for i, r in enumerate(loaded) if r is None]
            kept = [r for r in loaded if r is not None]

            raw_results = worker.infer_with_oom_halving(kept) if kept else []
            rows = [r for r in raw_results if not r.get("_failed")]
            skipped_ids = read_failed_ids + [r["id"] for r in raw_results if r.get("_failed")]
            elapsed = time.time() - t0
            emit({
                "type": "result",
                "task_id": task_id,
                "rows": rows,
                "skipped_ids": skipped_ids,
                "metrics": {
                    "items_s": round(len(rows) / elapsed, 2) if elapsed > 0 else 0.0
                },
            })
        except Exception as e:
            emit({"type": "error", "task_id": task_id, "error": str(e), "retryable": True})

    io_pool.shutdown(wait=False)


# ===========================================================================
# speaker.cluster
# ===========================================================================

# Agglomerative clustering is O(n^2) memory. Above this threshold we cluster a
# random sample and assign every remaining point to its nearest sample-centroid
# by cosine similarity, keeping memory bounded. Configurable via env var to
# allow ad-hoc scaling without code changes (mirrors the legacy script).
_CLUSTER_SAMPLE_MAX = int(os.environ.get("SPK_CLUSTER_SAMPLE_MAX", "12000"))

CLUSTER_DISCOVER_SQL = """
    SELECT DISTINCT source
    FROM speaker_embeddings
    ORDER BY source
"""


def discover_cluster(conn) -> list[str]:
    """Return distinct source values present in speaker_embeddings."""
    return [row[0] for row in conn.execute(CLUSTER_DISCOVER_SQL).fetchall()]


def cluster_embeddings(
    embeddings: np.ndarray,
    source_prefix: str,
    threshold: float = 0.25,
) -> np.ndarray:
    """Cluster by cosine distance.

    Exact agglomerative clustering for small N (<= _CLUSTER_SAMPLE_MAX);
    scalable sample-then-assign for large N (keeps memory bounded).
    Returns an integer label array of shape (N,).

    Ported verbatim from scripts/08_speaker_id.py:cluster_embeddings() to
    guarantee golden parity: same embeddings + same algorithm + same threshold
    -> same co-clustering groups (though not necessarily the same integer IDs).
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    emb_norm = normalize(embeddings)
    n = len(emb_norm)
    if n < 2:
        return np.zeros(n, dtype=int)

    def _agglom(x: np.ndarray) -> np.ndarray:
        return AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="cosine",
            linkage="average",
        ).fit_predict(x)

    if n <= _CLUSTER_SAMPLE_MAX:
        return _agglom(emb_norm)

    # --- Large source: sample -> cluster -> assign-to-nearest-centroid ------
    log.info(
        f"  {source_prefix}: {n} embeddings > {_CLUSTER_SAMPLE_MAX}; "
        f"using scalable sample-and-assign clustering"
    )
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, _CLUSTER_SAMPLE_MAX, replace=False)
    sample = emb_norm[sample_idx]
    sample_labels = _agglom(sample)

    uniq = np.unique(sample_labels)
    # centroid = renormalized mean of each sample cluster
    centroids = np.stack([
        normalize(sample[sample_labels == c].mean(axis=0, keepdims=True))[0]
        for c in uniq
    ])
    # cosine similarity = dot product (both sides are already L2-normalized)
    # assign each point to argmax centroid
    labels = np.empty(n, dtype=int)
    BATCH = 8192
    for start in range(0, n, BATCH):
        block = emb_norm[start : start + BATCH]
        best = (block @ centroids.T).argmax(axis=1)
        labels[start : start + len(block)] = best  # contiguous 0..len(uniq)-1
    return labels


def _load_npy(args: tuple[str, str]) -> tuple[str, str, np.ndarray | None]:
    """I/O task: load a single .npy embedding file.

    Returns (id, embedding_ref, array_or_None).
    None signals a load failure (corrupt / missing despite a non-null ref).
    """
    seg_id, embedding_ref = args
    try:
        arr = np.load(embedding_ref)
        return (seg_id, embedding_ref, arr)
    except Exception as e:
        log.error(f"speaker.cluster: failed to load {embedding_ref}: {e}")
        return (seg_id, embedding_ref, None)


async def run_speaker_cluster(
    *,
    conn=None,
    threshold: float = 0.25,
    sources: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    """Supervisor coroutine for the speaker.cluster DAG node.

    Iterates each source in speaker_embeddings, loads all embeddings for that
    source with a thread pool (I/O-bound), runs cluster_embeddings(), then
    upserts full rows into speakers.

    Unlike every other P3 node this is NOT an anti-join discovery node:
    clustering requires ALL embeddings for a source loaded together, and the
    whole source is always reclustered on every invocation (mirrors legacy
    script behaviour; upsert_rows handles idempotency via INSERT OR REPLACE).

    Parameters
    ----------
    conn:
        Optional pre-opened DuckDB connection (or cursor) — pass one when
        running alongside other nodes under `pipe run-many` (see filter.py's
        run_filter_acoustic docstring for the rationale). Defaults to a fresh
        self-managed connect() for standalone `pipe run speaker.cluster` usage.
    threshold:
        Agglomerative clustering cosine-distance threshold (default 0.25,
        matches the legacy script default).
    sources:
        Optional allow-list of source names. Useful for partial/test runs.
    limit:
        If given, cap the number of segments loaded per source (testing only).
    """
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    all_sources = discover_cluster(conn)
    if sources:
        all_sources = [s for s in all_sources if s in sources]
    log.info(f"speaker.cluster: {len(all_sources)} source(s) to cluster: {all_sources}")
    if not all_sources:
        return {"sources_processed": 0, "total_segments": 0, "total_speakers": 0}

    run_id = new_run_id("speaker.cluster")
    t0 = time.time()

    total_segments = 0
    total_speakers = 0
    io_workers = min(32, (os.cpu_count() or 4) * 4)

    for source in all_sources:
        # Load (id, embedding, embedding_ref) rows for this source, skipping
        # read_failed placeholders (both embedding and embedding_ref NULL).
        # Phase 3 (2026-07-12, docs/IO_OPTIMIZATION_PLAN.md): prefer the
        # in-table `embedding` column -- a single columnar SQL scan instead of
        # opening one .npy sidecar file per segment, which was speaker.cluster's
        # dominant cost (ext4 dentry-cache thrashing, not clustering compute).
        # Rows with embedding IS NULL (not yet backfilled, or legacy-reused
        # sidecar hits that speaker.embed never re-reads at discovery time)
        # fall back to the old per-file _load_npy() path.
        source_rows = conn.execute(
            """
            SELECT id, embedding, embedding_ref
            FROM speaker_embeddings
            WHERE source = ? AND (embedding IS NOT NULL OR embedding_ref IS NOT NULL)
            ORDER BY id
            """,
            [source],
        ).fetchall()

        if limit:
            source_rows = source_rows[:limit]

        if not source_rows:
            log.info(f"{source}: 0 embeddings, skipping")
            continue

        in_table = [(seg_id, ref, np.array(emb, dtype=np.float32))
                    for seg_id, emb, ref in source_rows if emb is not None]
        needs_npy = [(seg_id, ref) for seg_id, emb, ref in source_rows
                     if emb is None and ref is not None]

        loaded_npy: list[tuple[str, str, np.ndarray | None]] = []
        if needs_npy:
            with ThreadPoolExecutor(max_workers=io_workers) as pool:
                loaded_npy = list(pool.map(_load_npy, needs_npy))

        log.info(
            f"{source}: {len(in_table)} from catalog column, "
            f"{len(needs_npy)} fell back to .npy sidecar"
        )

        # Filter out load failures
        valid = in_table + [(seg_id, ref, arr) for seg_id, ref, arr in loaded_npy if arr is not None]
        if not valid:
            log.warning(f"{source}: all {len(source_rows)} embeddings failed to load, skipping")
            continue

        seg_ids = [v[0] for v in valid]
        refs = [v[1] for v in valid]
        embeddings = np.stack([v[2] for v in valid])

        # NOTE (2026-07-12): tried wrapping this call in
        # `loop.run_in_executor(None, cluster_embeddings, ...)` to keep the
        # event loop responsive when this coroutine shares a loop with
        # another node under `pipe run-many` (speaker.cluster's synchronous
        # clustering on a large source was observed starving asr.transcribe's
        # worker-spawn sequence for 30+ minutes). That made things worse, not
        # better: moving the call off the main thread pushed sklearn's joblib
        # backend into a single-threaded fallback (joblib's default backends
        # generally require the main thread to fork/spawn worker processes),
        # making a solo run take 20x+ longer (5+ min vs ~13s for the same
        # source). Reverted -- run speaker.cluster and asr.transcribe as
        # separate solo `pipe run` invocations instead of pairing them under
        # `run-many`; see pending_task.md's T15 entry for the full story.
        labels = cluster_embeddings(embeddings, source, threshold)
        n_clusters = int(len(set(labels.tolist())))

        log.info(f"{source}: {len(seg_ids)} segs -> {n_clusters} speakers")

        # Upsert full rows into speakers (speakers table sole writer; always
        # write every column — mirrors filter.decide being the sole writer of
        # the filters table in the sibling node)
        speaker_rows = [
            {
                "id": seg_id,
                "speaker_id": f"{source}_{int(cluster_id):03d}",
                "cluster_id": int(cluster_id),
                "embedding_ref": ref,
                "gender": "unknown",
                "provenance": "speaker_cluster",
            }
            for seg_id, ref, cluster_id in zip(seg_ids, refs, labels)
        ]
        upsert_rows(conn, "speakers", speaker_rows, ["id"])
        record_batch(
            conn, run_id, "speaker.cluster",
            [r["id"] for r in speaker_rows], "ok",
            metrics={"n_clusters": n_clusters, "n_segments": len(speaker_rows)},
        )

        total_segments += len(speaker_rows)
        total_speakers += n_clusters

    elapsed = time.time() - t0
    log.info(
        f"speaker.cluster DONE: {len(all_sources)} source(s), "
        f"{total_segments} segments -> {total_speakers} estimated speakers, "
        f"{elapsed:.0f}s, run_id={run_id}"
    )
    return {
        "sources_processed": len(all_sources),
        "total_segments": total_segments,
        "total_speakers": total_speakers,
        "run_id": run_id,
    }


# ===========================================================================
# embed.backfill -- one-time (2026-07-12, docs/IO_OPTIMIZATION_PLAN.md Phase 3)
# ===========================================================================

BACKFILL_DISCOVER_SQL = """
    SELECT id, embedding_ref
    FROM speaker_embeddings
    WHERE embedding_ref IS NOT NULL AND embedding IS NULL
    ORDER BY source, id
"""


def discover_backfill(conn) -> list[tuple[str, str]]:
    """Return (id, embedding_ref) for every speaker_embeddings row that has a
    sidecar .npy reference but no in-table embedding value yet. Naturally
    idempotent: a row drops out of this WHERE clause the moment it's
    backfilled, so a killed/resumed run never redoes work."""
    return conn.execute(BACKFILL_DISCOVER_SQL).fetchall()


async def run_embed_backfill(*, conn=None, limit: int | None = None,
                              batch_size: int = 5000) -> dict:
    """One-time migration: stream every existing .embed.npy sidecar's content
    into speaker_embeddings.embedding, so speaker.cluster (and any future
    reader) never has to touch the per-file sidecar again for these rows.

    This is the LAST time these particular files are read the slow way (one
    open() per segment on the flat segments/{source}/ directories) -- after
    this, speaker.cluster's discovery query reads the `embedding` column
    directly. embedding_ref is left untouched (not cleared) so the mapping to
    the physical file survives for the owner-approved sidecar-deletion step
    that follows, once this backfill is verified.

    Uses a raw parameterised UPDATE (not upsert_rows' INSERT OR REPLACE,
    which would require re-supplying source/embedding_ref/provenance to avoid
    nulling them out -- see catalog.py's upsert_rows docstring) so only the
    `embedding` column is touched per row.

    conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many`. Defaults to a fresh
    self-managed connect() for standalone `pipe run embed.backfill` usage.
    """
    from pipeline.catalog.catalog import connect
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()
    log.info("embed.backfill: scanning speaker_embeddings for un-backfilled rows...")
    rows = discover_backfill(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"embed.backfill: {len(rows)} embedding(s) to migrate into the catalog column")
    if not rows:
        return {"scanned": 0, "backfilled": 0, "errors": 0}

    run_id = new_run_id("embed.backfill")
    io_workers = min(32, (os.cpu_count() or 4) * 4)

    scanned = 0
    backfilled = 0
    errors = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        for batch in _batches(rows, batch_size):
            loaded = list(pool.map(_load_npy, batch))
            update_batch = [
                (arr.astype(np.float32).tolist(), seg_id)
                for seg_id, _ref, arr in loaded if arr is not None
            ]
            failed_ids = [seg_id for seg_id, _ref, arr in loaded if arr is None]
            scanned += len(batch)
            if update_batch:
                conn.executemany(
                    "UPDATE speaker_embeddings SET embedding = ? WHERE id = ?",
                    update_batch,
                )
                backfilled += len(update_batch)
                record_batch(conn, run_id, "embed.backfill",
                              [pair[1] for pair in update_batch], "ok")
            if failed_ids:
                errors += len(failed_ids)
                log.warning(f"embed.backfill: {len(failed_ids)} sidecar(s) failed to load "
                            f"(left for the existing .npy fallback path): {failed_ids[:5]}")
                record_batch(conn, run_id, "embed.backfill", failed_ids,
                              "error", error="npy load failed")
            rate = scanned / (time.time() - t0) if time.time() > t0 else 0.0
            log.info(f"embed.backfill: {scanned}/{len(rows)} ({rate:.1f}/s) -- "
                     f"backfilled={backfilled}, errors={errors}")

    elapsed = time.time() - t0
    log.info(
        f"embed.backfill DONE: {scanned} scanned, {backfilled} backfilled, "
        f"{errors} errors, {elapsed:.0f}s, run_id={run_id}"
    )
    return {"scanned": scanned, "backfilled": backfilled, "errors": errors, "run_id": run_id}


# ---------------------------------------------------------------------------
# Subprocess entry point (speaker.embed GPU worker only;
# speaker.cluster has no subprocess)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    worker_main()
