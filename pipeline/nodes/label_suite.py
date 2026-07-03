"""
pipeline/nodes/label_suite.py
label.suite DAG node — P2 "Mode A" in-process fan-out: one GPU worker hosts
mms-lid (language ID) + pyannote/segmentation-3.0 (overlapped-speech detection) +
PANNs CNN14 (music-family tagging), sharing a SINGLE decode-once read per segment
(pipeline/audio/bus.py) instead of each detector independently reading+resampling
the same file (the B1 bottleneck named in REARCHITECTURE_IMPLEMENTATION_PLAN.md §1.3).

Since label.lang (labels_lang) and label.overlap (labels_overlap) were already
~100% completed by the P0 legacy import, this node is NOT a redo-everything pass —
discover() only asks for whichever of {lang, overlap, music} a given segment is
still missing, so in practice most rows only need music (the 344,727-segment
backlog from P1) while a handful of legacy-import gaps (~24 each) get lang/overlap
filled in alongside it for free, in the same decode.

Per-item label need is heterogeneous (item A might need only music, item B might
need lang+overlap), so forward_batch() partitions the batch per detector: mms-lid
and PANNs batch normally (padded tensor, one forward pass per detector per batch);
pyannote's Inference(window="whole") has no cross-file batching API, so overlap
detection stays a per-item loop even inside this shared-decode worker — the value
here is that it reuses the SAME 16k array mms-lid already decoded, not that it runs
faster than scripts/13_overlap_detect.py per se.

An `emotion` slot is deliberately NOT wired in yet: LABEL_FRAMEWORK_SPEC.md §8.4 and
§12.1 require a Cantonese emotion2vec spot-check (owner must listen to ~100 clips)
before that detector is trusted — this node's discover()/worker plumbing is built to
make adding it a small diff once the gate passes, not a rewrite.

Discovery: segments missing labels_lang OR labels_overlap OR labels_music (3-way
LEFT JOIN anti-join, analogous to label_music.py's single-table version).
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
import torch

from pipeline.audio.bus import decode_multi
from pipeline.workers.gpu_base import GPUWorkerBase

log = logging.getLogger(__name__)

LID_MODEL_ID = "facebook/mms-lid-126"
OSD_MODEL_ID = "pyannote/segmentation-3.0"
SHARED_SR = 16000       # mms-lid + pyannote/segmentation-3.0 both expect 16 kHz
PANNS_SR = 32000        # PANNs CNN14 native rate
ACTIVE_THR = 0.5        # pyannote per-speaker activity threshold (matches scripts/13)

# Same music-family taxonomy as scripts/11_audio_tag.py / pipeline/nodes/label_music.py
# — kept byte-identical so label_suite's music_prob is comparable to existing rows.
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


# ---------------------------------------------------------------------------
# Catalog discovery (supervisor side)
# ---------------------------------------------------------------------------

DISCOVER_SQL = """
    SELECT s.id, s.source, s.audio_path, s.duration_sec,
           (l.id IS NULL) AS need_lang,
           (o.id IS NULL) AS need_overlap,
           (m.id IS NULL) AS need_music
    FROM segments s
    LEFT JOIN labels_lang l ON s.id = l.id
    LEFT JOIN labels_overlap o ON s.id = o.id
    LEFT JOIN labels_music m ON s.id = m.id
    WHERE l.id IS NULL OR o.id IS NULL OR m.id IS NULL
    ORDER BY s.duration_sec
"""


def discover(conn) -> list[tuple]:
    return conn.execute(DISCOVER_SQL).fetchall()


# ---------------------------------------------------------------------------
# Supervisor: pool + sampler + worker-protocol wiring (P2 — same shape as
# pipeline/nodes/label_music.py's run_label_music, generalised to 3 output tables)
# ---------------------------------------------------------------------------

def _batches(rows: list[tuple], size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


async def run_label_suite(
    devices: list[str],
    *,
    gpu_policy: str = "cap",
    batch_size: int = 16,
    mem_fraction: float | None = 0.15,
    limit: int | None = None,
) -> dict:
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch
    from pipeline.orchestrator.pools import PoolRegistry
    from pipeline.orchestrator.resources import GpuPolicy, Sampler
    from pipeline.orchestrator.worker import spawn_worker

    conn = connect()
    rows = discover(conn)
    if limit:
        rows = rows[:limit]
    log.info(f"label.suite: {len(rows)} segments with at least one missing label")
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
        cmd = [sys.executable, "-m", "pipeline.nodes.label_suite", "--device", dev]
        if mem_fraction is not None and dev.startswith("cuda"):
            cmd += ["--mem-fraction", str(mem_fraction)]
        handle = await spawn_worker(cmd)
        await handle.wait_ready(timeout=180.0)  # 3 models to load — longer than label_music
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

    run_id = new_run_id("label.suite")
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
            meta = {r[0]: (r[1], r[3]) for r in batch}  # id -> (source, duration_sec)
            items = [
                {
                    "id": r[0], "path": r[2], "duration_sec": r[3],
                    "need_lang": bool(r[4]), "need_overlap": bool(r[5]), "need_music": bool(r[6]),
                }
                for r in batch
            ]
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

            lang_rows, overlap_rows, music_rows = [], [], []
            for r in result["rows"]:
                sid = r["id"]
                source, duration_sec = meta[sid]
                if r.get("lang") is not None:
                    lang_rows.append({
                        "id": sid, "source": source, "duration_sec": duration_sec,
                        **r["lang"],
                    })
                if r.get("overlap") is not None:
                    overlap_rows.append({
                        "id": sid, "source": source, "duration_sec": duration_sec,
                        **r["overlap"],
                    })
                if r.get("music") is not None:
                    music_rows.append({
                        "id": sid, "source": source, "duration_sec": duration_sec,
                        "music_prob": r["music"]["music_prob"],
                        "music_tags": r["music"]["music_tags"],
                        "provenance": "p2_suite",
                    })

            # Unreadable files: mark whichever labels this id still needed as a
            # (mostly-null) row so discover()'s anti-join stops resurfacing it —
            # same rationale as label_music.py's read_failed handling. labels_lang
            # / labels_overlap have no provenance column (P0 schema), so this is a
            # plain null-value row for those two, and a provenance-tagged one for
            # labels_music (which does have that column).
            skipped = result.get("skipped_ids", [])
            if skipped:
                by_id = {it["id"]: it for it in items}
                for sid in skipped:
                    src, dur = meta[sid]
                    it = by_id[sid]
                    if it["need_lang"]:
                        lang_rows.append({"id": sid, "source": src, "duration_sec": dur,
                                           "lang": None, "lang_prob": None,
                                           "yue_prob": None, "cmn_prob": None, "top3": []})
                    if it["need_overlap"]:
                        overlap_rows.append({"id": sid, "source": src, "duration_sec": dur,
                                              "overlap_ratio": None, "overlap_sec": None,
                                              "speech_ratio": None})
                    if it["need_music"]:
                        music_rows.append({"id": sid, "source": src, "duration_sec": dur,
                                            "music_prob": None, "music_tags": [],
                                            "provenance": "read_failed"})
                log.warning(f"{pool_name}: {len(skipped)} unreadable segment(s): {skipped[:5]}")

            if lang_rows:
                upsert_rows(conn, "labels_lang", lang_rows, ["id"])
            if overlap_rows:
                upsert_rows(conn, "labels_overlap", overlap_rows, ["id"])
            if music_rows:
                upsert_rows(conn, "labels_music", music_rows, ["id"])

            all_ids = [r["id"] for r in result["rows"]]
            record_batch(conn, run_id, "label.suite", all_ids, "ok",
                         metrics=result.get("metrics"))
            if skipped:
                record_batch(conn, run_id, "label.suite", skipped, "error",
                             error="unreadable audio file")

            processed += len(all_ids) + len(skipped)
            errors += len(skipped)
            queue.task_done()
            if processed and processed % (batch_size * 20) < batch_size:
                rate = processed / (time.time() - t0)
                log.info(f"{processed}/{len(rows)} processed ({rate:.1f}/s), "
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

class SuiteWorker(GPUWorkerBase):
    def load_model(self):
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        log.info(f"loading {LID_MODEL_ID} on {self.device} ...")
        self.lid_fe = AutoFeatureExtractor.from_pretrained(LID_MODEL_ID)
        lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(LID_MODEL_ID)
        lid_model = lid_model.to(self.device).eval()
        if self.use_fp16:
            lid_model = lid_model.half()
        self.lid_model = lid_model
        self.lid_id2label = lid_model.config.id2label
        self.lid_yue_i = next(i for i, l in self.lid_id2label.items() if l == "yue")
        self.lid_cmn_i = next(i for i, l in self.lid_id2label.items() if l == "cmn")

        log.info(f"loading {OSD_MODEL_ID} on {self.device} ...")
        from pyannote.audio import Inference, Model
        osd_model = Model.from_pretrained(OSD_MODEL_ID)
        self.osd_inf = Inference(osd_model, window="whole", device=torch.device(self.device))

        log.info(f"loading PANNs CNN14 on {self.device} ...")
        from panns_inference import AudioTagging
        from panns_inference.config import labels as panns_labels
        # panns_inference.AudioTagging does an *exact* string match `device == 'cuda'`
        # to decide GPU vs CPU — "cuda:0"/"cuda:1" silently fall through to CPU.
        # set_device() pins the process's default GPU so unqualified "cuda" resolves
        # to the right physical card.
        panns_device = "cuda" if str(self.device).startswith("cuda") else "cpu"
        if panns_device == "cuda":
            torch.cuda.set_device(self.device)
        real_stdout = sys.stdout
        sys.stdout = sys.stderr  # panns_inference prints to stdout on init — would
        try:                     # corrupt the JSONL worker protocol stream.
            self.panns = AudioTagging(checkpoint_path=None, device=panns_device)
        finally:
            sys.stdout = real_stdout
        if isinstance(self.panns.model, torch.nn.DataParallel):
            # See label_music.py's identical fix: AudioTagging's DataParallel wrap
            # assumes cuda:0 regardless of which GPU set_device() actually pinned —
            # unwrap since we want one GPU per worker process, not data splitting.
            self.panns.model = self.panns.model.module
        self.panns_labels = panns_labels
        self.music_idx = np.array(music_indices(panns_labels))

        return None  # per-detector attrs are used directly; no single "self.model"

    # -- per-detector batch helpers -----------------------------------------

    def _lid_infer(self, wavs: list[np.ndarray]) -> np.ndarray:
        inp = self.lid_fe(wavs, sampling_rate=SHARED_SR, return_tensors="pt", padding=True)
        inp = {k: (v.half() if (self.use_fp16 and v.is_floating_point()) else v).to(self.device)
               for k, v in inp.items()}
        with torch.no_grad():
            logits = self.lid_model(**inp).logits
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()

    def _panns_infer(self, wavs: list[np.ndarray]) -> list[dict]:
        maxlen = max(len(w) for w in wavs)
        batch = np.zeros((len(wavs), maxlen), dtype=np.float32)
        for i, w in enumerate(wavs):
            batch[i, : len(w)] = w
        clip, _ = self.panns.inference(batch)  # (N, 527)
        rows = []
        for probs in clip:
            mprob = float(probs[self.music_idx].max())
            top3 = np.argsort(probs)[-3:][::-1]
            tags = [[self.panns_labels[j], round(float(probs[j]), 4)] for j in top3]
            rows.append({"music_prob": round(mprob, 4), "music_tags": tags})
        return rows

    def _osd_infer(self, y16: np.ndarray, duration_sec: float) -> dict:
        waveform = torch.from_numpy(y16).float().unsqueeze(0)
        d = np.asarray(self.osd_inf({"waveform": waveform, "sample_rate": SHARED_SR}))
        if d.ndim == 3:
            d = d[0]
        frames = d.shape[0]
        if frames == 0 or duration_sec <= 0:
            return {"overlap_ratio": 0.0, "overlap_sec": 0.0, "speech_ratio": 0.0}
        step = duration_sec / frames
        active = (d > ACTIVE_THR).sum(axis=-1)
        overlap_sec = float((active >= 2).sum() * step)
        speech_sec = float((active >= 1).sum() * step)
        return {
            "overlap_ratio": round(overlap_sec / duration_sec, 4),
            "overlap_sec": round(overlap_sec, 3),
            "speech_ratio": round(speech_sec / duration_sec, 4),
        }

    # -- fan-out --------------------------------------------------------------

    def forward_batch(self, items: list[dict]) -> list[dict]:
        """items: dicts with y16/y32 (as available) + need_lang/need_overlap/need_music.
        Returns one {"id","lang","overlap","music"} dict per item, same order —
        each of lang/overlap/music is None where that item didn't need it.
        """
        n = len(items)
        lang_out: list[dict | None] = [None] * n
        overlap_out: list[dict | None] = [None] * n
        music_out: list[dict | None] = [None] * n

        lid_idx = [i for i, it in enumerate(items) if it["need_lang"]]
        if lid_idx:
            probs = self._lid_infer([items[i]["y16"] for i in lid_idx])
            for k, i in enumerate(lid_idx):
                p = probs[k]
                arg = int(np.argmax(p))
                top3i = np.argsort(p)[-3:][::-1]
                lang_out[i] = {
                    "lang": self.lid_id2label[arg], "lang_prob": round(float(p[arg]), 4),
                    "yue_prob": round(float(p[self.lid_yue_i]), 4),
                    "cmn_prob": round(float(p[self.lid_cmn_i]), 4),
                    "top3": [[self.lid_id2label[int(j)], round(float(p[j]), 4)] for j in top3i],
                }

        music_idx_list = [i for i, it in enumerate(items) if it["need_music"]]
        if music_idx_list:
            rows = self._panns_infer([items[i]["y32"] for i in music_idx_list])
            for k, i in enumerate(music_idx_list):
                music_out[i] = rows[k]

        for i, it in enumerate(items):
            if it["need_overlap"]:
                overlap_out[i] = self._osd_infer(it["y16"], it["duration_sec"])

        return [
            {"id": items[i]["id"], "lang": lang_out[i], "overlap": overlap_out[i],
             "music": music_out[i]}
            for i in range(n)
        ]


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

    worker = SuiteWorker(args.device, mem_fraction=args.mem_fraction, fp16=args.fp16)

    def emit(msg: dict) -> None:
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    emit({"type": "ready", "node": "label.suite", "pid": __import__("os").getpid(), "proto": 1})

    ex = ThreadPoolExecutor(max_workers=args.io_workers)

    def prep(it: dict) -> tuple[dict, dict | None]:
        srs = []
        if it["need_lang"] or it["need_overlap"]:
            srs.append(SHARED_SR)
        if it["need_music"]:
            srs.append(PANNS_SR)
        arrs = decode_multi(it["path"], srs) if srs else {}
        return it, arrs

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
            kept_items = []
            skipped_ids = []
            for it, arrs in preps:
                if arrs is None:
                    skipped_ids.append(it["id"])
                    continue
                kept_items.append({
                    **it,
                    "y16": arrs.get(SHARED_SR),
                    "y32": arrs.get(PANNS_SR),
                })
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
