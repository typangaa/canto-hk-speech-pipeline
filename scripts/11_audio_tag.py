#!/usr/bin/env python3
"""
scripts/11_audio_tag.py
Stage 11 — audio tagging (quality enrichment, TTS-focused).

DETECTOR A — Music / jingle  [this pass]
  PANNs CNN14 (AudioSet, 527 tags) → per-segment music-family probability.
  News intros/outros, podcast jingles, and background-music beds are the biggest
  remaining quality lever for TTS: background music teaches the model musical
  artifacts. We TAG, we do NOT delete — the downstream TTS-subset builder decides.

Non-destructive & additive: never touches the audio or the manifest rows. Writes a
sidecar metadata/audio_tags.jsonl (id-keyed, resumable). 11b_merge_tags.py folds the
tags back into the manifest later.

Usage:
  python scripts/11_audio_tag.py --source all
  python scripts/11_audio_tag.py --source youtube --shard 0/4 --device cuda:1
  python scripts/11_audio_tag.py --source rthk --limit 150 --out metadata/tag_calib.jsonl
      (calibration sample: small stratified run to set the music threshold by ear)

Output rows (metadata/audio_tags.jsonl):
  {"id": "...", "source": "youtube", "duration_sec": 8.1,
   "music_prob": 0.07, "music_tags": [["Speech", 0.93], ["Music", 0.07], ...]}
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from scipy.signal import firwin, upfirdn
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("11_audio_tag")

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "metadata" / "manifest.jsonl"
DEFAULT_OUT = REPO / "metadata" / "audio_tags.jsonl"

PANNS_SR = 32000  # CNN14 native sample rate

# Precomputed polyphase resample filter for the common 48k->32k (up 2 / down 3).
# Designing the Kaiser FIR once (instead of per-clip inside resample_poly) removes
# the firwin/Bessel cost from the hot path; upfirdn then just convolves.
_RS_CACHE: dict = {}


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

# --- Music-family AudioSet labels --------------------------------------------
# Built from a keyword include over the 527 labels, minus false positives that
# are NOT background music for our purposes (synthesised speech, animal calls,
# generic sound effects). music_prob = max prob over these indices.
_INCLUDE_KW = (
    "music", "jingle", "singing", "song", "choir", "rapping", "melody", "tune",
    "instrument", "guitar", "piano", "drum", "orchestr", "violin", "trumpet",
    "harmonica", "accordion", "synthesizer", "bass", "cello", "flute", "saxophone",
    "organ", "banjo", "mandolin", "harp", "trombone", "brass", "wind instrument",
    "percussion", "cymbal", "gong", "string", "keyboard (musical)", "theme",
)
_EXCLUDE_EXACT = {
    "Speech synthesizer",                       # TTS, not background music
    "Bird vocalization, bird call, bird song",  # animal, not music
}


def music_indices(labels):
    idx = []
    for i, lab in enumerate(labels):
        if lab in _EXCLUDE_EXACT:
            continue
        low = lab.lower()
        if any(k in low for k in _INCLUDE_KW):
            idx.append(i)
    return idx


def load_done_ids(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    continue
    return done


def load_manifest(source: str):
    rows = []
    with MANIFEST.open() as f:
        for line in f:
            d = json.loads(line)
            if source != "all" and d.get("source") != source:
                continue
            rows.append((d["id"], d.get("source", ""), d["audio_path"],
                         float(d.get("duration_sec", 0.0))))
    return rows


def read_audio_32k(path: str) -> np.ndarray | None:
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:
        log.warning(f"read fail {path}: {e}")
        return None
    if y.ndim > 1:
        y = y.mean(axis=1)
    return _resample(y, sr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all",
                    choices=["rthk", "youtube", "podcast", "hktv", "all"])
    ap.add_argument("--device", default="cuda:0",
                    help="cuda:0 / cuda:1 / cpu. GPUs may be busy with training — "
                         "pick the idle one or cpu.")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard", default=None, help="n/m, e.g. 0/4 — process shard n of m")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N (after stratified shuffle) — for calibration")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads cap — keep low so the trainer keeps cores")
    ap.add_argument("--io-workers", type=int, default=6,
                    help="threads prefetching read+resample (GIL-released) to overlap GPU")
    args = ap.parse_args()

    # keep CPU footprint small so the co-running trainer's dataloader keeps its cores
    torch.set_num_threads(args.threads)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA not available — falling back to cpu")
        device = "cpu"

    rows = load_manifest(args.source)
    log.info(f"manifest rows for source={args.source}: {len(rows)}")

    if args.limit:
        # deterministic stratified-ish sample: stride across the (source-ordered) list
        stride = max(1, len(rows) // args.limit)
        rows = rows[::stride][: args.limit]
        log.info(f"calibration sample: {len(rows)} rows (stride {stride})")

    if args.shard:
        n, m = (int(x) for x in args.shard.split("/"))
        rows = rows[n::m]
        log.info(f"shard {n}/{m}: {len(rows)} rows")

    done = load_done_ids(out_path)
    if done:
        before = len(rows)
        rows = [r for r in rows if r[0] not in done]
        log.info(f"resume: {len(done)} already tagged, {before - len(rows)} skipped, "
                 f"{len(rows)} remaining")
    if not rows:
        log.info("nothing to do")
        return

    # length-bucketing: sort by duration so each fixed-size batch holds near-equal
    # lengths → almost no zero-padding waste on the (training-shared) GPU.
    rows.sort(key=lambda r: r[3])

    log.info(f"loading PANNs CNN14 on {device} ...")
    from panns_inference import AudioTagging
    from panns_inference.config import labels
    at = AudioTagging(checkpoint_path=None, device=device)
    mus_idx = np.array(music_indices(labels))
    log.info(f"music-family labels: {len(mus_idx)}")

    fout = out_path.open("a")
    t0 = time.time()
    n_done = 0
    buf_ids, buf_meta, buf_wavs = [], [], []

    def flush_batch():
        nonlocal n_done
        if not buf_wavs:
            return
        maxlen = max(len(w) for w in buf_wavs)
        batch = np.zeros((len(buf_wavs), maxlen), dtype=np.float32)
        for i, w in enumerate(buf_wavs):
            batch[i, : len(w)] = w
        clip, _ = at.inference(batch)  # (N, 527)
        for i, (sid, (src, dur)) in enumerate(zip(buf_ids, buf_meta)):
            probs = clip[i]
            mprob = float(probs[mus_idx].max())
            top3 = np.argsort(probs)[-3:][::-1]
            tags = [[labels[j], round(float(probs[j]), 4)] for j in top3]
            fout.write(json.dumps({
                "id": sid, "source": src, "duration_sec": round(dur, 3),
                "music_prob": round(mprob, 4), "music_tags": tags,
            }, ensure_ascii=False) + "\n")
            n_done += 1
        buf_ids.clear(); buf_meta.clear(); buf_wavs.clear()

    # Prefetch: read+resample run in a thread pool (sf.read & scipy upfirdn release
    # the GIL → real parallelism) and OVERLAP with GPU inference. A bounded sliding
    # window keeps `depth` reads in flight so the GPU never waits on disk/resample.
    def prefetch(rows, ex, depth):
        q = deque()
        it = iter(rows)
        for _ in range(depth):
            try:
                r = next(it)
            except StopIteration:
                break
            q.append((r, ex.submit(read_audio_32k, r[2])))
        while q:
            r, fut = q.popleft()
            try:
                nr = next(it)
                q.append((nr, ex.submit(read_audio_32k, nr[2])))
            except StopIteration:
                pass
            yield r, fut.result()

    with ThreadPoolExecutor(max_workers=args.io_workers) as ex:
        for (sid, src, path, dur), y in prefetch(rows, ex, args.io_workers * 3):
            if y is None or len(y) < PANNS_SR // 10:  # <0.1s → skip
                continue
            buf_ids.append(sid); buf_meta.append((src, dur)); buf_wavs.append(y)
            if len(buf_wavs) >= args.batch:
                flush_batch()
            if n_done and n_done % args.flush_every < args.batch:
                fout.flush()
                rate = n_done / (time.time() - t0)
                log.info(f"{n_done} tagged ({rate:.1f}/s)")
    flush_batch()
    fout.flush()
    fout.close()
    log.info(f"DONE: {n_done} tagged in {time.time()-t0:.0f}s → {out_path}")


if __name__ == "__main__":
    main()
