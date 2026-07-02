#!/usr/bin/env python3
"""
scripts/13_overlap_detect.py
Stage 13 — overlapped-speech detection (OSD), quality enrichment.

WHY: TTS maps one text stream → one vocal tract. Two people talking at once breaks
the text↔audio alignment, which is FATAL at any training stage (cannot be fixed by
clean fine-tuning, unlike background music). Stage 3 diarization already rejects
overlap *turns*, but short residual overlaps / backchannels slip through (e.g. the
podcast clip the language-ID model misread as 'tha' was actually two speakers at
once). This detects them at the segment level so they can be excluded even from
Stage-1 pretraining.

Model: pyannote/segmentation-3.0 (powerset, ≤2 speakers/frame). Run in window="whole"
mode → per-frame speaker activity over the whole clip; overlap = frames with ≥2
speakers active. Validated: known-overlap clip → ratio 0.39, clean clips → 0.00.

Non-destructive & additive: writes a sidecar metadata/overlap.jsonl (id-keyed,
resumable). 11b_merge_tags.py folds it into the manifest later.

Usage:
  python scripts/13_overlap_detect.py --source all --device cuda:0 --shard 0/2
  python scripts/13_overlap_detect.py --source all --limit 200 --out metadata/overlap_calib.jsonl

Output rows (metadata/overlap.jsonl):
  {"id": "...", "source": "podcast", "duration_sec": 4.69,
   "overlap_ratio": 0.389, "overlap_sec": 1.83, "speech_ratio": 1.0}
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("13_overlap")

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "metadata" / "manifest.jsonl"
DEFAULT_OUT = REPO / "metadata" / "overlap.jsonl"

MODEL_ID = "pyannote/segmentation-3.0"
ACTIVE_THR = 0.5  # per-speaker activity threshold


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all",
                    choices=["rthk", "youtube", "podcast", "hktv", "all"])
    ap.add_argument("--device", default="cuda:0", help="cuda:0 / cuda:1 / cpu")
    ap.add_argument("--shard", default=None, help="n/m, e.g. 0/2")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N (stratified stride) — for calibration")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--mem-fraction", type=float, default=None,
                    help="Cap this process's GPU memory fraction to protect co-running jobs.")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA not available — falling back to cpu")
        device = "cpu"

    rows = load_manifest(args.source)
    log.info(f"manifest rows for source={args.source}: {len(rows)}")

    if args.limit:
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

    if device.startswith("cuda") and args.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device=device)
        log.info(f"GPU mem fraction capped at {args.mem_fraction}")

    log.info(f"loading {MODEL_ID} on {device} ...")
    from pyannote.audio import Model, Inference
    model = Model.from_pretrained(MODEL_ID)
    inf = Inference(model, window="whole", device=torch.device(device))

    def overlap_of(path: str, dur: float):
        d = np.asarray(inf(path))           # (frames, 3) speaker-activity probs
        if d.ndim == 3:
            d = d[0]
        frames = d.shape[0]
        if frames == 0:
            return 0.0, 0.0, 0.0
        step = dur / frames
        active = (d > ACTIVE_THR).sum(axis=-1)   # speakers active per frame
        overlap_sec = float((active >= 2).sum() * step)
        speech_sec = float((active >= 1).sum() * step)
        return overlap_sec, speech_sec, step

    fout = out_path.open("a")
    t0 = time.time()
    n_done = 0
    for sid, src, path, dur in rows:
        if dur <= 0:
            try:
                dur = sf.info(path).duration
            except Exception:
                continue
        try:
            overlap_sec, speech_sec, _ = overlap_of(path, dur)
        except Exception as exc:
            log.warning(f"overlap fail {Path(path).name}: {exc}")
            continue
        fout.write(json.dumps({
            "id": sid, "source": src, "duration_sec": round(dur, 3),
            "overlap_ratio": round(overlap_sec / dur, 4) if dur else 0.0,
            "overlap_sec": round(overlap_sec, 3),
            "speech_ratio": round(speech_sec / dur, 4) if dur else 0.0,
        }, ensure_ascii=False) + "\n")
        n_done += 1
        if n_done % args.flush_every == 0:
            fout.flush()
            rate = n_done / (time.time() - t0)
            log.info(f"{n_done} tagged ({rate:.1f}/s)")
    fout.flush()
    fout.close()
    log.info(f"DONE: {n_done} tagged in {time.time()-t0:.0f}s → {out_path}")


if __name__ == "__main__":
    main()
