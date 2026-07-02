#!/usr/bin/env python3
"""
scripts/12_language_id.py
Stage 12 — spoken-language ID (Cantonese vs Mandarin), quality enrichment.

WHY acoustic LID (not text): Stage 6 already has a text-based mandarin_ratio gate,
but it MISSES Mandarin audio that the Cantonese ASR model transcribed into
Cantonese-looking orthography (the ASR "washes out" the dialect). Wrong-language
audio pollutes the phonetic embedding space and must be excluded even from Stage-1
pretraining (per TTS data best practice). We detect language acoustically from the
waveform, which is robust to the ASR text bias.

Model: facebook/mms-lid-126 (Meta MMS, wav2vec2). Has SEPARATE `yue` (Cantonese)
and `cmn` (Mandarin) classes. Validated on known clips: Mandarin→cmn 0.999,
Cantonese→yue 1.0. Chosen over the tiantiaf voxlect specialist (broken config) and
VoxLingua107/Whisper (no separate Cantonese class / Mandarin bias).

Non-destructive & additive: writes a sidecar metadata/lang_id.jsonl (id-keyed,
resumable). 11b_merge_tags.py folds it into the manifest later.

Usage:
  python scripts/12_language_id.py --source all --device cuda:1
  python scripts/12_language_id.py --source podcast --shard 0/4 --device cpu
  python scripts/12_language_id.py --source all --limit 200 --out metadata/lang_calib.jsonl

Output rows (metadata/lang_id.jsonl):
  {"id": "...", "source": "podcast", "duration_sec": 8.1,
   "lang": "yue", "lang_prob": 0.998, "yue_prob": 0.998, "cmn_prob": 0.001,
   "top3": [["yue", 0.998], ["cmn", 0.001], ["vie", 0.0]]}
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("12_language_id")

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "metadata" / "manifest.jsonl"
DEFAULT_OUT = REPO / "metadata" / "lang_id.jsonl"

MODEL_ID = "facebook/mms-lid-126"
LID_SR = 16000


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


def read_audio_16k(path: str) -> np.ndarray | None:
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:
        log.warning(f"read fail {path}: {e}")
        return None
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != LID_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=LID_SR)
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all",
                    choices=["rthk", "youtube", "podcast", "hktv", "all"])
    ap.add_argument("--device", default="cuda:0", help="cuda:0 / cuda:1 / cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard", default=None, help="n/m, e.g. 0/4")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N (stratified stride) — for calibration")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--mem-fraction", type=float, default=None,
                    help="Cap this process's GPU memory fraction (e.g. 0.15) to protect "
                         "a co-running training job from OOM.")
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

    log.info(f"loading {MODEL_ID} on {device} ...")
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
    if device.startswith("cuda") and args.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device=device)
        log.info(f"GPU mem fraction capped at {args.mem_fraction}")
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()
    use_fp16 = device.startswith("cuda")
    if use_fp16:
        model = model.half()  # halve VRAM (~1.3GB→0.65GB weights) to protect co-running training
        log.info("model in fp16")
    id2label = model.config.id2label
    yue_i = next(i for i, l in id2label.items() if l == "yue")
    cmn_i = next(i for i, l in id2label.items() if l == "cmn")

    fout = out_path.open("a")
    t0 = time.time()
    n_done = 0
    buf_ids, buf_meta, buf_wavs = [], [], []

    def infer(wavs):
        """Forward a list of waveforms → (N, n_lang) probs. On CUDA OOM, free cache
        and recurse on halves (down to 1) so a transient training memory spike never
        crashes this job or drops data — it just backs off."""
        try:
            inp = fe(wavs, sampling_rate=LID_SR, return_tensors="pt", padding=True)
            inp = {k: (v.half() if (use_fp16 and v.is_floating_point()) else v).to(device)
                   for k, v in inp.items()}
            with torch.no_grad():
                logits = model(**inp).logits
            return torch.softmax(logits.float(), dim=-1).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(wavs) == 1:
                log.warning("OOM on single clip — retrying after cache clear")
                inp = fe(wavs, sampling_rate=LID_SR, return_tensors="pt", padding=True)
                inp = {k: (v.half() if (use_fp16 and v.is_floating_point()) else v).to(device)
                       for k, v in inp.items()}
                with torch.no_grad():
                    logits = model(**inp).logits
                return torch.softmax(logits.float(), dim=-1).cpu().numpy()
            mid = len(wavs) // 2
            log.warning(f"OOM on batch {len(wavs)} — splitting (training spike?)")
            return np.concatenate([infer(wavs[:mid]), infer(wavs[mid:])], axis=0)

    def flush_batch():
        nonlocal n_done
        if not buf_wavs:
            return
        probs = infer(buf_wavs)  # (N, n_lang)
        for i, (sid, (src, dur)) in enumerate(zip(buf_ids, buf_meta)):
            p = probs[i]
            top3i = np.argsort(p)[-3:][::-1]
            top3 = [[id2label[int(j)], round(float(p[j]), 4)] for j in top3i]
            arg = int(np.argmax(p))
            fout.write(json.dumps({
                "id": sid, "source": src, "duration_sec": round(dur, 3),
                "lang": id2label[arg], "lang_prob": round(float(p[arg]), 4),
                "yue_prob": round(float(p[yue_i]), 4),
                "cmn_prob": round(float(p[cmn_i]), 4),
                "top3": top3,
            }, ensure_ascii=False) + "\n")
            n_done += 1
        buf_ids.clear(); buf_meta.clear(); buf_wavs.clear()

    for sid, src, path, dur in rows:
        y = read_audio_16k(path)
        if y is None or len(y) < LID_SR // 10:
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
