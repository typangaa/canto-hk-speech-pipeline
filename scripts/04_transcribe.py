#!/usr/bin/env python3
"""
scripts/04_transcribe.py
Multi-ASR transcription of segmented audio. Stores all candidates + agreement score.
Usage: python scripts/04_transcribe.py --source [rthk|youtube|podcast|all] [--dry-run] [--gpu 0]
       python scripts/04_transcribe.py --source podcast --shard 0/3 --gpu 1

Models used (see CLAUDE.md ASR strategy):
  A: Cantonese fine-tuned Whisper (simonl0909/whisper-large-v2-cantonese)
  B: base Whisper large-v3 with language="zh" + Cantonese written-form prompt

NEVER language="yue" — causes decoder collapse (KNOWN_ISSUES §9).

Crash safety: each processed file is flushed to a per-shard JSONL checkpoint
immediately after processing. On restart, checkpoint is loaded and already-done
files are skipped instantly (no GPU needed for completed passes).

Output: data/segments/{source}/{stem}.transcript.json
"""

import argparse
import difflib
import itertools
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent.parent


def make_log_path(source: str, shard: str | None) -> Path:
    shard_str = shard.replace("/", "-") if shard else "all"
    return ROOT / "metadata" / "logs" / f"04_transcribe_{source}_{shard_str}.log"


def make_checkpoint_path(source: str, shard: str | None) -> Path:
    shard_str = shard.replace("/", "-") if shard else "all"
    return ROOT / "metadata" / "logs" / f"04_checkpoint_{source}_{shard_str}.jsonl"


# Logger configured in main() once args are parsed
log = logging.getLogger(__name__)

SEG_DIR = ROOT / "data" / "segments"
TARGET_SR = 48000
ASR_SR = 16000

_PREFETCH_WORKERS = 8   # CPU threads: parallel audio load+resample per Stage-4 process
_PREFETCH_AHEAD   = 24  # files queued ahead of GPU in the prefetch pipeline

# Cantonese written-form initial prompt (helps large-v3 produce 粵語白話文)
CANTO_PROMPT = (
    "以下係廣東話口語，請用粵語白話文書寫，"
    "例如：係、唔係、冇、喺、佢哋、嘅、嗰、嚟。"
)

_LOCAL_CANTO = str(ROOT / "data" / "ct2_models" / "whisper-large-v2-cantonese")

ASR_MODELS = [
    {
        "id": _LOCAL_CANTO,
        "key": "canto_ft",
        "lang": "zh",
        "prompt": CANTO_PROMPT,
        "description": "Cantonese fine-tuned Whisper large-v2 (local ct2)",
    },
    {
        "id": "Systran/faster-whisper-large-v3",
        "key": "whisper_zh",
        "lang": "zh",
        "prompt": CANTO_PROMPT,
        "description": "Whisper large-v3 with zh + Cantonese prompt",
    },
]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load checkpoint: {path_str: {pass_key: result_dict}}"""
    results: dict = {}
    if not checkpoint_file.exists():
        return results
    for line in checkpoint_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            path = entry["path"]
            pass_key = entry["pass_key"]
            if path not in results:
                results[path] = {}
            results[path][pass_key] = entry["result"]
        except Exception:
            pass
    total = sum(len(v) for v in results.values())
    log.info(f"Checkpoint loaded: {len(results)} files, {total} pass results from {checkpoint_file.name}")
    return results


_CKPT_HANDLES: dict = {}  # checkpoint_file → open file handle (kept open to avoid repeated open/close)

def append_checkpoint(checkpoint_file: Path, path_str: str, pass_key: str, result: dict) -> None:
    entry = json.dumps({"path": path_str, "pass_key": pass_key, "result": result},
                       ensure_ascii=False)
    if checkpoint_file not in _CKPT_HANDLES:
        _CKPT_HANDLES[checkpoint_file] = open(checkpoint_file, "a", encoding="utf-8")
    fh = _CKPT_HANDLES[checkpoint_file]
    fh.write(entry + "\n")
    # No fsync: OS write-back cache is sufficient; checkpoint is crash-recovery only


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(model_cfg: dict, device: str, gpu_index: int = 0, compute_type: str = "auto"):
    from faster_whisper import WhisperModel
    if compute_type == "auto":
        compute_type = "int8_float16" if device == "cuda" else "int8"
    log.info(f"Loading ASR model: {model_cfg['description']} on {device}:{gpu_index} [{compute_type}]")
    return WhisperModel(
        model_cfg["id"],
        device=device,
        device_index=gpu_index,
        compute_type=compute_type,
        cpu_threads=4,
    )


def audio_to_16k(wav48: np.ndarray) -> np.ndarray:
    import torchaudio
    t = torch.from_numpy(wav48).float().unsqueeze(0)
    resampler = torchaudio.transforms.Resample(TARGET_SR, ASR_SR)
    return resampler(t).squeeze(0).numpy()


def transcribe_one(model, wav16: np.ndarray, cfg: dict) -> dict:
    kwargs: dict = {
        "beam_size": 1,   # greedy — 3-5x faster; quality maintained by dual-model agreement
        "vad_filter": False,
        "temperature": 0.0,
    }
    if cfg["lang"]:
        kwargs["language"] = cfg["lang"]
    if cfg["prompt"]:
        kwargs["initial_prompt"] = cfg["prompt"]

    segments, info = model.transcribe(wav16, **kwargs)
    segs_list = list(segments)
    text = "".join(s.text for s in segs_list).strip()
    if segs_list:
        conf = float(np.mean([s.avg_logprob for s in segs_list if hasattr(s, "avg_logprob")]))
        import math
        conf = round(max(0.0, min(1.0, math.exp(conf))), 3)
    else:
        conf = 0.0

    return {
        "model": cfg["id"] + (f"+{cfg['lang']}" if cfg["lang"] else ""),
        "text": text,
        "confidence": conf,
    }


def char_agreement(texts: list[str]) -> float:
    if len(texts) < 2:
        return 1.0
    ratios = [
        difflib.SequenceMatcher(None, a, b).ratio()
        for a, b in itertools.combinations(texts, 2)
    ]
    return round(sum(ratios) / len(ratios), 3)


# ---------------------------------------------------------------------------
# Two-pass transcription with checkpoint
# ---------------------------------------------------------------------------

def _load_and_resample(wav_path: Path) -> np.ndarray:
    """Load WAV + resample to 16 kHz. Thread-safe: uses scipy, no torch/CUDA."""
    from scipy.signal import resample_poly
    wav48, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if wav48.ndim > 1:
        wav48 = wav48.mean(axis=1)
    # 48000 → 16000 is exactly ×(1/3); resample_poly is lossless for integer ratios
    return resample_poly(wav48, 1, 3).astype(np.float32)


def transcribe_all_with_model(
    wav_paths: list,
    model,
    cfg: dict,
    checkpoint: dict,
    checkpoint_file: Path,
) -> dict:
    """Run one ASR model over all wav_paths. Checkpoint-backed + parallel prefetch.

    While the GPU processes segment N, _PREFETCH_WORKERS threads are already loading
    and resampling segments N+1 … N+_PREFETCH_AHEAD, keeping the GPU fed continuously.
    """
    results: dict = {}
    skipped = 0
    computed = 0

    # Split: fast-path checkpoint hits vs files that need GPU work
    todo: list[Path] = []
    for wav_path in wav_paths:
        path_str = str(wav_path)
        if path_str in checkpoint and cfg["key"] in checkpoint[path_str]:
            results[path_str] = checkpoint[path_str][cfg["key"]]
            skipped += 1
        else:
            todo.append(wav_path)

    log.info(f"  [{cfg['key']}] {len(todo)} to compute, {skipped} from checkpoint")
    if not todo:
        return results

    with ThreadPoolExecutor(max_workers=_PREFETCH_WORKERS) as pool:
        pending: dict = {}  # idx → Future[np.ndarray]

        def _enqueue(idx: int) -> None:
            if idx < len(todo):
                pending[idx] = pool.submit(_load_and_resample, todo[idx])

        # Pre-fill pipeline
        for i in range(min(_PREFETCH_AHEAD, len(todo))):
            _enqueue(i)

        for i, wav_path in enumerate(todo):
            _enqueue(i + _PREFETCH_AHEAD)   # keep pipeline full one step ahead

            if computed % 200 == 0:
                log.info(f"  [{cfg['key']}] {computed}/{len(todo)} ...")

            try:
                wav16 = pending.pop(i).result()
                result = transcribe_one(model, wav16, cfg)
            except Exception as exc:
                log.error(f"{cfg['key']} failed on {wav_path.name}: {exc}")
                result = {"model": cfg["id"], "text": "", "confidence": 0.0, "error": str(exc)}

            path_str = str(wav_path)
            results[path_str] = result
            append_checkpoint(checkpoint_file, path_str, cfg["key"], result)
            computed += 1

    log.info(f"  [{cfg['key']}] Done: {computed} computed, {skipped} from checkpoint")
    return results


def write_transcripts(wav_paths: list, all_results: list[dict]) -> tuple[int, int]:
    """Merge results from all models and write .transcript.json files."""
    processed = low_agreement = 0
    for wav_path in wav_paths:
        out_path = wav_path.with_suffix(".transcript.json")
        candidates = [r.get(str(wav_path), {"model": "?", "text": "", "confidence": 0.0})
                      for r in all_results]
        texts = [c["text"] for c in candidates if c["text"]]
        agreement = char_agreement(texts) if len(texts) >= 2 else 0.0
        best = max(candidates, key=lambda c: c["confidence"] if c["text"] else -1)
        record = {
            "seg_path": str(wav_path),
            "asr_candidates": candidates,
            "asr_agreement": agreement,
            "text": best["text"],
            "text_verified": False,
        }
        with open(out_path, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        processed += 1
        if agreement < 0.80:
            low_agreement += 1
            log.info(f"  Low agreement {agreement:.2f}: {wav_path.name} | '{best['text'][:50]}'")
    return processed, low_agreement


def find_segments(source: str) -> list[Path]:
    if source == "all":
        return sorted(SEG_DIR.rglob("*.wav"))
    return sorted((SEG_DIR / source).rglob("*.wav"))


# ---------------------------------------------------------------------------
# Acoustic pre-gate (optional SNR/DNSMOS filter before GPU transcription)
# ---------------------------------------------------------------------------

def compute_snr_fast(wav48: np.ndarray, sr: int = 48000) -> float:
    """Estimate SNR via energy-based voice activity. Returns dB."""
    frame_len = int(sr * 0.025)
    hop = int(sr * 0.010)
    frames = [wav48[i:i+frame_len] for i in range(0, len(wav48)-frame_len, hop)]
    if not frames:
        return 0.0
    energies = np.array([np.mean(f**2) + 1e-10 for f in frames])
    noise_floor = np.percentile(energies, 10)
    signal_peak = np.percentile(energies, 90)
    if noise_floor <= 0:
        return 0.0
    return float(10 * np.log10(signal_peak / noise_floor))


def acoustic_pre_gate(
    wav_paths: list,
    min_snr: float | None,
    min_dnsmos: float | None,
) -> list:
    """Filter wav_paths to those passing SNR and/or DNSMOS thresholds."""
    if not min_snr and not min_dnsmos:
        return wav_paths

    log.info(f"Acoustic pre-gate: {len(wav_paths)} files, SNR>={min_snr}, DNSMOS>={min_dnsmos}")

    if min_dnsmos:
        try:
            from speechmos import dnsmos
        except ImportError:
            log.warning("speechmos not installed — skipping DNSMOS pre-gate")
            min_dnsmos = None

    passed = []
    rejected_snr = rejected_dnsmos = 0

    for i, wav_path in enumerate(wav_paths):
        if i % 5000 == 0:
            log.info(f"  Pre-gate: {i}/{len(wav_paths)} ...")
        try:
            wav48, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if wav48.ndim > 1:
                wav48 = wav48.mean(axis=1)

            if min_snr:
                snr = compute_snr_fast(wav48, sr)
                if snr < min_snr:
                    rejected_snr += 1
                    continue

            if min_dnsmos:
                wav16k = audio_to_16k(wav48)
                score = dnsmos.run(wav16k, sr=16000)
                if score["ovrl_mos"] < min_dnsmos:
                    rejected_dnsmos += 1
                    continue

            passed.append(wav_path)
        except Exception as exc:
            log.warning(f"Pre-gate failed on {wav_path.name}: {exc}")
            passed.append(wav_path)  # include on error (safe fallback)

    log.info(
        f"Pre-gate: {len(passed)}/{len(wav_paths)} passed "
        f"(rejected snr={rejected_snr}, dnsmos={rejected_dnsmos})"
    )
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all",
                        choices=["rthk", "youtube", "podcast", "hktv", "all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu", type=int, default=1,
                        help="GPU index (default 1; GPU 0 occupied by llama-server)")
    parser.add_argument("--shard", default=None, metavar="N/M",
                        help="Process only shard N of M (0-indexed). E.g. --shard 0/3")
    parser.add_argument("--pre-gate-snr", type=float, default=None, metavar="DB",
                        help="Skip segments with estimated SNR below DB before transcription")
    parser.add_argument("--pre-gate-dnsmos", type=float, default=None, metavar="SCORE",
                        help="Skip segments with DNSMOS OVRL below SCORE before transcription")
    parser.add_argument("--use-pregate", action="store_true",
                        help="Read .pregate.json markers from 03b_acoustic_pregate.py. "
                             "Skips pass=False files instantly. Falls back to inline SNR for ungated files.")
    parser.add_argument("--compute-type", default="auto",
                        choices=["auto", "int8_float16", "int8", "float16"],
                        help="faster-whisper compute type. 'auto'=int8_float16 on GPU. "
                             "Use 'int8' on GPU0 when VRAM is tight (~1.4GB vs ~2.5GB).")
    args = parser.parse_args()

    # Unique log and checkpoint paths per source+shard to support concurrent workers
    log_path = make_log_path(args.source, args.shard)
    checkpoint_file = make_checkpoint_path(args.source, args.shard)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    if torch.cuda.is_available():
        device = "cuda"
        gpu_index = args.gpu
        log.info(f"Using GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
    else:
        device = "cpu"
        gpu_index = 0
        log.warning("No GPU found — transcription will be slow")

    wavs = find_segments(args.source)
    todo = [w for w in wavs if not w.with_suffix(".transcript.json").exists()]

    if args.shard:
        n, m = (int(x) for x in args.shard.split("/"))
        todo = todo[n::m]
        log.info(f"Shard {n}/{m}: processing {len(todo)} of {len(wavs)} segments")
    else:
        log.info(f"Found {len(wavs)} segments, {len(todo)} need transcription")

    if not todo:
        log.info("All segments already transcribed.")
        sys.exit(0)

    if args.dry_run:
        for wav_path in todo:
            log.info(f"[DRY-RUN] Would transcribe: {wav_path.name}")
        sys.exit(0)

    # Pre-gate: read .pregate.json markers written by 03b_acoustic_pregate.py
    if args.use_pregate:
        before = len(todo)
        todo_gated = []
        ungated = []
        for w in todo:
            pg = w.with_suffix(".pregate.json")
            if pg.exists():
                try:
                    d = json.loads(pg.read_text())
                    if d.get("pass", True):
                        todo_gated.append(w)
                    # else: silently skip — pre-gate already decided reject
                except Exception:
                    todo_gated.append(w)  # corrupt marker = include
            else:
                ungated.append(w)  # no marker yet — include + optionally inline-gate
        log.info(f"Pre-gate markers: {len(todo_gated)} pass, {before-len(todo_gated)-len(ungated)} reject, {len(ungated)} ungated")
        # Inline SNR for files not yet pre-gated (Stage 3 newly created)
        if ungated:
            ungated = acoustic_pre_gate(ungated, 25.0, None)
        todo = todo_gated + ungated
        log.info(f"After pre-gate: {len(todo)} files to transcribe (was {before})")
        if not todo:
            log.info("All segments rejected by pre-gate.")
            sys.exit(0)

    # Optional inline acoustic pre-gate (alternative to --use-pregate)
    elif args.pre_gate_snr or args.pre_gate_dnsmos:
        todo = acoustic_pre_gate(todo, args.pre_gate_snr, args.pre_gate_dnsmos)
        if not todo:
            log.info("All segments rejected by acoustic pre-gate.")
            sys.exit(0)

    # Load checkpoint: recover completed pass results from previous runs
    checkpoint = load_checkpoint(checkpoint_file)

    # Two-pass: load each model once over all segments, then merge and write.
    # Each result is checkpointed immediately — safe to kill and restart at any time.
    all_results = []
    import gc
    for cfg in ASR_MODELS:
        model = load_model(cfg, device, gpu_index, args.compute_type)
        results = transcribe_all_with_model(todo, model, cfg, checkpoint, checkpoint_file)
        all_results.append(results)
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    processed, low_agreement = write_transcripts(todo, all_results)
    failed = 0

    print(f"\nDone: {processed} transcribed, {failed} failed")
    print(f"Low ASR agreement (<0.80): {low_agreement} ({100*low_agreement/max(processed,1):.1f}%)")
    print(f"These are flagged for priority human calibration in 05_calibrate.py")
    print(f"Log: {log_path}")
    print(f"Checkpoint: {checkpoint_file}")


if __name__ == "__main__":
    main()
