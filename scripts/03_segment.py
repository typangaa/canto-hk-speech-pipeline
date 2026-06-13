#!/usr/bin/env python3
"""
scripts/03_segment.py
Diarize raw audio and cut single-speaker VAD segments (3–20s, 48 kHz mono WAV).
Usage: python scripts/03_segment.py --source [rthk|youtube|podcast|all] [--dry-run] [--workers N]

Pipeline per file:
  1. pyannote diarization → speaker turns
  2. Silero VAD within each single-speaker turn → silence boundaries
  3. Cut clips from 48 kHz master → data/segments/
  4. Write segment metadata alongside each clip

Requires: export HUGGING_FACE_HUB_TOKEN=hf_...
          Accept pyannote/speaker-diarization-3.1 at huggingface.co before first run.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "metadata" / "logs" / "03_segment.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw"
SEG_DIR = ROOT / "data" / "segments"
TARGET_SR = 48000
VAD_SR = 16000
MIN_DUR = 3.0
MAX_DUR = 20.0

# Lazy-loaded globals
_diarize_pipeline = None
_vad_model = None
_vad_utils = None


def get_hf_token() -> Optional[str]:
    return os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def get_diarize_pipeline():
    global _diarize_pipeline
    if _diarize_pipeline is None:
        token = get_hf_token()
        if not token:
            log.warning(
                "HUGGING_FACE_HUB_TOKEN not set — using VAD-only segmentation "
                "(no speaker diarization; whole file treated as single speaker). "
                "Set the token and accept pyannote terms for multi-speaker diarization."
            )
            return None  # triggers VAD-only path
        from pyannote.audio import Pipeline
        log.info("Loading pyannote speaker-diarization-3.1 ...")
        try:
            _diarize_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=token,
            )
        except Exception as exc:
            if "403" in str(exc) or "gated" in str(exc).lower():
                log.error(
                    "pyannote/speaker-diarization-3.1 requires accepting model terms.\n"
                    "  1. Visit: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                    "  2. Accept user conditions with your HuggingFace account.\n"
                    "  3. Re-run this script.\n"
                    "Falling back to VAD-only segmentation (no diarization — multi-speaker risk)."
                )
                return None  # triggers VAD-only path below
            raise
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _diarize_pipeline.to(device)
        log.info(f"Diarization pipeline on {device}")
    return _diarize_pipeline


def get_vad_model():
    global _vad_model, _vad_utils
    if _vad_model is None:
        log.info("Loading Silero VAD ...")
        _vad_model, _vad_utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        # Keep VAD on CPU — Silero VAD is fast enough on CPU and avoids
        # device mismatch issues when passing numpy-derived tensors
    return _vad_model, _vad_utils


def audio_to_16k(wav48: np.ndarray) -> np.ndarray:
    """Downsample from 48 kHz to 16 kHz (in-memory, transient)."""
    import torchaudio
    t = torch.from_numpy(wav48).float().unsqueeze(0)
    resampler = torchaudio.transforms.Resample(TARGET_SR, VAD_SR)
    return resampler(t).squeeze(0).numpy()


def get_vad_segments_in_window(
    wav16: np.ndarray,
    window_start: float,
    window_end: float,
    chunk_sec: float = 60.0,
) -> list[tuple[float, float]]:
    """Run Silero VAD on a mono 16 kHz array; return (start, end) in seconds.

    Processes in chunks of `chunk_sec` to avoid TorchScript memory issues with
    very long audio (Silero VAD can fail on tensors > a few minutes).
    """
    model, utils = get_vad_model()
    get_speech_timestamps = utils[0]

    start_sample = int(window_start * VAD_SR)
    end_sample = int(window_end * VAD_SR)
    chunk_samples = int(chunk_sec * VAD_SR)

    all_timestamps = []
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
            log.warning(f"VAD chunk failed (offset {cursor/VAD_SR:.1f}s): {exc}")
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


def cut_segment(wav48: np.ndarray, start: float, end: float, out_path: Path) -> bool:
    s = int(start * TARGET_SR)
    e = int(end * TARGET_SR)
    clip = wav48[s:e]
    if len(clip) < MIN_DUR * TARGET_SR:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), clip, TARGET_SR, subtype="PCM_16")
    return True


def segment_file(wav_path: Path, out_dir: Path, source_meta: dict, dry_run: bool) -> int:
    stem = wav_path.stem
    seg_meta_path = out_dir / f"{stem}_segments.jsonl"

    if seg_meta_path.exists():
        existing = sum(1 for _ in open(seg_meta_path))
        if existing > 0:
            log.info(f"Skip (already segmented, {existing} segs): {wav_path.name}")
            return existing

    log.info(f"Segmenting: {wav_path.name}")

    # Load 48 kHz master
    wav48, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if sr != TARGET_SR:
        log.warning(f"Expected 48 kHz, got {sr} Hz: {wav_path.name}. Re-reading with torchaudio.")
        import torchaudio
        wav48_t, sr = torchaudio.load(str(wav_path))
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            wav48_t = resampler(wav48_t)
        wav48 = wav48_t.mean(0).numpy()

    if wav48.ndim > 1:
        wav48 = wav48.mean(axis=1)

    duration = len(wav48) / TARGET_SR
    log.info(f"  Duration: {duration:.1f}s")

    if dry_run:
        log.info(f"  [DRY-RUN] Would diarize + segment {wav_path.name}")
        return 0

    # Transient 16 kHz copy for diarization / VAD
    wav16 = audio_to_16k(wav48)

    # --- Diarization ---
    pipeline = get_diarize_pipeline()

    if pipeline is None:
        # Fallback: treat whole file as a single speaker (VAD-only mode)
        log.warning(f"  VAD-only mode (no diarization): {stem}")
        full_duration = len(wav48) / TARGET_SR
        turns = [(0.0, full_duration, "SPEAKER_UNKNOWN")]
    else:
        # pyannote needs a file or tensor; pass as tensor
        wav16_tensor = torch.from_numpy(wav16).float().unsqueeze(0)
        output = pipeline({"waveform": wav16_tensor, "sample_rate": VAD_SR})

        # pyannote 4.x returns DiarizeOutput; older versions return Annotation directly
        # Use exclusive_speaker_diarization (no overlapping turns) — ideal for TTS
        if hasattr(output, 'exclusive_speaker_diarization'):
            annotation = output.exclusive_speaker_diarization
        else:
            annotation = output

        # Collect single-speaker turns
        turns = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append((turn.start, turn.end, speaker))

    log.info(f"  Diarization: {len(turns)} turns, "
             f"{len(set(s for _, _, s in turns))} speakers")

    # --- VAD within single-speaker turns ---
    n_seg = 0
    written_segs = []

    for t_start, t_end, speaker in turns:
        if t_end - t_start < MIN_DUR:
            continue
        vad_windows = get_vad_segments_in_window(wav16, t_start, t_end)
        for v_start, v_end in vad_windows:
            dur = v_end - v_start
            if dur < MIN_DUR or dur > MAX_DUR:
                continue
            seg_name = f"{stem}_seg{n_seg:05d}.wav"
            seg_path = out_dir / seg_name
            if cut_segment(wav48, v_start, v_end, seg_path):
                seg_record = {
                    "seg_path": str(seg_path),
                    "source_wav": str(wav_path),
                    "source_url": source_meta.get("source_url", ""),
                    "program": source_meta.get("program", ""),
                    "source": source_meta.get("source", ""),
                    "domain": source_meta.get("domain", ""),
                    "style": source_meta.get("style", ""),
                    "speaker_tag": speaker,
                    "start_sec": round(v_start, 3),
                    "end_sec": round(v_end, 3),
                    "duration_sec": round(dur, 3),
                    "sample_rate": TARGET_SR,
                }
                written_segs.append(seg_record)
                n_seg += 1

    del wav16  # free transient copy

    # Write segment metadata
    with open(seg_meta_path, "w") as f:
        for rec in written_segs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(f"  Wrote {n_seg} segments → {out_dir}")
    return n_seg


def load_downloaded_meta(source: str) -> dict[str, dict]:
    """Return wav_path → metadata from downloaded.jsonl for given source."""
    log_path = ROOT / "metadata" / "downloaded.jsonl"
    meta = {}
    if not log_path.exists():
        return meta
    with open(log_path) as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                if rec.get("source") == source or source == "all":
                    meta[rec["wav_path"]] = rec
            except Exception:
                pass
    return meta


def find_raw_wavs(source: str) -> list[Path]:
    if source == "all":
        return sorted(RAW_DIR.rglob("*.wav"))
    return sorted((RAW_DIR / source).rglob("*.wav"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all",
                        choices=["rthk", "youtube", "podcast", "hktv", "all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default 1 — GPU memory limited)")
    args = parser.parse_args()

    wavs = find_raw_wavs(args.source)
    log.info(f"Found {len(wavs)} WAV files in data/raw/{args.source}")

    if not wavs:
        log.warning("No WAV files found. Run 02_download.py first.")
        sys.exit(0)

    meta = load_downloaded_meta(args.source)

    processed = skipped = failed = 0
    for wav_path in wavs:
        source_meta = meta.get(str(wav_path), {})
        # Infer source from path if not in downloaded log
        if not source_meta.get("source"):
            for src in ("rthk", "youtube", "podcast", "hktv"):
                if src in str(wav_path):
                    source_meta["source"] = src
                    break

        out_subdir = wav_path.relative_to(RAW_DIR).parts[0]  # e.g. "rthk"
        out_dir = SEG_DIR / out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            n = segment_file(wav_path, out_dir, source_meta, args.dry_run)
            if n > 0:
                processed += 1
            else:
                skipped += 1
        except Exception as exc:
            log.error(f"Failed on {wav_path.name}: {exc}", exc_info=True)
            failed += 1

    total_segs = sum(1 for _ in SEG_DIR.rglob("*.wav"))
    print(f"\nDone: {processed} files segmented, {skipped} skipped, {failed} failed")
    print(f"Total segments in data/segments/: {total_segs}")
    print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
