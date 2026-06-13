#!/usr/bin/env python3
"""
scripts/03b_acoustic_pregate.py
Fast acoustic pre-filter BEFORE transcription to avoid wasting GPU on segments
that Stage 6 would reject anyway.

Computes SNR + DNSMOS on all segments and writes per-file .pregate.json markers.
Stage 4 (04_transcribe.py) can optionally check these before loading into Whisper.

Usage:
  python scripts/03b_acoustic_pregate.py --source podcast [--dry-run]
  python scripts/03b_acoustic_pregate.py --source all --min-snr 25 --min-dnsmos 3.0
  python scripts/03b_acoustic_pregate.py --report    # show stats on existing markers

Output:
  data/segments/{source}/{stem}.pregate.json  →  {"snr": 32.1, "dnsmos": 3.7, "pass": true}
  Rejected files get  {"snr": ..., "dnsmos": ..., "pass": false, "reason": "snr"}

Stats at end: how many pass/fail, savings vs transcribing everything.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "metadata" / "logs" / "03b_pregate.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

SEG_DIR = ROOT / "data" / "segments"
TARGET_SR = 48000
ASR_SR = 16000

DEFAULT_MIN_SNR = 25.0
DEFAULT_MIN_DNSMOS = 3.0


def audio_to_16k(wav48: np.ndarray, sr: int = 48000) -> np.ndarray:
    import torch
    import torchaudio
    t = torch.from_numpy(wav48).float().unsqueeze(0)
    resampler = torchaudio.transforms.Resample(sr, ASR_SR)
    return resampler(t).squeeze(0).numpy()


def compute_snr(wav48: np.ndarray, sr: int = 48000) -> float:
    """Energy-based SNR estimate. Uses bottom 10% as noise floor, top 90% as signal."""
    frame_len = int(sr * 0.025)
    hop = int(sr * 0.010)
    frames = [wav48[i:i+frame_len] for i in range(0, len(wav48) - frame_len, hop)]
    if not frames:
        return 0.0
    energies = np.array([np.mean(f**2) + 1e-10 for f in frames])
    noise_floor = float(np.percentile(energies, 10))
    signal_peak = float(np.percentile(energies, 90))
    if noise_floor <= 0:
        return 0.0
    return round(10 * np.log10(signal_peak / noise_floor), 2)


def find_segments(source: str) -> list[Path]:
    if source == "all":
        return sorted(SEG_DIR.rglob("*.wav"))
    return sorted((SEG_DIR / source).rglob("*.wav"))


def show_report(source: str) -> None:
    wavs = find_segments(source)
    pass_count = fail_count = missing = 0
    snr_values: list[float] = []
    dns_values: list[float] = []
    fail_reasons: dict[str, int] = {}

    for wav in wavs:
        pg = wav.with_suffix(".pregate.json")
        if not pg.exists():
            missing += 1
            continue
        try:
            d = json.loads(pg.read_text())
            if d.get("pass"):
                pass_count += 1
                if d.get("snr") is not None:
                    snr_values.append(d["snr"])
                if d.get("dnsmos") is not None:
                    dns_values.append(d["dnsmos"])
            else:
                fail_count += 1
                r = d.get("reason", "unknown")
                fail_reasons[r] = fail_reasons.get(r, 0) + 1
        except Exception:
            missing += 1

    total = pass_count + fail_count + missing
    print(f"\n=== Acoustic Pre-gate Report ({source}) ===")
    print(f"Total WAVs:    {total}")
    print(f"Gated (pass):  {pass_count}  ({100*pass_count/max(total,1):.1f}%)")
    print(f"Rejected:      {fail_count}  ({100*fail_count/max(total,1):.1f}%)")
    print(f"Not yet gated: {missing}")
    if fail_reasons:
        print(f"Rejection reasons: {dict(sorted(fail_reasons.items(), key=lambda x: -x[1]))}")
    if snr_values:
        print(f"SNR (pass): median={np.median(snr_values):.1f} mean={np.mean(snr_values):.1f} min={min(snr_values):.1f}")
    if dns_values:
        print(f"DNSMOS (pass): median={np.median(dns_values):.2f} mean={np.mean(dns_values):.2f} min={min(dns_values):.2f}")
    if fail_count > 0:
        hrs_saved = fail_count * 7.5 / 3600  # assume 7.5s avg per transcription
        print(f"Estimated GPU transcription time saved: ~{hrs_saved:.1f}h (at 1s/seg)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="podcast",
                        choices=["rthk", "youtube", "podcast", "hktv", "all"])
    parser.add_argument("--min-snr", type=float, default=DEFAULT_MIN_SNR,
                        help=f"SNR threshold in dB (default {DEFAULT_MIN_SNR})")
    parser.add_argument("--min-dnsmos", type=float, default=DEFAULT_MIN_DNSMOS,
                        help=f"DNSMOS OVRL threshold (default {DEFAULT_MIN_DNSMOS}). Set 0 to skip.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without writing .pregate.json files")
    parser.add_argument("--report", action="store_true",
                        help="Show stats on existing .pregate.json markers and exit")
    parser.add_argument("--no-dnsmos", action="store_true",
                        help="Skip DNSMOS (faster; SNR only)")
    parser.add_argument("--workers", type=int, default=4,
                        help="CPU worker threads for SNR computation (default 4)")
    args = parser.parse_args()

    if args.report:
        show_report(args.source)
        return

    if args.no_dnsmos:
        args.min_dnsmos = 0.0

    use_dnsmos = args.min_dnsmos > 0
    if use_dnsmos:
        try:
            from speechmos import dnsmos as _dnsmos_mod
            log.info("DNSMOS enabled via speechmos")
        except ImportError:
            log.warning("speechmos not installed — falling back to SNR-only pre-gate")
            use_dnsmos = False

    wavs = find_segments(args.source)
    todo = [w for w in wavs if not w.with_suffix(".pregate.json").exists()]

    log.info(f"Source: {args.source}, total WAVs: {len(wavs)}, need gating: {len(todo)}")
    log.info(f"Thresholds: SNR>={args.min_snr} dB, DNSMOS>={args.min_dnsmos if use_dnsmos else 'skip'}")

    if not todo:
        log.info("All segments already pre-gated.")
        show_report(args.source)
        return

    if args.dry_run:
        log.info(f"[DRY-RUN] Would process {len(todo)} segments")
        return

    passed = failed_snr = failed_dnsmos = errors = 0

    for i, wav_path in enumerate(todo):
        if i % 2000 == 0:
            log.info(f"Progress: {i}/{len(todo)} | passed={passed} rejected_snr={failed_snr} rejected_dnsmos={failed_dnsmos}")

        pregate_path = wav_path.with_suffix(".pregate.json")

        try:
            wav48, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if wav48.ndim > 1:
                wav48 = wav48.mean(axis=1)

            snr = compute_snr(wav48, sr)
            dns_score = None
            reason = None

            if snr < args.min_snr:
                reason = "snr"
                failed_snr += 1
            elif use_dnsmos:
                wav16k = audio_to_16k(wav48, sr)
                result = _dnsmos_mod.run(wav16k, sr=16000)
                dns_score = round(float(result["ovrl_mos"]), 3)
                if dns_score < args.min_dnsmos:
                    reason = "dnsmos"
                    failed_dnsmos += 1

            record = {
                "snr": snr,
                "dnsmos": dns_score,
                "pass": reason is None,
            }
            if reason:
                record["reason"] = reason
            else:
                passed += 1

            with open(pregate_path, "w") as f:
                json.dump(record, f)

        except Exception as exc:
            log.warning(f"Pre-gate error on {wav_path.name}: {exc}")
            errors += 1
            with open(pregate_path, "w") as f:
                json.dump({"snr": None, "dnsmos": None, "pass": True, "error": str(exc)}, f)
            passed += 1

    total = len(todo)
    log.info(
        f"Pre-gate complete: {passed}/{total} passed "
        f"(rejected snr={failed_snr}, dnsmos={failed_dnsmos}, errors={errors})"
    )
    show_report(args.source)


if __name__ == "__main__":
    main()
