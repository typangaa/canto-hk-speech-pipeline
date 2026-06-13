#!/usr/bin/env python3
"""
scripts/05_calibrate.py
Human-in-loop calibration tool. Plays each segment and shows ASR candidates.
You type the correct Cantonese transcript; it writes text_verified=true.

Usage: python scripts/05_calibrate.py --source [rthk|youtube|podcast|all]
       python scripts/05_calibrate.py --low-agreement-first  (review disagreements first)
       python scripts/05_calibrate.py --resume               (skip already verified)

Controls:
  Type transcript + Enter  → accept your text as canonical
  Enter (blank)            → accept the best-confidence ASR candidate
  s                        → skip this segment (leave text_verified=false)
  d                        → mark as reject/delete (do not include in corpus)
  q                        → quit and save progress

Output: updates *.transcript.json files in place, sets text_verified=true
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "metadata" / "logs" / "05_calibrate.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

SEG_DIR = ROOT / "data" / "segments"


def play_audio(wav_path: Path) -> None:
    """Play a WAV file. Tries aplay (Linux), then sox play, then ffplay."""
    for cmd in [
        ["aplay", "-q", str(wav_path)],
        ["play", "-q", str(wav_path)],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(wav_path)],
    ]:
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode == 0:
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    print("  [Audio playback not available — install aplay or ffplay]")


def load_transcript(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_transcript(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_transcripts(source: str, low_agreement_first: bool) -> list[Path]:
    if source == "all":
        paths = sorted(SEG_DIR.rglob("*.transcript.json"))
    else:
        paths = sorted((SEG_DIR / source).rglob("*.transcript.json"))

    if low_agreement_first:
        def sort_key(p):
            try:
                d = load_transcript(p)
                return d.get("asr_agreement", 1.0)
            except Exception:
                return 1.0
        paths = sorted(paths, key=sort_key)

    return paths


def display_candidates(data: dict) -> None:
    print()
    print("─" * 70)
    candidates = data.get("asr_candidates", [])
    for i, c in enumerate(candidates):
        model_short = c["model"].split("/")[-1][:30]
        conf = c.get("confidence", 0)
        text = c.get("text", "")
        print(f"  ASR {i+1} [{model_short}] conf={conf:.2f}: {text}")
    agreement = data.get("asr_agreement", 0)
    print(f"  Agreement: {agreement:.2f} {'✓' if agreement >= 0.80 else '⚠ low'}")


def calibrate_session(transcripts: list[Path], args: argparse.Namespace) -> None:
    todo = []
    for t in transcripts:
        data = load_transcript(t)
        if args.resume and data.get("text_verified"):
            continue
        if data.get("rejected"):
            continue
        todo.append(t)

    total = len(todo)
    if not total:
        print("Nothing to calibrate. All segments already verified.")
        return

    print(f"\n{'='*70}")
    print(f"Calibration session: {total} segments to review")
    print("Controls: [Enter]=accept ASR | type text=override | s=skip | d=reject | q=quit | r=replay")
    print(f"{'='*70}")

    verified = skipped = rejected = 0

    for i, t_path in enumerate(todo, 1):
        data = load_transcript(t_path)
        wav_path = Path(data.get("seg_path", str(t_path).replace(".transcript.json", ".wav")))

        print(f"\n[{i}/{total}] {wav_path.name}")
        display_candidates(data)

        # Play audio
        if wav_path.exists():
            play_audio(wav_path)
        else:
            print(f"  [WAV not found: {wav_path}]")

        best_text = data.get("text", "")
        if not best_text and data.get("asr_candidates"):
            best = max(data["asr_candidates"], key=lambda c: c.get("confidence", 0))
            best_text = best.get("text", "")

        while True:
            try:
                inp = input(f"  Transcript [Enter='{best_text[:50]}']: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nInterrupted. Saving progress.")
                log.info(f"Session ended: {verified} verified, {skipped} skipped, {rejected} rejected")
                return

            if inp.lower() == "q":
                print("Quitting.")
                log.info(f"Session ended: {verified} verified, {skipped} skipped, {rejected} rejected")
                return
            elif inp.lower() == "r":
                if wav_path.exists():
                    play_audio(wav_path)
                continue
            elif inp.lower() == "s":
                skipped += 1
                break
            elif inp.lower() == "d":
                data["rejected"] = True
                save_transcript(t_path, data)
                rejected += 1
                log.info(f"Rejected: {wav_path.name}")
                break
            else:
                canonical = inp if inp else best_text
                if not canonical:
                    print("  (No text — type a transcript or 's' to skip)")
                    continue
                data["text"] = canonical
                data["text_verified"] = True
                save_transcript(t_path, data)
                verified += 1
                log.info(f"Verified: {wav_path.name} → '{canonical[:60]}'")
                break

    print(f"\n{'='*70}")
    print(f"Session done: {verified} verified, {skipped} skipped, {rejected} rejected")
    remaining = total - verified - skipped - rejected
    if remaining > 0:
        print(f"  {remaining} remain for next session")
    print(f"Log: {LOG_PATH}")


def stats(transcripts: list[Path]) -> None:
    total = verified = rejected = low_agr = 0
    for t in transcripts:
        try:
            data = load_transcript(t)
            total += 1
            if data.get("text_verified"):
                verified += 1
            if data.get("rejected"):
                rejected += 1
            if data.get("asr_agreement", 1.0) < 0.80:
                low_agr += 1
        except Exception:
            pass
    print(f"\nCalibration status:")
    print(f"  Total transcripts:   {total}")
    print(f"  Verified:            {verified} ({100*verified/max(total,1):.1f}%)")
    print(f"  Rejected:            {rejected}")
    print(f"  Low agreement (<0.8):{low_agr}")
    print(f"  Remaining:           {total - verified - rejected}")


def batch_accept(transcripts: list[Path], threshold: float) -> None:
    """Auto-accept segments where asr_agreement >= threshold. No human needed."""
    accepted = skipped_already = 0
    for t_path in transcripts:
        try:
            data = load_transcript(t_path)
            if data.get("text_verified") or data.get("rejected"):
                skipped_already += 1
                continue
            agreement = data.get("asr_agreement", 0.0)
            if agreement >= threshold:
                best_text = data.get("text", "")
                if not best_text:
                    candidates = data.get("asr_candidates", [])
                    if candidates:
                        best = max(candidates, key=lambda c: c.get("confidence", 0))
                        best_text = best.get("text", "")
                if best_text:
                    data["text"] = best_text
                    data["text_verified"] = True
                    save_transcript(t_path, data)
                    accepted += 1
                    log.info(f"Auto-accepted (agr={agreement:.2f}): {t_path.stem[:50]}")
        except Exception as exc:
            log.warning(f"Batch accept error {t_path.name}: {exc}")

    print(f"\nBatch auto-accept (threshold={threshold}): {accepted} accepted, {skipped_already} already done")
    log.info(f"Batch accept done: {accepted} new verifications at threshold {threshold}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all",
                        choices=["rthk", "youtube", "podcast", "hktv", "all"])
    parser.add_argument("--low-agreement-first", action="store_true",
                        help="Review segments with low ASR agreement first")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already verified segments")
    parser.add_argument("--stats", action="store_true",
                        help="Just show calibration statistics, no interactive session")
    parser.add_argument("--batch-accept", type=float, default=None, metavar="THRESHOLD",
                        help="Non-interactively accept all segments with asr_agreement >= THRESHOLD "
                             "(e.g. --batch-accept 0.90). Safe to run before manual calibration.")
    args = parser.parse_args()

    transcripts = find_transcripts(args.source, args.low_agreement_first)
    log.info(f"Found {len(transcripts)} transcript files")

    if args.stats:
        stats(transcripts)
        return

    if args.batch_accept is not None:
        batch_accept(transcripts, args.batch_accept)
        stats(transcripts)
        return

    calibrate_session(transcripts, args)


if __name__ == "__main__":
    main()
