"""
scripts/test_sensevoice.py
Quick SenseVoice-Small integration test for the Cantonese pipeline.

Usage:
    /tmp/sv_venv/bin/python scripts/test_sensevoice.py [wav_file ...]

If no wav files given, uses TEST_WAVS list below.
Prints raw output, cleaned text, emotion tag, and compares with existing
canto_ft / whisper_v3 / qwen3_asr transcripts from corpus.duckdb (read-only,
only if DB is not locked by asr.agreement).
"""
import sys
import time
import os
import re
import argparse
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# ── Default test files (updated by caller or auto-discovered) ───────────────
TEST_WAVS = []

ASR_SR = 16000  # SenseVoice expects 16 kHz

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_wav_16k(path: str) -> np.ndarray | None:
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != ASR_SR:
            # Resample to 16 kHz
            from math import gcd
            g = gcd(sr, ASR_SR)
            wav = resample_poly(wav, ASR_SR // g, sr // g).astype(np.float32)
        return wav
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


def strip_tags(text: str) -> str:
    """Remove SenseVoice inline tags like <|zh|><|NEUTRAL|><|Speech|><|woitn|>"""
    return re.sub(r"<\|[^|]+\|>", "", text).strip()


def extract_emotion(raw_text: str) -> str:
    """Pull out the emotion token from raw SenseVoice output."""
    m = re.search(r"<\|(HAPPY|SAD|ANGRY|NEUTRAL|DISGUSTED|FEARFUL|SURPRISED)\|>",
                  raw_text, re.IGNORECASE)
    return m.group(1) if m else "UNKNOWN"


def extract_lang(raw_text: str) -> str:
    m = re.search(r"<\|(zh|en|yue|ja|ko|nospeech)\|>", raw_text, re.IGNORECASE)
    return m.group(1) if m else "?"


def extract_event(raw_text: str) -> str:
    m = re.search(r"<\|(Speech|BGM|Applause|Laughter|Cry|Cough|Sneeze|Breath|Music)\|>",
                  raw_text, re.IGNORECASE)
    return m.group(1) if m else "?"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="*", help="WAV files to transcribe")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lang", default="yue", help="Language code (yue=Cantonese)")
    args = ap.parse_args()

    wav_paths = args.wavs if args.wavs else TEST_WAVS
    if not wav_paths:
        print("No WAV files specified. Pass paths as arguments or set TEST_WAVS.")
        sys.exit(1)

    print("=" * 70)
    print("SenseVoice-Small — Cantonese pipeline integration test")
    print(f"  device={args.device}  lang={args.lang}  files={len(wav_paths)}")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n[1/3] Loading SenseVoiceSmall model...")
    t0 = time.time()
    try:
        from funasr import AutoModel
    except ImportError:
        print("ERROR: funasr not installed. Run: pip install funasr modelscope")
        sys.exit(1)

    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        device=args.device,
        disable_update=True,        # don't phone home on every run
    )
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # ── Try to get reference transcripts from DB (read-only, best-effort) ─────
    refs = {}
    db_path = os.path.join(os.path.dirname(__file__), "..", "metadata", "corpus.duckdb")
    db_path = os.path.normpath(db_path)
    try:
        import duckdb
        con = duckdb.connect(db_path, read_only=True)
        # Map audio_path → {canto_ft, whisper_v3, qwen3_asr}
        placeholders = ",".join(["?" for _ in wav_paths])
        rows = con.execute(f"""
            SELECT s.audio_path, r.model, r.text
            FROM segments s
            JOIN asr_results r ON r.id = s.id
            WHERE s.audio_path IN ({placeholders})
              AND r.model IN (
                'Qwen/Qwen3-ASR-1.7B+Cantonese',
                'Systran/faster-whisper-large-v3+zh',
                '/home/typangaa/Documents/canto-hk-speech-pipeline/data/ct2_models/whisper-large-v2-cantonese+zh'
              )
        """, wav_paths).fetchall()
        for path, model_id, text in rows:
            refs.setdefault(path, {})
            if "Qwen" in model_id:
                refs[path]["qwen3"] = text
            elif "large-v3" in model_id:
                refs[path]["whisper_v3"] = text
            else:
                refs[path]["canto_ft"] = text
        con.close()
        print(f"  Loaded reference transcripts for {len(refs)} files from DB")
    except Exception as e:
        print(f"  (DB not available for reference comparison: {e})")

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"\n[2/3] Running SenseVoice inference on {len(wav_paths)} files...\n")

    total_audio_s = 0.0
    t_infer_start = time.time()

    for i, wav_path in enumerate(wav_paths, 1):
        wav = load_wav_16k(wav_path)
        if wav is None:
            continue
        dur = len(wav) / ASR_SR
        total_audio_s += dur
        fname = os.path.basename(wav_path)

        t1 = time.time()
        try:
            res = model.generate(
                input=wav,
                language=args.lang,
                use_itn=True,           # inverse text normalisation (numbers, dates)
                batch_size_s=300,
            )
        except Exception as e:
            print(f"[{i}] ERROR on {fname}: {e}")
            continue
        elapsed = time.time() - t1

        raw_text = res[0]["text"] if res else ""
        clean = strip_tags(raw_text)
        emotion = extract_emotion(raw_text)
        lang_tag = extract_lang(raw_text)
        event = extract_event(raw_text)

        print(f"[{i}] {fname}  ({dur:.1f}s audio, {elapsed*1000:.0f}ms inference)")
        print(f"     RAW      : {raw_text[:120]}")
        print(f"     CLEAN    : {clean}")
        print(f"     lang={lang_tag}  emotion={emotion}  event={event}")

        ref = refs.get(wav_path, {})
        if ref.get("canto_ft"):
            print(f"     canto_ft : {ref['canto_ft']}")
        if ref.get("whisper_v3"):
            print(f"     whisper3 : {ref['whisper_v3']}")
        if ref.get("qwen3"):
            print(f"     qwen3    : {ref['qwen3']}")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_infer_start
    rtf = total_audio_s / total_elapsed if total_elapsed > 0 else 0
    print("=" * 70)
    print(f"[3/3] Done.  {total_audio_s:.1f}s audio in {total_elapsed:.1f}s wall → {rtf:.1f}× real-time")
    print(f"      (Whisper-large RTF for comparison: ~4-6× real-time)")
    print("=" * 70)


if __name__ == "__main__":
    main()
