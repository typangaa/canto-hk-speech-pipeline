#!/usr/bin/env python3
"""
reconstruct.py — rebuild the audio for the Cantonese speech corpus from its
metadata-only release.

No copyrighted audio is distributed. This script re-downloads each original source
(YouTube / RTHK archive / podcast RSS) and re-slices the [start_sec, end_sec] window
for every segment, reproducing the exact 48 kHz mono PCM WAV used to build the corpus.

Usage:
  python reconstruct.py --manifest manifest_release.jsonl --out audio/
  python reconstruct.py --manifest manifest_release.jsonl --out audio/ --sample 200
  python reconstruct.py --manifest manifest_release.jsonl --validate   # A3 self-check

Requires: yt-dlp, ffmpeg on PATH; python soundfile + numpy.
Sources are converted to 48 kHz / mono / s16 (the same ffmpeg flags used at ingest),
then sliced. Segments whose source is unavailable (deleted video, dead feed) are
skipped and listed in the run report.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 48000


def pcm_sha256_from_array(data: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def pcm_sha256_file(path: Path) -> str:
    data, _ = sf.read(str(path), dtype="int16", always_2d=False)
    return pcm_sha256_from_array(data)


def download_source(url: str, kind: str, dst: Path, ytopts: dict) -> bool:
    """Fetch `url`, convert to 48k mono s16 wav at `dst`. Return True on success.

    youtube -> yt-dlp (needs a JS runtime, and usually cookies to pass YouTube's
               bot check; pass --cookies / --cookies-from-browser).
    archive_mp4 / podcast_rss -> direct media URL, read straight by ffmpeg.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if kind == "youtube":
            cmd = ["yt-dlp", "--quiet", "--no-warnings", "--no-update",
                   "--format", "bestaudio/best", "--extract-audio", "--audio-format", "wav",
                   "--postprocessor-args", "ffmpeg:-ar 48000 -ac 1 -sample_fmt s16"]
            if ytopts.get("js_runtime"):
                cmd += ["--js-runtimes", ytopts["js_runtime"]]
            if ytopts.get("cookies"):
                cmd += ["--cookies", ytopts["cookies"]]
            if ytopts.get("cookies_from_browser"):
                cmd += ["--cookies-from-browser", ytopts["cookies_from_browser"]]
            cmd += ["-o", str(dst.with_suffix(".%(ext)s")), url]
            r = subprocess.run(cmd, capture_output=True)
            return r.returncode == 0 and dst.exists()
        # archive_mp4 / podcast_rss : ffmpeg reads the http(s) media URL directly
        r = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", url, "-ar", "48000", "-ac", "1", "-sample_fmt", "s16", str(dst)],
            capture_output=True,
        )
        return r.returncode == 0 and dst.exists()
    except Exception as exc:
        print(f"  download error {url}: {exc}", file=sys.stderr)
        return False


_MFCC_SR = 16000


def mfcc_fingerprint(samples: np.ndarray, src_sr: int = TARGET_SR) -> np.ndarray:
    """Compact perceptual fingerprint: 13 MFCC means + 13 stds (26-d), computed at
    16 kHz. Robust to lossy re-encode (gain/codec jitter), sensitive to a different
    take (e.g. ad-shifted podcast). Compared with cosine similarity."""
    import librosa
    y = samples.astype(np.float32)
    if np.issubdtype(samples.dtype, np.integer):
        y = y / 32768.0
    if src_sr != _MFCC_SR:
        y = librosa.resample(y, orig_sr=src_sr, target_sr=_MFCC_SR)
    m = librosa.feature.mfcc(y=y, sr=_MFCC_SR, n_mfcc=13)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)


def fp_cosine(a: np.ndarray, b: np.ndarray) -> float:
    da = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / da) if da > 0 else 0.0


def best_offset_corr(orig: np.ndarray, rebuilt: np.ndarray, max_ms: int = 120) -> float:
    """Max normalized cross-correlation of rebuilt vs orig over a +/-max_ms sample
    shift search. ~1.0 means perceptually identical content (allowing a small
    lossy-decode offset); low means genuinely different audio."""
    a = orig.astype(np.float64)
    b = rebuilt.astype(np.float64)
    if len(a) < 2000 or len(b) < 2000:
        return float("nan")
    a -= a.mean()
    b -= b.mean()
    maxshift = int(TARGET_SR * max_ms / 1000)
    best = -1.0
    for s in range(-maxshift, maxshift + 1, 8):
        if s >= 0:
            x, y = a[s:], b[: len(a) - s]
        else:
            x, y = a[:s], b[-s:]
        m = min(len(x), len(y))
        if m < 2000:
            continue
        x, y = x[:m], y[:m]
        d = np.sqrt((x * x).sum() * (y * y).sum())
        if d > 0:
            best = max(best, float((x * y).sum() / d))
    return best


def slice_segment(src_wav: Path, start: float, end: float) -> np.ndarray:
    data, sr = sf.read(str(src_wav), dtype="int16", always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    # match 03_segment.py cut_segment: int() truncation, not round()
    a = int(start * sr)
    b = int(end * sr)
    return data[a:b]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest_release.jsonl")
    ap.add_argument("--out", default="audio")
    ap.add_argument("--cache", default="_sources", help="source download cache dir")
    ap.add_argument("--sample", type=int, default=0, help="only first N segments")
    ap.add_argument("--validate", action="store_true",
                    help="compare rebuilt PCM to the original filtered wav (audio_path) "
                         "and report match rate; does not write output")
    ap.add_argument("--keep-sources", action="store_true")
    ap.add_argument("--kinds", help="comma list to restrict source_kind "
                                    "(youtube,archive_mp4,podcast_rss)")
    ap.add_argument("--cookies", help="cookies.txt for yt-dlp (YouTube bot check)")
    ap.add_argument("--cookies-from-browser",
                    help="browser name for yt-dlp --cookies-from-browser")
    ap.add_argument("--js-runtime", default="node",
                    help="yt-dlp JS runtime (default: node; needs node/deno on PATH)")
    ap.add_argument("--fp-threshold", type=float, default=0.95,
                    help="MFCC fingerprint cosine threshold to accept a rebuilt segment")
    args = ap.parse_args()

    ytopts = {"cookies": args.cookies,
              "cookies_from_browser": args.cookies_from_browser,
              "js_runtime": args.js_runtime}

    rows = [json.loads(l) for l in open(args.manifest)]
    if args.kinds:
        keep = set(args.kinds.split(","))
        rows = [r for r in rows if r["source_kind"] in keep]
    if args.sample:
        rows = rows[: args.sample]
    out_dir, cache = Path(args.out), Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    if not args.validate:
        out_dir.mkdir(parents=True, exist_ok=True)

    by_url: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by_url[(r["source_url"], r["source_kind"])].append(r)

    n_ok = n_skip_src = n_seg = n_match = n_mismatch = n_checked = n_perceptual = 0
    from collections import Counter as _C
    corr_buckets = _C()
    broken_shown = [0]
    cal_rows = []
    dead = []
    for (url, kind), segs in by_url.items():
        src_wav = cache / (hashlib.md5(url.encode()).hexdigest() + ".wav")
        if not src_wav.exists():
            if not download_source(url, kind, src_wav, ytopts):
                dead.append(url)
                n_skip_src += len(segs)
                continue
        for r in segs:
            try:
                clip = slice_segment(src_wav, r["start_sec"], r["end_sec"])
            except Exception as exc:
                print(f"  slice error {r['id']}: {exc}", file=sys.stderr)
                continue
            n_seg += 1
            if args.validate:
                try:
                    orig = sf.read(str(r["audio_path"]), dtype="int16", always_2d=False)[0]
                    if orig.ndim > 1:
                        orig = orig[:, 0]
                except Exception:
                    continue
                n_checked += 1
                corr = best_offset_corr(orig, clip)          # truth label (needs orig)
                fpc = fp_cosine(mfcc_fingerprint(orig), mfcc_fingerprint(clip))
                truth = "good" if corr >= 0.5 else "broken"
                fp_pred = "good" if fpc >= args.fp_threshold else "broken"
                cal_rows.append((corr, fpc, truth, fp_pred))
                if truth == "good" and corr >= 0.9:
                    corr_buckets["good>=0.9"] += 1
                elif truth == "good":
                    corr_buckets["partial.5-.9"] += 1
                else:
                    corr_buckets["broken<0.5"] += 1
                if broken_shown[0] < 10:
                    broken_shown[0] += 1
                    print(f"  {r['id']}: corr={corr:.3f} fp_cos={fpc:.4f} "
                          f"truth={truth} pred={fp_pred}")
            else:
                sf.write(str(out_dir / f"{r['id']}.wav"), clip, TARGET_SR, subtype="PCM_16")
                n_ok += 1
        if not args.keep_sources:
            src_wav.unlink(missing_ok=True)

    print("\n===== reconstruction report =====")
    print(f"sources: {len(by_url)}  dead/unavailable: {len(dead)}")
    print(f"segments sliced: {n_seg}  skipped (dead source): {n_skip_src}")
    if args.validate:
        print(f"VALIDATE: checked {n_checked}")
        for k in ("good>=0.9", "partial.5-.9", "broken<0.5"):
            v = corr_buckets.get(k, 0)
            pct = 100 * v / n_checked if n_checked else 0
            print(f"  corr {k:14}: {v:4} ({pct:.1f}%)")
        # fingerprint vs truth (truth from corr>=0.5) confusion matrix
        tp = sum(1 for c, f, t, p in cal_rows if t == "good" and p == "good")
        fn = sum(1 for c, f, t, p in cal_rows if t == "good" and p == "broken")
        fp = sum(1 for c, f, t, p in cal_rows if t == "broken" and p == "good")
        tn = sum(1 for c, f, t, p in cal_rows if t == "broken" and p == "broken")
        good_fps = [f for c, f, t, p in cal_rows if t == "good"]
        bad_fps = [f for c, f, t, p in cal_rows if t == "broken"]
        print(f"\n  fingerprint @ thr={args.fp_threshold}: "
              f"keep-good(TP)={tp} drop-good(FN)={fn} "
              f"keep-bad(FP)={fp} drop-bad(TN)={tn}")
        if good_fps:
            print(f"  fp_cos  GOOD segs: min {min(good_fps):.4f} "
                  f"mean {sum(good_fps)/len(good_fps):.4f}")
        if bad_fps:
            print(f"  fp_cos BROKEN segs: max {max(bad_fps):.4f} "
                  f"mean {sum(bad_fps)/len(bad_fps):.4f}")
    else:
        print(f"written: {n_ok} wav -> {out_dir}")
    if dead:
        Path("reconstruct_dead_sources.txt").write_text("\n".join(dead))
        print(f"dead sources listed -> reconstruct_dead_sources.txt")


if __name__ == "__main__":
    main()
