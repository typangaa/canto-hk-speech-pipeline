# Pipeline Specification — canto-hk-speech-pipeline

> Detailed implementation guide for each pipeline stage.
> Read alongside `CLAUDE.md` (overview + constraints) and `docs/KNOWN_ISSUES.md`.

---

## Core Audio Rate Strategy (read first)

Every stored segment is a **48 kHz mono master**. Downsampling is irreversible, and
every modern TTS codec needs ≥24 kHz (NeuCodec=24k, F5-TTS=24k, MOSS-Nano=48k). See
`KNOWN_ISSUES.md §11`.

VAD, diarization, ASR, and DNSMOS all need 16 kHz internally. Generate a **transient**
16 kHz copy (in memory via `librosa.load(path, sr=16000)`), use it, and discard it.
**Never** overwrite the 48 kHz master with a downsampled version.

```python
import librosa, soundfile as sf

def load_16k(path):                     # transient working copy for tools
    audio, _ = librosa.load(path, sr=16000, mono=True)
    return audio

def save_master(path, audio48k):        # what persists to disk
    sf.write(path, audio48k, 48000, subtype="PCM_16")
```

---

## General Rules for All Scripts

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--source", choices=["rthk","youtube","podcast","hktv","all"], default="all")
parser.add_argument("--dry-run", action="store_true", help="Log actions without writing files")
parser.add_argument("--limit", type=int, default=None, help="Process only N files (for testing)")

# Idempotency — skip files that already have output
if output_path.exists():
    stats["skipped"] += 1
    continue

# Log to metadata/logs/{script_name}.log
# Print summary at end: processed / skipped / failed
# Exit code 1 if >10% of files failed
```

---

## Stage 1 — Source Discovery (`01_discover.py`)

**Purpose**: Survey configured sources, estimate available hours. No download.

**Input**: `sources/*.yaml`  **Output**: `metadata/discovery_report.json`

```python
# For each source entry:
#   - yt-dlp --dump-json --flat-playlist (no download) to enumerate videos
#   - estimate duration from metadata
#   - cross-ref metadata/downloaded.jsonl to skip already-downloaded
# Output: per-source {n_videos, estimated_hours, already_downloaded}
```

```bash
yt-dlp --dump-json --flat-playlist "https://www.youtube.com/playlist?list=..." 2>/dev/null
# duration = info.get('duration', 0)  # seconds
```

---

## Stage 2 — Download (`02_download.py`)

**Purpose**: Download the **highest-quality** audio available.

**Output**: `data/raw/{source}/{channel_slug}/{YYYYMMDD}_{title_slug}_{id}.wav`

**Critical**: keep the best audio. YouTube typically serves 48 kHz AAC — preserve it.
Do not let yt-dlp/ffmpeg downsample below 48 kHz.

```python
YDL_OPTS = [
    "yt-dlp",
    "--format", "bestaudio/best",
    "--extract-audio",
    "--audio-format", "wav",
    "--audio-quality", "0",
    "--postprocessor-args", "ffmpeg:-ar 48000 -ac 1",   # force 48 kHz mono, never lower
    "--output", "%(upload_date)s_%(title)s_%(id)s.%(ext)s",
    "--restrict-filenames",
    "--no-playlist",                 # unless downloading a playlist
    "--retries", "5", "--fragment-retries", "5",
    "--sleep-interval", "3", "--max-sleep-interval", "8",
    "--match-filter", "duration > 120 & duration < 14400",
]
```

**Download log**: append per file to `metadata/downloaded.jsonl`:
```json
{"url":"https://youtube.com/watch?v=...","local_path":"/mnt/Drive3/.../raw/rthk/...",
 "source":"rthk","program":"創科新里程","duration_sec":1842,"sample_rate":48000,
 "downloaded_at":"2026-06-09T14:23:01"}
```

**Idempotency**: skip URLs already in `downloaded.jsonl`.

**Podcasts**: parse RSS with `feedparser`, download MP3/M4A with `requests`, then keep
the native rate (convert to 48 kHz WAV only if the source is already ≥48 kHz; if a
podcast is genuinely 44.1 kHz, store 44.1 kHz — never upsample, never go below source).

---

## Stage 3 — Diarization + VAD Segmentation (`03_segment.py`)

**Purpose**: Split raw audio into **single-speaker** 3–20 s clips at silence boundaries,
saved as 48 kHz masters.

**Output**: `data/segments/{source}/{channel}/{stem}_seg{N:05d}.wav` (48 kHz mono)

**Two requirements that must both hold**:
1. No clip spans a speaker change (`KNOWN_ISSUES.md §10`).
2. No clip is hard-cut at a fixed duration (`KNOWN_ISSUES.md §2`).

**Algorithm**: diarize first → within each single-speaker turn, run VAD to cut clips.

```python
import torch, librosa, soundfile as sf
from pyannote.audio import Pipeline

# 1. Diarization (single-speaker turns). HF token needed for pretrained pipeline.
dia = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                               use_auth_token=HF_TOKEN)

# 2. Silero VAD (runs on 16 kHz)
vad_model, vad_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
get_speech_timestamps, _, read_audio, _, _ = vad_utils

def segment_audio(audio_path, out_dir, min_dur=3.0, max_dur=20.0):
    out = []
    # Diarize on the file (pyannote internally resamples)
    diarization = dia(str(audio_path))

    # Load both: 48 kHz master (to cut + save) and 16 kHz (for VAD)
    master48, _ = librosa.load(str(audio_path), sr=48000, mono=True)
    audio16     = librosa.load(str(audio_path), sr=16000, mono=True)[0]

    seg_i = 0
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Reject overlapping speech regions (diarization marks them); skip very short turns
        t0, t1 = turn.start, turn.end
        if (t1 - t0) < min_dur:
            continue

        # Run VAD WITHIN this single-speaker turn (slice the 16 kHz copy)
        seg16 = audio16[int(t0*16000):int(t1*16000)]
        ts = get_speech_timestamps(
            torch.from_numpy(seg16), vad_model, sampling_rate=16000,
            min_silence_duration_ms=400,
            min_speech_duration_ms=int(min_dur*1000),
            max_speech_duration_s=max_dur,
            speech_pad_ms=100,
        )
        for t in ts:
            # map VAD timestamps (within-turn, 16k) back to absolute seconds
            abs_start = t0 + t['start']/16000
            abs_end   = t0 + t['end']/16000
            dur = abs_end - abs_start
            if not (min_dur <= dur <= max_dur):
                continue
            # cut from the 48 kHz master
            clip48 = master48[int(abs_start*48000):int(abs_end*48000)]
            p = out_dir / f"{audio_path.stem}_seg{seg_i:05d}.wav"
            sf.write(str(p), clip48, 48000, subtype="PCM_16")
            # record local speaker label alongside (e.g. a sidecar .speaker file or dict)
            out.append((p, speaker, dur))
            seg_i += 1
    return out
```

**Post-segmentation assertions**:
```python
for p, spk, dur in out:
    info = sf.info(str(p))
    assert info.samplerate == 48000, f"Not 48 kHz: {p}"
    assert 3.0 <= dur <= 20.0,       f"Out of duration bounds: {p} ({dur:.2f}s)"
```

**Loudness normalisation** (apply to the 48 kHz master, in place is fine):
```python
import pyloudnorm as pyln, numpy as np
def normalise_loudness(wav_path, target_lufs=-23.0):
    data, sr = sf.read(str(wav_path))
    loud = pyln.Meter(sr).integrated_loudness(data)
    if np.isfinite(loud):
        sf.write(str(wav_path), pyln.normalize.loudness(data, loud, target_lufs), sr)
```

**Note on diarization speaker labels**: the label here is **per-file** (`SPEAKER_00`,
…). Stage 8 clusters these across files into global `speaker_id`s.

---

## Stage 4 — Multi-ASR Transcription (`04_transcribe.py`)

**Purpose**: Produce **multiple** candidate transcripts per segment for later human
calibration. Never `language="yue"` (`KNOWN_ISSUES.md §9`).

**Output**: `data/segments/.../{seg}.transcript.json`

```python
# Run N models. Recommended starting set (extend as desired):
#   A) Cantonese fine-tuned Whisper — outputs written Cantonese, avoids collapse
#   B) base whisper-large-v3 with language="zh" + written-Cantonese initial_prompt
# Each runs on a 16 kHz copy of the 48 kHz master.

import librosa
from faster_whisper import WhisperModel   # or transformers pipeline

MODELS = [
    {"id": "simonl0909/whisper-large-v2-cantonese", "lang": None,  "prompt": None},
    {"id": "openai/whisper-large-v3",               "lang": "zh",
     "prompt": "以下係廣東話口語，請用粵語白話文書寫，例如：係、唔係、冇、喺、佢哋、嘅、嗰、嚟。"},
]

def transcribe_all(wav_path):
    audio16 = librosa.load(str(wav_path), sr=16000, mono=True)[0]
    candidates = []
    for m in MODELS:
        text, conf = run_model(m, audio16)     # implement per backend
        candidates.append({"model": m["id"] + (f"+{m['lang']}" if m["lang"] else ""),
                           "text": text.strip(), "confidence": round(conf, 3)})
    agreement = char_agreement([c["text"] for c in candidates])
    return {"candidates": candidates, "asr_agreement": round(agreement, 3)}

def char_agreement(texts):
    """Mean pairwise character-level overlap (e.g. via difflib ratio). 1.0 = identical."""
    import difflib, itertools
    if len(texts) < 2:
        return 1.0
    ratios = [difflib.SequenceMatcher(None, a, b).ratio()
              for a, b in itertools.combinations(texts, 2)]
    return sum(ratios) / len(ratios)
```

**Cantonese marker preference** (`KNOWN_ISSUES.md §9`):
```python
CANTONESE_MARKERS = set("係冇喺佢哋嘅嗰嚟咁啩囉𠮶")
def marker_count(text): return sum(1 for c in text if c in CANTONESE_MARKERS)
# Use to pick a sensible default candidate to pre-fill the calibration UI.
```

**transcript.json format**:
```json
{
  "candidates": [
    {"model": "simonl0909/whisper-large-v2-cantonese", "text": "...", "confidence": 0.88},
    {"model": "openai/whisper-large-v3+zh", "text": "...", "confidence": 0.82}
  ],
  "asr_agreement": 0.91,
  "suggested_text": "...",
  "language": "yue"
}
```

---

## Stage 5 — Manual Calibration (`05_calibrate.py`) — HUMAN-IN-LOOP

**Purpose**: A human produces the **canonical** `text` from the ASR candidates. This is
the project's chosen quality bar — no segment is finalised on a single ASR output.

**Output**: `data/segments/.../{seg}.verified.json`

```json
{
  "text": "心臟病中風這些常見的心腦血管疾病",
  "text_verified": true,
  "verified_by": "human",
  "verified_at": "2026-06-10",
  "source_candidate": "simonl0909/whisper-large-v2-cantonese"
}
```

**Tooling the script must provide** (it is an assistant for the human, not an
auto-labeller):

1. **Prioritise by disagreement.** Sort segments by ascending `asr_agreement` so the
   human reviews the most uncertain first. High-agreement segments can be batch-accepted
   (pre-fill `text` with the agreed candidate) and spot-checked.
2. **Side-by-side candidates + audio playback path** for each segment.
3. **Write `verified.json`** only when the human confirms. Idempotent: skip segments
   already verified.
4. Track progress: `metadata/calibration_progress.json` (verified / pending counts).

```python
# Suggested batching policy (the human sets the canonical text in all cases):
#   agreement >= 0.95  -> pre-fill best candidate, human spot-checks (fast accept)
#   0.80 - 0.95        -> human picks/edits among candidates
#   < 0.80             -> human transcribes from audio (candidates unreliable)
```

**This is a CHECKPOINT.** Stages 6+ consume `verified.json`. Do not run G2P or build the
final manifest from unverified text unless explicitly instructed to produce an interim
ASR-only manifest (mark such entries `text_verified: false`).

---

## Stage 6 — Quality Filtering (`06_filter.py`)

**Purpose**: Apply quality filters. Move passing files to `data/filtered/`.

**Input**: `data/segments/` + transcript + verified JSONs
**Output**: `data/filtered/...` + `metadata/filter_report.json`

```python
import librosa, numpy as np
from speechmos import dnsmos          # pip install speechmos  (KNOWN_ISSUES §4)

def compute_dnsmos(wav_path) -> float:
    audio16, _ = librosa.load(str(wav_path), sr=16000, mono=True)   # DNSMOS needs 16 kHz
    return dnsmos.run(audio16, sr=16000)["ovrl_mos"]                # range [1,5]

def compute_snr(wav_path) -> float:
    data, sr = librosa.load(str(wav_path), sr=None, mono=True)
    fs = sr // 100
    frames = [data[i:i+fs] for i in range(0, len(data)-fs, fs)]
    rms = sorted((np.sqrt(np.mean(f**2)) for f in frames if len(f)==fs), reverse=True)
    if len(rms) < 10: return 0.0
    sig = np.mean(rms[:len(rms)//10]); noi = np.mean(rms[-len(rms)//10:]) + 1e-10
    return 20*np.log10(sig/noi)

def passes_filters(wav_path, verified, transcript, thr):
    dur = librosa.get_duration(path=str(wav_path)); reasons = {}
    text = verified["text"].strip()
    cjk = sum(1 for c in text if '一' <= c <= '鿿')
    eng = sum(1 for c in text if c.isascii() and c.isalpha())
    alpha = sum(1 for c in text if c.isalpha()) or 1

    reasons["duration"]      = thr["min_dur"] <= dur <= thr["max_dur"]
    reasons["min_chars"]     = cjk >= thr["min_chars"]
    reasons["max_chars"]     = len(text) <= thr["max_chars"]
    reasons["english_ratio"] = (eng/alpha) <= thr["max_english_ratio"]
    reasons["agreement"]     = transcript["asr_agreement"] >= thr["min_agreement"]
    reasons["snr"]           = compute_snr(wav_path) >= thr["min_snr"]
    reasons["dnsmos"]        = compute_dnsmos(wav_path) >= thr["min_dnsmos"]
    return all(reasons.values()), reasons
```

**DNSMOS sanity check (REQUIRED before scaling — `KNOWN_ISSUES.md §4`)**:
```python
scores = [compute_dnsmos(f) for f in first_100_files]
assert all(1.0 <= s <= 5.0 for s in scores), "DNSMOS out of range — check setup"
assert 2.0 <= sorted(scores)[50] <= 4.5, "DNSMOS median suspicious"
```

**Thresholds** (see `docs/QUALITY_SPEC.md`):
```python
THRESHOLDS = {"min_dur":3.0,"max_dur":20.0,"min_agreement":0.80,"min_chars":5,
              "max_chars":150,"max_english_ratio":0.30,"min_snr":25.0,"min_dnsmos":3.0}
```

---

## Stage 7 — G2P Processing (`07_g2p.py`)

**Purpose**: Convert the **verified** Cantonese text to Jyutping with tone numbers.

**Input**: `data/filtered/` + `*.verified.json`  **Output**: `*.jyutping.json`

```python
import re, ToJyutping        # verified API; ~99% accuracy

JYUTPING_TOKEN = re.compile(r'^[a-z]+[1-6]$')

def normalise_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # TODO: convert Arabic numerals to Cantonese written form (e.g. 2024 → 二零二四)
    return text

def text_to_jyutping(text):
    norm = normalise_text(text)
    pairs = ToJyutping.get_jyutping_list(norm)     # [('心','sam1'), ...]
    tokens = []
    for char, jp in pairs:
        if jp is None:                              # English / unknown char
            if char.isascii() and char.isalpha():
                tokens.append(f"[{char.upper()}]")
            continue
        tokens.extend(jp.split())                   # a char may yield >1 syllable (e.g. 瓩)
    jp_str = " ".join(tokens)
    real = [t for t in tokens if not t.startswith("[")]
    valid = sum(1 for t in real if JYUTPING_TOKEN.match(t))
    frac = valid/len(real) if real else 0.0
    return jp_str, frac

jp, frac = text_to_jyutping(verified["text"])
if frac < 0.95:
    log.warning(f"Low Jyutping validity ({frac:.1%}): {verified['text']!r}")
# reject if frac < 0.80 (QUALITY_SPEC.md secondary filters)
```

**jyutping.json**:
```json
{"text":"心臟病中風","jyutping":"sam1 zong6 beng6 zung1 fung1","valid_fraction":0.98,
 "g2p_tool":"ToJyutping","g2p_version":"3.x.x"}
```

---

## Stage 8 — Speaker Identification (`08_speaker_id.py`)

**Purpose**: Cluster the per-file diarization labels into **global** speaker IDs using
voice embeddings.

**Input**: `data/filtered/` WAVs + per-file diarization labels (from stage 3)
**Output**: `metadata/speaker_clusters.json`

```python
import torchaudio, numpy as np
from speechbrain.inference import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                     savedir="models/spkrec")

def get_embedding(wav_path):
    sig, _ = torchaudio.load(str(wav_path))         # ECAPA model handles resampling
    return clf.encode_batch(sig).squeeze().numpy()

def cluster_speakers(embs, paths, source, distance_threshold=0.6):
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold,
                                     linkage="average", metric="cosine").fit_predict(np.stack(embs))
    return {str(p): f"{source}_{l:03d}" for p, l in zip(paths, labels)}
```

**Fallback**: if embedding/clustering is unavailable, assign `speaker_id` from
`{source}_{diarization_label}` per file. Worse diversity estimate, but never corrupts
the manifest. (Speaker IDs are per-source; cross-source dedup is out of scope unless
explicitly added — document in `DECISIONS.md` if implemented.)

---

## Stage 9 — Manifest Generation (`09_manifest.py`)

**Input**: filtered WAVs + verified + transcript + jyutping + speaker clusters
**Output**: `metadata/manifest.jsonl`, `train.jsonl` (95%), `val.jsonl` (5%)

```python
import hashlib, soundfile as sf
from datetime import date

def build_entry(wav_path, verified, transcript, jyutping, speaker_id, src):
    p = wav_path.resolve()
    assert str(p).startswith("/mnt/Drive3/")
    info = sf.info(str(p))
    assert info.samplerate == 48000, f"Not 48 kHz: {p}"
    return {
        "id":             hashlib.md5(str(p).encode()).hexdigest()[:12],
        "audio_path":     str(p),
        "source":         src["source"], "source_url": src["url"],
        "program":        src.get("program",""), "domain": src.get("domain","other"),
        "text":           verified["text"],
        "text_verified":  verified.get("text_verified", False),
        "asr_candidates": transcript["candidates"],
        "asr_agreement":  transcript["asr_agreement"],
        "jyutping":       jyutping["jyutping"],
        "duration_sec":   round(info.frames/info.samplerate, 3),
        "sample_rate":    info.samplerate,
        "speaker_id":     speaker_id,
        "gender":         src.get("gender","unknown"), "style": src.get("style","formal"),
        "snr_db":         round(transcript.get("snr_db",0.0),1),
        "dnsmos":         round(transcript.get("dnsmos",0.0),2),
        "english_ratio":  round(transcript.get("english_ratio",0.0),3),
        "created_at":     str(date.today()),
    }
```

**Train/Val split**: stratify by `source`; no `speaker_id` in both splits; 95/5 by
duration.
```python
from sklearn.model_selection import GroupShuffleSplit
tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
              .split(entries, groups=[e["speaker_id"] for e in entries]))
```

---

## Stage 10 — Dataset Report (`10_report.py`)

**Output**: `metadata/DATASET_REPORT.md`

Must include: total hours (train/val/total); unique speakers; per-source breakdown;
duration / DNSMOS / SNR percentiles (p5,p25,p50,p75,p95); Jyutping tone distribution
(1–6); domain / style / gender breakdown; **`text_verified` coverage**; **sample-rate
check**; acceptance-criteria checklist (pass/fail).

**Final validation**:
```bash
grep -c "/mnt/d/" metadata/manifest.jsonl   # must be 0
grep -c "/mnt/c/" metadata/manifest.jsonl   # must be 0
python -c "
import json, soundfile as sf, os
miss=low=unver=0
for l in open('metadata/manifest.jsonl'):
    e=json.loads(l)
    if not os.path.exists(e['audio_path']): miss+=1; continue
    if sf.info(e['audio_path']).samplerate < 48000: low+=1
    if not e.get('text_verified'): unver+=1
print(f'missing={miss} below48k={low} unverified={unver}')   # all should be 0
"
```
