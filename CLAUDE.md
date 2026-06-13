# canto-hk-speech-pipeline — HK Cantonese TTS Dataset

## Project in One Sentence

Collect, clean, and annotate **100–500 hours** of **Hong Kong Cantonese** speech from multiple public sources to build a self-owned, high-quality TTS training corpus with **100+ unique speakers**.

---

## Session Start Protocol

Every session, in this order before writing any code:

```bash
# 1. Read progress from last session
cat PROGRESS.md

# 2. Check disk space (dataset will be 50–200 GB)
df -h /mnt/Drive3/

# 3. Check GPU availability (needed for WhisperX and DNSMOS)
nvidia-smi --query-gpu=name,memory.free --format=csv

# 4. Check what scripts already exist
ls scripts/

# 5. Check what data has been collected so far
ls data/raw/ data/segments/ data/filtered/ 2>/dev/null
```

Update `PROGRESS.md` at the end of every session using the format at the bottom of this file.

---

## Directory Layout

```
canto-hk-speech-pipeline/
├── CLAUDE.md                   ← you are here (read every session)
├── PROGRESS.md                 ← session log (read first, update last)
├── DECISIONS.md                ← technical decision log
│
├── docs/
│   ├── PIPELINE_SPEC.md        ← full pipeline implementation details
│   ├── QUALITY_SPEC.md         ← filtering thresholds and rationale
│   ├── MANIFEST_SCHEMA.md      ← manifest field definitions + examples
│   ├── KNOWN_ISSUES.md         ← all known failure modes from prior attempts
│   └── SOURCE_GUIDE.md         ← how to research and add new sources
│
├── sources/
│   ├── rthk_sources.yaml       ← RTHK programs to download
│   ├── youtube_channels.yaml   ← YouTube channels to download
│   └── podcast_sources.yaml    ← Podcast RSS feeds
│
├── scripts/                    ← pipeline scripts (you write these)
│   ├── 01_discover.py          stage 1: assess sources, estimate hours
│   ├── 02_download.py          stage 2: download audio (keep highest quality)
│   ├── 03_segment.py           stage 3: diarization + VAD segmentation
│   ├── 04_transcribe.py        stage 4: multi-ASR transcription
│   ├── 05_calibrate.py         stage 5: manual calibration tooling (human-in-loop)
│   ├── 06_filter.py            stage 6: quality filtering
│   ├── 07_g2p.py               stage 7: G2P → Jyutping (on verified text)
│   ├── 08_speaker_id.py        stage 8: cross-file speaker clustering
│   ├── 09_manifest.py          stage 9: build JSONL manifest
│   └── 10_report.py            stage 10: dataset statistics report
│
├── data/
│   ├── raw/                    ← downloaded originals (by source, ≥48 kHz)
│   │   ├── rthk/
│   │   ├── youtube/
│   │   ├── podcast/
│   │   └── hktv/
│   ├── segments/               ← single-speaker clips (48 kHz mono WAV master)
│   ├── filtered/               ← QC-passed clips (ready for training)
│   └── final/                  ← merged manifest + final audio set
│
└── metadata/
    ├── downloaded.jsonl        ← download log
    ├── filter_report.json      ← per-stage filter pass rates
    ├── g2p_report.json         ← G2P validation summary
    ├── speaker_report.json     ← speaker clustering results
    ├── manifest.jsonl          ← full manifest (all splits)
    ├── train.jsonl             ← training split (95%)
    ├── val.jsonl               ← validation split (5%)
    └── DATASET_REPORT.md       ← human-readable summary (final output)
```

---

## Pipeline Overview

Run stages in order. Each script is idempotent (safe to re-run; skips already-processed files).

| Stage | Script | Input | Output | Detail |
|-------|--------|-------|--------|--------|
| 1 | `01_discover.py` | `sources/*.yaml` | discovery log | Survey available content; estimate hours |
| 2 | `02_download.py` | `sources/*.yaml` | `data/raw/` | yt-dlp / RSS download, keep best audio (≥48 kHz) |
| 3 | `03_segment.py` | `data/raw/` | `data/segments/` | pyannote diarization + Silero VAD → single-speaker 3–20s clips @ 48 kHz |
| 4 | `04_transcribe.py` | `data/segments/` | `*.transcript.json` | **Multiple** ASR models → candidate transcripts + agreement score |
| 5 | `05_calibrate.py` | transcripts | `*.verified.json` | **Human-in-loop**: surface low-agreement segments, produce canonical text |
| 6 | `06_filter.py` | segments + verified | `data/filtered/` | DNSMOS / SNR / duration / language ratios |
| 7 | `07_g2p.py` | filtered + verified text | `*.jyutping.json` | text norm → Jyutping with tones via **canto-hk-g2p** (on **verified** text) |
| 8 | `08_speaker_id.py` | `data/filtered/` | speaker clusters | ECAPA-TDNN embedding + cross-file clustering |
| 9 | `09_manifest.py` | all above | `metadata/manifest.jsonl` | unified JSONL per schema |
| 10 | `10_report.py` | manifest | `DATASET_REPORT.md` | statistics and quality summary |

All scripts accept `--source [rthk|youtube|podcast|hktv|all]` and `--dry-run`.
Full implementation spec: `docs/PIPELINE_SPEC.md`

**Audio rate strategy**: store every segment as a **48 kHz mono master** (downsampling is lossy and irreversible — never store below this). VAD, diarization, ASR and DNSMOS all internally need 16 kHz; generate a *transient* 16 kHz copy for those tools and discard it. The 48 kHz master is what TTS training will consume (NeuCodec=24 kHz, MOSS-Nano=48 kHz, F5-TTS=24 kHz — all need ≥24 kHz).

**ASR strategy**: run **several** ASR models per segment (e.g. a Cantonese fine-tuned Whisper + base `large-v3` with `language="zh"` + prompt). Store every candidate transcript. A human then calibrates (stage 5). Never auto-trust a single ASR output, and **never use `language="yue"`** — it triggers decoder collapse on large-v3 (see `docs/KNOWN_ISSUES.md §9`).

---

## Data Sources

### Existing RTHK Data (Already Collected — Do Not Re-download)

Prior pipeline collected segments from `創科新里程` (tech documentaries). Audio files are at:
```
../cantonese-tts-old/data/segments/
```
Manifests (with Windows paths — need remapping) are at:
```
../cantonese-tts-old/data/dataset/cantonese_manifest_fixed.jsonl
```

**What to do with existing RTHK data:**
1. Remap Windows paths (`/mnt/d/cantonese-tts/` → find actual Linux location)
2. Re-run quality filtering with new thresholds (DNSMOS ≥ 3.0, not old threshold)
3. Re-run G2P with new validated pipeline (old Jyutping may have concatenation bug)
4. Integrate passing segments into new manifest with full schema

**What to add — RTHK expansion:**
See `sources/rthk_sources.yaml` for target programs. RTHK publishes content on YouTube at `https://www.youtube.com/@rthkhongkong` and various sub-channels. Target programs with multiple presenters and diverse content styles.

### New Sources to Add

| Source | Config file | Priority | Est. hours available |
|--------|-------------|----------|----------------------|
| RTHK (expansion) | `sources/rthk_sources.yaml` | High | 200–500h |
| YouTube HK Cantonese | `sources/youtube_channels.yaml` | High | varies |
| HK Cantonese Podcasts | `sources/podcast_sources.yaml` | Medium | 50–200h |
| Other HK TV | `sources/hktv_sources.yaml` | Low | TBD |

Source research guide: `docs/SOURCE_GUIDE.md`

---

## Hard Constraints

These are non-negotiable. Never override for any technical reason.

1. **HK Cantonese only.** Reject audio that is primarily Mandarin, Guangzhou Cantonese, or overseas Cantonese. Use language detection to filter.

2. **No WenetSpeech-Yue or third-party datasets.** This corpus is self-owned and self-collected. Only publicly accessible web sources downloaded directly.

3. **No code from prior project directories.** The following are deprecated — do not `cp` or `import` from them:
   ```
   ../cantonese-tts/        ../cantonese-tts-old/    ../CantoNeu/
   ../gemini-hk-canto-tts/  ../gemma-hermes--tts/    ../qwen36-hermes-tts/
   ```
   Their **lessons** are encoded in `docs/KNOWN_ISSUES.md`. Read that instead.

4. **Absolute Linux paths only.** Every `audio_path` in every manifest must be an absolute path under your data root (e.g. `/mnt/Drive3/Development/AI-ML/canto-hk-speech-pipeline/`). No relative paths, no Windows-style paths (`/mnt/d/`, `/mnt/c/Users/`).

5. **Single-speaker, VAD-based segmentation only.** Never hard-cut audio at a fixed duration. Segment at natural speech pause boundaries (Silero VAD) *within* diarization-detected single-speaker turns, so no clip spans a speaker change. Target 3–20 seconds. See `docs/KNOWN_ISSUES.md §2, §10`.

6. **48 kHz mono master, never lower.** Store every segment at 48 kHz. Downsampling is irreversible — a 16 kHz corpus would be unusable for every modern TTS codec. Create transient 16 kHz copies for VAD/ASR/DNSMOS only. See `docs/KNOWN_ISSUES.md §11`.

7. **Never `language="yue"` in Whisper.** It causes decoder collapse on large-v3. Use a Cantonese fine-tuned model and/or `language="zh"` with a written-Cantonese prompt. Run multiple ASR models; a human calibrates the canonical text. See `docs/KNOWN_ISSUES.md §9`.

8. **Validate Jyutping format explicitly.** Every Jyutping output must be validated: each space-separated token must match `^[a-z]+[1-6]$`. Reject segments where > 5% of tokens fail. Run G2P on the **human-verified** text, never raw single-ASR output. See `docs/KNOWN_ISSUES.md §1`.

---

## Critical Known Issues (Summary)

Full details in `docs/KNOWN_ISSUES.md`. These three will silently corrupt your data if ignored:

**Issue 1 — G2P tool history**
`pycantonese` ≥ 4.1.0 concatenates Jyutping syllables without spaces in ~47% of cases — do not use.
`ToJyutping` handles English letter-by-letter (`[S][E][N][D]`) causing alignment problems in TTS training.
→ Use **canto-hk-g2p** (`github.com/typangaa/canto-hk-g2p`): Rust-core, passes English through unchanged (1:1 alignment), expands numbers/dates. Mandatory regex validation still required: each token must match `^[a-z]+[1-6]$`.

**Issue 2 — Hard audio cut at fixed duration**
Prior pipeline cut all audio at exactly 15s, leaving 44.5% of segments mid-sentence.
→ Use Silero VAD. Never cut with `ffmpeg -t`. Verify cuts fall at silence boundaries.

**Issue 3 — Windows absolute paths in manifests**
Old manifests reference `/mnt/d/` and `/mnt/c/Users/TY_Windows/` — these break on Linux.
→ Always write `/mnt/Drive3/...` paths. Run `grep -c "/mnt/d/" metadata/manifest.jsonl` before finalising; must return 0.

**Issue 9 — Whisper `yue` decoder collapse**
Forcing `language="yue"` on large-v3 produces repetition loops / garbage. → Cantonese fine-tuned model and/or `language="zh"` + prompt; multi-ASR + human calibration.

**Issue 11 — 16 kHz dead-end**
A 16 kHz corpus cannot train any modern TTS codec (all need ≥24 kHz). → Store 48 kHz master; downsample only transiently.

---

## Script Conventions

Follow these conventions in all scripts you write:

```python
# --- File header ---
#!/usr/bin/env python3
"""
scripts/NN_name.py
One-line description.
Usage: python scripts/NN_name.py --source [rthk|youtube|podcast|all] [--dry-run]
"""

# --- Idempotency: skip already-processed files ---
if output_path.exists():
    log.info(f"Skip (already done): {output_path}")
    continue

# --- Logging: always write to metadata/logs/{script_name}.log ---
import logging
log_path = Path("metadata/logs") / f"{Path(__file__).stem}.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)

# --- End-of-script summary ---
print(f"\nDone: {processed} processed, {skipped} skipped, {failed} failed")
print(f"Log: {log_path}")
```

**Audio format for stored files**: 48 kHz, mono, 16-bit (or 24-bit) PCM WAV — this is the master that survives into `data/filtered/`. Use `soundfile` or `torchaudio` for I/O. For VAD / diarization / ASR / DNSMOS, generate a transient 16 kHz copy in memory or `/tmp` and discard it — never overwrite the 48 kHz master with a downsampled version.

**File naming**:
```
raw:      {YYYYMMDD}_{program_slug}_{video_id}.webm
segments: {YYYYMMDD}_{program_slug}_{video_id}_seg{N:05d}.wav
filtered: same as segments (symlink or copy into filtered/)
```

**Never print sensitive info** (API keys, full file paths in error messages visible to log aggregators).

---

## Quality Thresholds

| Filter | Threshold | Notes |
|--------|-----------|-------|
| Duration | 3–20 seconds | VAD-based, not hard cut |
| DNSMOS P.835 (OVRL) | ≥ 3.0 | `speechmos` → `dnsmos.run(wav16k, sr=16000)['ovrl_mos']`; verify in [1.0, 5.0] |
| SNR | ≥ 25 dB | |
| ASR agreement | ≥ 0.80 char-overlap | Cross-model agreement; low agreement → flag for manual calibration |
| Mandarin ratio | ≤ 0.15 | Fraction of Mandarin-only words |
| English ratio | ≤ 0.30 | Allow code-switching, reject English-dominant |
| Min characters | ≥ 5 | After punctuation removal |
| Max characters | ≤ 150 | Guard against ASR hallucination |
| Single speaker | required | Diarization must report exactly 1 speaker per segment |

Full threshold rationale: `docs/QUALITY_SPEC.md`

---

## Manifest Schema (Quick Reference)

```jsonc
{
  "id":             "rthk_20230325_seg00000",      // unique, stable ID
  "audio_path":     "/absolute/path/to/seg.wav",   // absolute Linux path, 48 kHz master
  "source":         "rthk",                         // rthk|youtube|podcast|hktv
  "source_url":     "https://youtube.com/...",
  "program":        "創科新里程",
  "domain":         "documentary",                  // see domain enum below
  "text":           "心臟病中風這些常見的心腦血管疾病",  // CANONICAL, human-verified text
  "text_verified":  true,                           // false = ASR-only, not yet calibrated
  "asr_candidates": [                               // every ASR model's raw output
    {"model": "simonl0909/whisper-large-v2-cantonese", "text": "...", "confidence": 0.88},
    {"model": "openai/whisper-large-v3+zh",            "text": "...", "confidence": 0.82}
  ],
  "asr_agreement":  0.91,                            // char-overlap across candidates
  "jyutping":       "sam1 zong6 beng6 zung1 fung1", // from verified text; tone digits 1-6
  "duration_sec":   6.97,
  "sample_rate":    48000,
  "speaker_id":     "rthk_001",
  "gender":         "male",                         // male|female|unknown
  "style":          "formal",                       // formal|casual|narration|interview
  "snr_db":         35.2,
  "dnsmos":         3.8,                             // speechmos ovrl_mos, range [1,5]
  "english_ratio":  0.02,
  "created_at":     "2026-06-09"
}
```

Domain enum: `documentary | news | talk_show | podcast | drama | vlog | educational | other`
Full schema with validation rules: `docs/MANIFEST_SCHEMA.md`

---

## Acceptance Criteria

Dataset is ready for TTS training when ALL pass:

| Criterion | Target |
|-----------|--------|
| Total clean hours | ≥ 100 h |
| Unique speakers | ≥ 100 |
| Sample rate | 48000 Hz, 100% of segments |
| Human-verified text (`text_verified`) | 100% |
| DNSMOS median (p50) | ≥ 3.2 |
| SNR median (p50) | ≥ 30 dB |
| Jyutping valid rate | ≥ 99% |
| Single-speaker segments | 100% |
| Distinct sources | ≥ 3 |
| Distinct domains | ≥ 3 |
| Duration in 3–20s range | 100% |
| Windows paths in manifest | 0 |

Check with: `python scripts/10_report.py`

---

## External Dependencies

| Tool | Source | Purpose |
|------|--------|---------|
| canto-hk-g2p | [github.com/typangaa/canto-hk-g2p](https://github.com/typangaa/canto-hk-g2p) | Stage 7 G2P (Rust+PyO3) |
| faster-whisper | PyPI | Stage 4 ASR |
| pyannote.audio | PyPI + HF model terms | Stage 3 diarization |
| speechbrain | PyPI | Stage 8 speaker embedding |
| speechmos | PyPI | Stage 3b / 6 DNSMOS quality gate |
| yt-dlp | PyPI / system | Stage 2 download |

**CHECKPOINT**: When `DATASET_REPORT.md` is written and all criteria pass, stop and present it for human review before the corpus is used for TTS model training.

---

## PROGRESS.md Update Format

```markdown
## Session YYYY-MM-DD
**Stage**: [e.g. Stage 4 — WhisperX transcription, RTHK source]
**Completed**:
- specific item
**Stats**: X h downloaded, Y h segmented, Z h passed QC, N speakers identified
**Next**:
- specific next action
**Blockers** (human input needed):
- item, or "none"
```
