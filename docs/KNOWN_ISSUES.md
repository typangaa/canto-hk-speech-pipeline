# Known Issues — canto-hk-speech-pipeline

> All issues below were confirmed in prior pipeline attempts in the following directories:
> `../cantonese-tts-old/`, `../CantoNeu/`, `../gemini-hk-canto-tts/`
> Do not repeat these mistakes. Each entry includes the root cause and required mitigation.

---

## §1 — pycantonese Jyutping Concatenation Bug

**Severity**: Critical — silently corrupts G2P output for ~47% of entries.

**What happens**: `pycantonese` version ≥ 4.1.0 returns Jyutping strings where syllables are concatenated without spaces.

```python
# Expected:
"sam1 zong6 beng6"

# Actual (buggy):
"sam1zong6beng6"    # or "sam1 zong6beng6"  (inconsistent)
```

**Why it is silent**: The pipeline accepted the output without validation. Training proceeded with malformed phoneme sequences. Audio quality degraded without any error.

**Root cause**: A regression in pycantonese's internal tokeniser. The bug was discovered only by manually inspecting output samples.

**Required mitigation**: After every G2P call, validate each output string:

```python
import re

JYUTPING_TOKEN = re.compile(r'^[a-z]+[1-6]$')

def validate_jyutping(jyutping_str: str) -> tuple[bool, float]:
    """Returns (is_valid, valid_fraction)."""
    if not jyutping_str or not jyutping_str.strip():
        return False, 0.0
    tokens = jyutping_str.strip().split()
    valid = sum(1 for t in tokens if JYUTPING_TOKEN.match(t))
    return valid / len(tokens) >= 0.95, valid / len(tokens)

# Usage: reject segment if valid_fraction < 0.95
ok, frac = validate_jyutping(output)
if not ok:
    log.warning(f"Jyutping validation failed ({frac:.1%} valid): {output!r}")
    continue
```

**Alternative**: Use `ToJyutping` as the primary G2P tool. Its output format is more consistent. Compare both tools on 100 sample sentences before committing.

```bash
pip install ToJyutping pycantonese
python -c "
import ToJyutping
print(ToJyutping.get_jyutping_text('心臟病中風'))
# Expected: sam1 zong6 beng6 zung1 fung1
"
```

---

## §2 — Hard Audio Cut at Fixed Duration (15 s)

**Severity**: High — degrades TTS model's end-of-sentence behaviour.

**What happened**: The segmentation script capped every audio clip at exactly 15 seconds using `ffmpeg -ss {start} -t 15`. Result: 44.5% of segments end mid-word or mid-sentence.

**Why it matters for TTS training**: The model never sees a complete sentence followed by silence in ~45% of training samples. It learns that sentences can end at any arbitrary point, causing runaway (non-stopping) generation at inference time.

**Required mitigation**: Use VAD (Voice Activity Detection) to find natural pause boundaries:

```python
import torch

# Load Silero VAD once
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                               model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

def segment_by_vad(audio_path: str, min_dur=3.0, max_dur=20.0):
    """Yield (start_sec, end_sec) tuples at speech boundaries."""
    wav = read_audio(audio_path, sampling_rate=16000)
    timestamps = get_speech_timestamps(wav, model, sampling_rate=16000,
                                       min_silence_duration_ms=500,
                                       min_speech_duration_ms=int(min_dur * 1000))
    # Merge short segments, split long ones at silence points
    # ... (see PIPELINE_SPEC.md for full implementation)
    return segments
```

**Hard rule**: Never call `ffmpeg` with a fixed `-t` duration for segmentation. Always cut at silence boundaries. Assert after segmentation: `assert all(3 <= dur <= 20 for dur in durations)`.

---

## §3 — Windows Absolute Paths in Manifests

**Severity**: High — manifest becomes unusable on Linux without path surgery.

**What happened**: Manifests were generated under WSL2. The `audio_path` fields referenced:
- `/mnt/d/cantonese-tts/data/segments/...` (Windows D: drive)
- `/mnt/c/Users/TY_Windows/Documents/...` (Windows C: drive user directory)

These paths work on the original machine only. Moving to another machine or remounting WSL2 breaks all paths.

**Required mitigation**:

```python
from pathlib import Path

# Always resolve to absolute Linux path before writing to manifest
audio_path = Path(audio_path).resolve()
assert str(audio_path).startswith("/mnt/Drive3/"), \
    f"Path must be under /mnt/Drive3/, got: {audio_path}"
```

**Validation check before finalising manifest**:

```bash
# Must return 0
grep -c "/mnt/d/" metadata/manifest.jsonl
grep -c "/mnt/c/" metadata/manifest.jsonl
grep -c "Users/TY_Windows" metadata/manifest.jsonl
```

**For remapping existing RTHK data**: The old segments are physically located at:
```bash
# Find actual Linux location of old RTHK segments
find /mnt/Drive3 -path "*/cantonese-tts-old/data/segments/*.wav" | head -5
```
Map old paths to new paths when re-ingesting:
```python
old_prefix = "/mnt/d/cantonese-tts/"
new_prefix = "/mnt/Drive3/Development/AI-ML/cantonese-tts-old/"
new_path = old_path.replace(old_prefix, new_prefix)
```

---

## §4 — DNSMOS Not Correctly Configured

**Severity**: Medium — quality filtering passes bad audio silently.

**What happened**: DNSMOS scoring was sometimes run with incorrect model weights or wrong input preprocessing, producing scores outside the valid [1.0, 5.0] range. Bad audio passed the filter because scores appeared to be high (out-of-range values).

**Required mitigation**: After computing DNSMOS on the first 100 files, assert sanity:

```python
scores = [compute_dnsmos(f) for f in first_100_files]
assert all(1.0 <= s <= 5.0 for s in scores), "DNSMOS scores out of range — check model setup"
median = sorted(scores)[50]
assert 2.0 <= median <= 4.5, f"DNSMOS median {median:.2f} is suspicious — check preprocessing"
print(f"DNSMOS sanity check passed. Median: {median:.2f}")
```

**Recommended DNSMOS implementation** (verified API — do not invent your own):

```bash
pip install speechmos librosa onnxruntime-gpu   # onnxruntime (CPU) if no GPU
```

```python
import librosa
from speechmos import dnsmos

def score_dnsmos(wav_path: str) -> float:
    # IMPORTANT: speechmos DNSMOS is trained for 16 kHz mono input.
    # Always feed a 16 kHz copy, even though the stored master is 48 kHz.
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    result = dnsmos.run(audio, sr=16000)
    # result keys: filename, ovrl_mos, sig_mos, bak_mos, p808_mos
    return result["ovrl_mos"]   # overall quality, range [1, 5]
```

Alternative: `torchmetrics`' `DeepNoiseSuppressionMeanOpinionScore` (also wraps the
Microsoft ONNX models). Either is fine — but the package is `speechmos`, NOT a package
named `dnsmos`, and the score key is `ovrl_mos`, NOT `OVRL`.

---

## §5 — NeuTTS Air Base Model Silent Updates

**Severity**: Medium — affects TTS training phase, not data pipeline, but document here for awareness.

**What happened**: During TTS training (not data collection), `neuphonic/neutts-air` on HuggingFace was silently updated between training sessions. A LoRA adapter trained against version N became incompatible with version N+1. Training loss jumped from 2.37 to 5.02 overnight.

**Impact on data pipeline**: None directly. But the data pipeline should record model versions used (for WhisperX, DNSMOS models) so that if a model is updated, the affected batches can be re-processed.

**Required mitigation**: Pin model versions in config files:

```yaml
# In sources config or pipeline config
models:
  whisperx: "large-v3"           # pin to specific version
  whisperx_commit: "abc123..."   # optionally pin commit hash
  dnsmos: "microsoft/DNSMOS"
  speaker_embed: "speechbrain/spkrec-ecapa-voxceleb"
```

Log model versions at the start of each pipeline run:

```python
import transformers
log.info(f"transformers version: {transformers.__version__}")
log.info(f"WhisperX model: {MODEL_ID}")
```

---

## §6 — Narrow Speaker Diversity from Single Source

**Severity**: Medium — limits TTS model generalisation and voice cloning quality.

**What happened**: All RTHK content was from `創科新里程` (tech/innovation documentary series). This series uses a small pool of professional narrators with a specific documentary style. The TTS model trained on this data:
- Produces "documentary narrator" prosody for all inputs
- Has poor speaker diversity for voice cloning
- Fails on casual speech inputs

**Required mitigation**:
1. Actively track domain and style per segment (CLAUDE.md §manifest schema)
2. Set minimum diversity targets before accepting the corpus:
   - ≥ 3 distinct sources
   - ≥ 3 distinct domains
   - ≥ 3 distinct styles
3. If a single source or domain accounts for > 50% of total hours, actively seek more diverse content

**Detection**:
```bash
# Check domain distribution in manifest
python -c "
import json
from collections import Counter
with open('metadata/manifest.jsonl') as f:
    domains = Counter(json.loads(l)['domain'] for l in f)
for d, n in domains.most_common():
    print(f'{d}: {n}')
"
```

---

## §7 — HuggingFace Trainer `trainer_state.json` Resume Bug

**Severity**: Medium — affects TTS training phase (not data pipeline). Document for awareness.

**What happened**: When resuming training from a checkpoint, HuggingFace `Trainer` reads `save_steps` from `trainer_state.json` inside the checkpoint directory rather than from the current `TrainingArguments`. This causes:
- Unexpected checkpoint save frequencies after resume
- Potential overwriting of good checkpoints

**Required mitigation** (apply when building TTS training scripts in the separate training project):
```python
# Before resuming, patch trainer_state.json
import json
state_path = checkpoint_dir / "trainer_state.json"
state = json.loads(state_path.read_text())
state["save_steps"] = training_args.save_steps
state_path.write_text(json.dumps(state, indent=2))
```

---

## §8 — Out-of-Memory with Large Vocabulary

**Severity**: Medium — affects TTS training phase. Document for awareness.

**What happened**: NeuCodec uses a vocabulary of 65,536 speech tokens + ~200 text/special tokens = ~65,736 total. Loading embeddings for this vocabulary with batch_size > 1 caused GPU OOM on 24 GB VRAM cards.

**Required mitigation** (apply in training project):
```python
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # effective batch = 8
    gradient_checkpointing=True,
)
# Also set:
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

---

## §9 — Whisper `yue` Decoder Collapse / Cantonese-Mandarin Confusion

**Severity**: High — corrupts transcripts, which corrupts G2P, which mistrains the TTS.

**Two distinct failure modes:**

**(a) `language="yue"` causes decoder collapse.** This corrects earlier guidance.
Forcing `language="yue"` on Whisper `large-v3` frequently produces repetition loops,
garbage tokens, and unusable output. Whisper saw far more `zh` than `yue` text in
training, and written Cantonese (粵語白話文) is under-represented. Confirmed in
practical testing (Jan 2026). **Do not use `language="yue"`.**

**(b) `language="zh"` transcribes Cantonese audio in Mandarin character forms**
(e.g. 係 → 是, 冇 → 没有). This makes G2P wrong because those characters carry Mandarin
readings.

**Required mitigation — multi-ASR + human calibration (matches project ASR strategy):**

1. Run **several** ASR models per segment and store every candidate:
   - A Cantonese fine-tuned Whisper, e.g. `simonl0909/whisper-large-v2-cantonese`
     (~7.65% CER) or `khleeloo/whisper-large-v3-cantonese`. These output written
     Cantonese and avoid the collapse.
   - Base `openai/whisper-large-v3` with `language="zh"` **plus** an `initial_prompt`
     seeded with written-Cantonese text to bias toward 粵語白話文 (never `yue`).

   ```python
   model.transcribe(
       audio, language="zh",
       initial_prompt="以下係廣東話口語，請用粵語白話文書寫，例如：係、唔係、冇、喺、佢哋、嘅、嗰、嚟。"
   )
   ```

2. Compute cross-model **agreement** (character-level overlap). High agreement →
   likely correct; low agreement → flag for the human calibration stage (stage 5).

3. Detect Mandarin-dominant output and prefer the Cantonese-marker-rich candidate:
   ```python
   CANTONESE_MARKERS = set("係冇喺佢哋嘅嗰嚟咁啩囉𠮶")
   def cantonese_marker_count(text: str) -> int:
       return sum(1 for c in text if c in CANTONESE_MARKERS)
   ```

4. **Never auto-finalise a single ASR output.** The canonical `text` field is set by a
   human in stage 5 (`05_calibrate.py`), using the candidates as references.

---

## §10 — Multi-Speaker Segments (No Diarization)

**Severity**: High — corrupts speaker labels and degrades voice-cloning quality.

**What happens**: Silero VAD detects silence, not speaker identity. In multi-speaker
programs (城市論壇, 鏗鏘集, 頭條新聞, interviews), a single VAD segment can span a
speaker change. A TTS training clip that contains two voices teaches the model an
incoherent target, and `speaker_id` clustering becomes meaningless.

**Required mitigation**: Run speaker **diarization** before/with VAD so every segment
contains exactly one speaker.

```python
# pyannote diarization (HF token required for the pretrained pipeline)
from pyannote.audio import Pipeline
dia = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                               use_auth_token="hf_xxx")

def single_speaker_turns(wav_path: str) -> list[tuple[float, float, str]]:
    """Return [(start_sec, end_sec, local_speaker_label), ...]."""
    diarization = dia(wav_path)
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, speaker))
    return turns

# Pipeline: diarize file → for each single-speaker turn, run Silero VAD WITHIN the turn
# to cut 3–20 s clips. Never let a clip cross a turn boundary.
```

Record the per-file local speaker label on each segment; stage 8 clusters these across
files (via ECAPA embeddings) into global `speaker_id`s. Reject any segment where
diarization reports overlapping speech / more than one speaker.

WhisperX also bundles diarization (`--diarize`), which can be reused instead of calling
pyannote directly — either is acceptable.

---

## §11 — 16 kHz Storage Dead-End

**Severity**: Critical — silently caps the entire corpus below TTS-usable quality.

**What happened**: The prior RTHK pipeline stored everything at 16 kHz (it was built for
ASR-style work). 16 kHz is telephone-band. Every modern neural TTS codec needs ≥24 kHz:
NeuCodec (NeuTTS) = 24 kHz, F5-TTS = 24 kHz, MOSS-Nano (VieNeu v3) = 48 kHz. A 16 kHz
corpus cannot train any of them, and upsampling cannot recover the missing high
frequencies — the data is permanently capped.

**Required mitigation**:
1. Download the **best available audio** (YouTube serves 48 kHz AAC; keep it).
2. Store every segment as a **48 kHz mono master** in `data/segments/` and
   `data/filtered/`.
3. VAD, diarization, ASR, and DNSMOS all need 16 kHz — generate a *transient* 16 kHz
   copy in memory or `/tmp` for those tools, and **never** write it over the master.

```python
import librosa, soundfile as sf

# WRONG — destroys the master:
# sf.write(master_path, librosa.load(master_path, sr=16000)[0], 16000)

# RIGHT — transient downsample for a tool, master untouched:
audio16, _ = librosa.load(master_path, sr=16000, mono=True)   # in-memory only
score = score_dnsmos_from_array(audio16)                      # use, then discard
```

**Validation**:
```bash
python -c "
import json, soundfile as sf
bad = 0
for l in open('metadata/manifest.jsonl'):
    p = json.loads(l)['audio_path']
    if sf.info(p).samplerate < 48000: bad += 1
print('Segments below 48 kHz:', bad)   # must be 0
"
```

---

## §12 — Source Licensing / Usage Scope

**Severity**: Medium — process/compliance, not data corruption.

**What it is**: "Self-owned" here means *self-collected*, not *licensed for
redistribution*. Audio downloaded from RTHK / YouTube / podcasts remains under its
original rights holders. Building a private corpus to train a model is a different act
from redistributing the source audio.

**Required practice**:
- Record source URLs (already in the manifest `source_url`) so provenance is traceable.
- Treat the corpus as **internal research / model-training only**. Do not redistribute
  the raw audio files publicly.
- Note the usage scope and any per-source terms in `DECISIONS.md`.
- Prefer public-broadcaster content (RTHK) where public-access intent is clearest.

---

*Last updated: 2026-06-09. Add new issues as they are discovered during this pipeline run.*
