# Manifest Schema — canto-hk-speech-pipeline

> Complete field-by-field specification for `metadata/manifest.jsonl`.
> Each line is a valid JSON object (not JSONC — no comments in the actual file).
> This schema must be consistent across all pipeline stages.

---

## Full Schema

```json
{
  "id":             "rthk_20230325_seg00000",
  "audio_path":     "/mnt/Drive3/canto/segments/rthk_20230325_canto_science_seg00000.flac",
  "source":         "rthk",
  "source_url":     "https://www.youtube.com/watch?v=XXXXXXXXXXX",
  "program":        "創科新里程",
  "domain":         "documentary",
  "text":           "心臟病中風這些常見的心腦血管疾病",
  "text_verified":  true,
  "asr_candidates": [
    {"model": "qwen3_asr",    "text": "心臟病中風這些常見的心腦血管疾病", "confidence": 0.88},
    {"model": "sense_voice",  "text": "心臟病中風這些常見的心腦血管疾病", "confidence": 0.82}
  ],
  "asr_agreement":  0.96,
  "jyutping":       "sam1 zong6 beng6 zung1 fung1 ze2 se1 soeng4 gin3 dik1 sam1 nou5 hyut3 gun2 zat6 beng6",
  "duration_sec":   6.97,
  "sample_rate":    48000,
  "speaker_id":     "rthk_001",
  "gender":         "male",
  "style":          "formal",
  "snr_db":         35.2,
  "dnsmos":         3.8,
  "english_ratio":  0.02,
  "created_at":     "2026-06-09"
}
```

---

## Field Reference

### `id` — string, required, unique

Stable segment identifier. Must not change if the file is regenerated.

**Format**: `{source}_{YYYYMMDD}_{slug}_{N:05d}`

**Generation**:
```python
import hashlib
def generate_id(wav_path: Path) -> str:
    """Stable ID from absolute path — survives re-runs."""
    return hashlib.md5(str(wav_path.resolve()).encode()).hexdigest()[:12]
```

**Constraints**: Unique across the entire manifest. If duplicates appear, the pipeline has a bug. Detection:
```bash
python -c "
import json
from collections import Counter
with open('metadata/manifest.jsonl') as f:
    ids = [json.loads(l)['id'] for l in f]
dups = [k for k,v in Counter(ids).items() if v > 1]
print(f'Duplicate IDs: {len(dups)}')
"
```

---

### `audio_path` — string, required

Absolute path to the segment master file.

**Constraints**:
- Must start with `/mnt/Drive2/`, `/mnt/Drive3/`, or `/mnt/Drive4/` — segments are
  3-way sharded across the three drives via `config/storage_layout.py:shard_index()`
  (deterministic `hash % 3` — never hand-pick a shard)
- File must exist and be readable
- File must be a **48 kHz mono lossless master** — FLAC for new segments, legacy 16-bit
  PCM WAV not re-encoded (`KNOWN_ISSUES.md §11`); never a downsampled or lossy copy

**Validation**:
```python
assert str(audio_path).startswith(("/mnt/Drive2/", "/mnt/Drive3/", "/mnt/Drive4/")), \
    f"Invalid path: {audio_path}"
assert audio_path.exists(), f"Missing: {audio_path}"
```

**Never use**:
- `/mnt/d/`, `/mnt/c/` — Windows-style paths
- Relative paths (`./data/...` or `../...`)
- A hand-picked shard drive — always resolve via `shard_index()`

---

### `source` — string enum, required

Which data source this segment came from.

| Value | Meaning |
|-------|---------|
| `rthk` | Radio Television Hong Kong (including YouTube channel) |
| `youtube` | YouTube channels (not RTHK) |
| `podcast` | Podcast RSS feeds |
| `hktv` | Other HK TV sources |

---

### `source_url` — string, required

Original URL where the source audio was downloaded from. Allows re-downloading if the local file is lost.

**Format**: Full URL including scheme. For RTHK/YouTube: `https://www.youtube.com/watch?v={id}`. For podcasts: the episode audio URL.

**Never use**: Local file paths or partial URLs.

---

### `program` — string, required

Name of the TV program, YouTube channel, or podcast series. In Traditional Chinese where applicable.

**Examples**: `創科新里程`, `鏗鏘集`, `財經快訊`, `毛記葵涌`, `荔枝角的事`

---

### `domain` — string enum, required

Content domain. Affects TTS model prosody generalisation; must be tracked to ensure diversity.

| Value | Description | Example programs |
|-------|-------------|-----------------|
| `documentary` | Documentary, science, history | 創科新里程, 鏗鏘集 |
| `news` | News broadcast, financial news | 財經快訊, 港聞直播 |
| `talk_show` | Talk show, variety, debate | 頭條新聞, 立場普通話 |
| `podcast` | Audio podcast | 荔枝角的事, TMHK podcast |
| `drama` | TV drama, soap opera | 家族榮耀, TVB dramas |
| `vlog` | YouTube vlog, informal | individual YouTubers |
| `educational` | Tutorial, lecture | educational channels |
| `other` | Does not fit above | |

**Diversity target**: No single domain > 50% of total hours. See `docs/QUALITY_SPEC.md`.

---

### `text` — string, required

The **canonical, human-verified** transcript in Traditional Chinese (written Cantonese,
粵語白話文). Includes code-switched English words as-is.

**Constraints**:
- Not empty
- Not padded with leading/trailing whitespace
- Punctuation may be included (it is stripped for G2P but preserved here)
- This is the text produced by human calibration (`pipe calibrate serve`), **not** raw
  ASR output. Until calibrated, an interim entry may carry an ASR candidate here with
  `text_verified: false`.

---

### `text_verified` — boolean, required

`true` once a human has confirmed `text` via `pipe calibrate serve` (a `'verified'`
decision flips `asr_agreement.text_verified` + `tiers.tier='gold'`). `false` for
interim ASR-only entries.

**Constraint**: every segment in the **final training corpus** must be `true`. The
acceptance criteria require 100% coverage.

---

### `asr_candidates` — array of objects, required

Every ASR model's raw output for this segment, kept for provenance and to support
re-calibration. Each element:

```json
{"model": "<model id, with +lang suffix if forced>", "text": "<raw output>", "confidence": 0.0}
```

**Constraints**:
- At least 2 candidates (the project runs multiple ASR models).
- `confidence` is advisory (range 0.0–1.0); do not use it as a hard gate.

---

### `asr_agreement` — float, required

Mean pairwise character-level overlap across `asr_candidates`, rounded to 3 decimals.
Range 0.000–1.000. 1.0 = all models produced identical text.

**Use**: prioritises human calibration (low agreement reviewed first) and is the ASR
filter signal (`docs/QUALITY_SPEC.md §4`). Not a final gate — `text_verified` is.

---

### `jyutping` — string, required

Jyutping romanisation with tone numbers. Space-separated, one token per syllable.

**Format**: `sam1 zong6 beng6 zung1 fung1`

**Constraints**:
- Each token must match `^[a-z]+[1-6]$`
- English words pass through unchanged (canto-hk-g2p does not tokenise or bracket them
  letter-by-letter — see `CLAUDE.md` "Issue 1 — G2P tool history")
- No empty tokens
- Reject the segment if more than 5% of tokens fail the regex (Hard Constraint 8)

**Validation** (must pass before writing to manifest):
```python
import re
JYUTPING_TOKEN = re.compile(r'^[a-z]+[1-6]$')

def validate_jyutping_field(jyutping: str) -> tuple[bool, float]:
    tokens = jyutping.strip().split()
    valid = sum(1 for t in tokens if JYUTPING_TOKEN.match(t))
    frac = valid / len(tokens) if tokens else 0.0
    return frac >= 0.95, frac
```

**Invalid examples**:
```
"sam1zong6beng6"         # ← missing spaces (pycantonese bug — do not use pycantonese)
"sam1 zong6beng6"        # ← partial concatenation
"sam 1 zong 6"           # ← tone digit separated from syllable
"sam1 zong6 "            # ← trailing space (minor, clean before writing)
```

---

### `duration_sec` — float, required

Duration of the audio segment in seconds, rounded to 3 decimal places.

**Valid range**: `3.000 ≤ duration_sec ≤ 20.000`

**Source**: Read from WAV file header using `soundfile.info(path).duration`.

---

### `sample_rate` — integer, required

Audio sample rate in Hz.

**Expected value**: `48000` for all segments in this corpus (48 kHz mono master).

This is a hard gate. A segment below 48 kHz cannot enter the corpus — downsampling is
irreversible and modern TTS codecs need ≥24 kHz (`KNOWN_ISSUES.md §11`). If any segment
has a lower rate, the pipeline has a bug (it downsampled the master instead of using a
transient 16 kHz copy for the tools).

---

### `speaker_id` — string, required

Estimated speaker identity from clustering.

**Format**: `{source}_{cluster_id:03d}` — e.g., `rthk_001`, `youtube_045`, `podcast_012`

**Note**: Speaker IDs are per-source (cluster IDs are not globally deduplicated across sources). A future merge step may assign global IDs, but that requires cross-source speaker matching — document in DECISIONS.md if implemented.

**Fallback**: If speaker clustering fails, use `{source}_unk` for all segments from that source. This is not ideal but does not corrupt training.

---

### `gender` — string enum

| Value | Meaning |
|-------|---------|
| `male` | Male voice |
| `female` | Female voice |
| `unknown` | Could not determine |

**Source**: May be inferred from speaker clustering, program metadata, or left as `unknown`.

---

### `style` — string enum

Speech style that describes the delivery manner.

| Value | Description |
|-------|-------------|
| `formal` | Scripted broadcast speech, news anchor delivery |
| `casual` | Conversational, unscripted, natural speech |
| `narration` | Documentary narration style (similar to formal but with slower pace) |
| `interview` | Q&A format, mixed formal/casual |

**Source**: Set from the `style` field in the source YAML configuration (e.g., RTHK documentaries → `narration`). May be overridden per-segment in future if prosody analysis is implemented.

---

### `snr_db` — float, required

Signal-to-noise ratio estimate in decibels, rounded to 1 decimal place.

**Valid range**: 0.0 – 80.0 (practical range for speech recordings)

**Minimum for inclusion**: ≥ 25.0 dB

---

### `dnsmos` — float, required

DNSMOS P.835 overall quality score (OVRL field), rounded to 2 decimal places.

**Valid range**: 1.00 – 5.00

**Minimum for inclusion**: ≥ 3.0

---

### `english_ratio` — float, required

Fraction of alphabetic characters that are English (ASCII a-z), rounded to 3 decimal places.

**Valid range**: 0.000 – 1.000

**Maximum for inclusion**: ≤ 0.30

---

### `created_at` — string, required

ISO date (YYYY-MM-DD) when this manifest entry was created.

**Format**: `str(date.today())` — e.g., `"2026-06-09"`

---

## Complete Validation Script

Run this after `manifest.build`/`.export` and before `report.build`:

```python
#!/usr/bin/env python3
"""Validate manifest.jsonl against schema."""
import json, re, sys
from pathlib import Path

REQUIRED_FIELDS = [
    "id","audio_path","source","source_url","program","domain",
    "text","text_verified","asr_candidates","asr_agreement","jyutping",
    "duration_sec","sample_rate","speaker_id",
    "gender","style","snr_db","dnsmos","english_ratio","created_at"
]
SOURCE_ENUM  = {"rthk","youtube","podcast","hktv"}
DOMAIN_ENUM  = {"documentary","news","talk_show","podcast","drama","vlog","educational","other"}
GENDER_ENUM  = {"male","female","unknown"}
STYLE_ENUM   = {"formal","casual","narration","interview"}
JP_TOKEN     = re.compile(r'^[a-z]+[1-6]$')
SHARD_ROOTS  = ("/mnt/Drive2/", "/mnt/Drive3/", "/mnt/Drive4/")

errors = 0
with open("metadata/manifest.jsonl") as f:
    for i, line in enumerate(f, 1):
        e = json.loads(line.strip())

        # Required fields
        for field in REQUIRED_FIELDS:
            if field not in e:
                print(f"Line {i}: Missing field: {field}")
                errors += 1

        # audio_path (3-way segment shard — see config/storage_layout.py:shard_index())
        if not e.get("audio_path","").startswith(SHARD_ROOTS):
            print(f"Line {i}: Bad audio_path: {e.get('audio_path')}")
            errors += 1

        # enums
        for field, enum in [("source",SOURCE_ENUM),("domain",DOMAIN_ENUM),
                             ("gender",GENDER_ENUM),("style",STYLE_ENUM)]:
            if e.get(field) not in enum:
                print(f"Line {i}: Invalid {field}: {e.get(field)}")
                errors += 1

        # duration
        dur = e.get("duration_sec", 0)
        if not (3.0 <= dur <= 20.0):
            print(f"Line {i}: duration_sec out of range: {dur}")
            errors += 1

        # sample rate (hard gate: 48 kHz master)
        if e.get("sample_rate") != 48000:
            print(f"Line {i}: sample_rate not 48000: {e.get('sample_rate')}")
            errors += 1

        # verified text (final corpus must be 100% verified)
        if e.get("text_verified") is not True:
            print(f"Line {i}: text_verified not true")
            errors += 1

        # at least 2 ASR candidates
        if len(e.get("asr_candidates", [])) < 2:
            print(f"Line {i}: fewer than 2 asr_candidates")
            errors += 1

        # dnsmos
        d = e.get("dnsmos", 0)
        if not (1.0 <= d <= 5.0):
            print(f"Line {i}: dnsmos out of range: {d}")
            errors += 1

        # jyutping (Hard Constraint 8: reject if > 5% of tokens fail)
        jp = e.get("jyutping", "")
        tokens = jp.strip().split()
        if tokens:
            valid = sum(1 for t in tokens if JP_TOKEN.match(t))
            if valid / len(tokens) < 0.95:
                print(f"Line {i}: Low jyutping validity: {jp!r}")
                errors += 1

print(f"\nValidation done: {i} entries, {errors} errors")
sys.exit(0 if errors == 0 else 1)
```

---

## JSONL Format Rules

- One JSON object per line. No trailing commas.
- UTF-8 encoding. BOM not allowed.
- No null values — use `""` for empty strings, `0.0` for missing numeric scores.
- All float fields use `round(value, n_decimals)` — no scientific notation.
- File must be sortable by `id` (useful for reproducible train/val splits).

---

## Split Files

`metadata/train.jsonl` and `metadata/val.jsonl` are subsets of `metadata/manifest.jsonl`. They:
- Are written by the `manifest.build`/`.export` DAG node (95/5 split)
- Use the exact same schema
- Satisfy: no `speaker_id` appears in both train and val
- Are stratified by `source` to preserve source distribution
