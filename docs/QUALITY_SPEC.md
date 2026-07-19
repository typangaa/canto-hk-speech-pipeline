# Quality Specification — canto-hk-speech-pipeline

> Threshold values, rationale, and configuration for all quality filters.
> These values were calibrated against the RTHK pilot dataset (49,195 segments examined).
> Adjust only with clear evidence; document any change in DECISIONS.md.

---

## Hard Gates (not scored — pass/fail prerequisites)

These are not threshold filters; a segment that fails any is structurally unusable.

| Gate | Requirement | Why |
|------|-------------|-----|
| Sample rate | 48 kHz mono master | < 24 kHz is unusable for any modern TTS codec; never store below 48 kHz (`KNOWN_ISSUES.md §11`) |
| Single speaker | diarization reports exactly 1 speaker, no overlap | Multi-speaker clips corrupt training + speaker labels (`KNOWN_ISSUES.md §10`) |
| Verified text | `text_verified: true` before entering corpus | No segment trained on raw single-ASR output (`KNOWN_ISSUES.md §9`) |

---

## Primary Filters

### 1. Duration Filter

| Parameter | Value | Notes |
|-----------|-------|-------|
| Minimum | 3.0 seconds | Below this, insufficient phoneme context for TTS |
| Maximum | 20.0 seconds | Above this, GPU VRAM insufficient for end-to-end attention at training |

**Rationale**: TTS attention mechanisms have quadratic complexity over sequence length. At 24 kHz with 50 tokens/second codec, 20s = 1,000 tokens per sample — a practical limit at batch_size=1 with 24 GB VRAM. The lower bound of 3s was chosen because shorter segments rarely contain a complete syntactic unit in Cantonese.

**How it is implemented**: Not by hard-cutting audio, but by VAD-based segmentation that targets this window. If VAD produces a segment outside this window, it is filtered here.

**Expected rejection rate**: ~5–10% (mostly very short clips from music segments or sign-off phrases).

---

### 2. DNSMOS P.835 Score

| Parameter | Value | Notes |
|-----------|-------|-------|
| Minimum score | ≥ 3.0 | On 1–5 scale |
| Valid range | [1.0, 5.0] | ASSERT this before applying filter |
| Expected median | 3.2–3.8 | For RTHK professional audio |

**Rationale**: DNSMOS P.835 (overall quality score OVRL) correlates well with perceived TTS training suitability. Scores below 3.0 typically indicate background noise, reverb, or codec artefacts that confuse the acoustic decoder. The old pilot used an incorrectly configured DNSMOS and accepted many 1.0–2.5 rated files, leading to noisy training data.

**Sanity check** (REQUIRED before running at scale):
```python
scores = [compute_dnsmos(f) for f in first_100_files]
assert all(1.0 <= s <= 5.0 for s in scores), "DNSMOS out of valid range — fix model setup"
assert 2.0 <= sorted(scores)[50] <= 4.5, "DNSMOS median suspicious"
```

**Expected rejection rate**: 15–25% (varies by source — podcast audio may be worse than RTHK broadcast).

---

### 3. SNR (Signal-to-Noise Ratio)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Minimum | ≥ 25 dB | Professional broadcast standard |
| Expected median | 30–40 dB | RTHK studio quality |

**Rationale**: SNR < 20 dB produces audible noise that the codec learns to reproduce. SNR < 25 dB causes the spectral codec to mis-encode phonemes in noisy frequency bands. Professional broadcast content (RTHK) typically achieves 35–45 dB; YouTube content varies widely.

**SNR estimation method** (from `docs/archive/PIPELINE_SPEC.md Stage 5`): Frame-energy-based estimate separating top 10% (signal) from bottom 10% (noise floor). This is an approximation — use `dnsmos` as the primary quality gate.

**Expected rejection rate**: 10–20% (higher for YouTube user-generated content).

---

### 4. ASR Agreement (Multi-Model)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Minimum agreement | ≥ 0.80 | Mean pairwise character-overlap across ASR candidates |
| Per-model confidence | advisory | Stored, not a hard gate (single-model confidence is unreliable for `yue`) |

**Rationale**: This project runs **multiple** ASR models per segment and a human
calibrates the canonical text (see `CLAUDE.md` ASR strategy and `KNOWN_ISSUES.md §9`).
Single-model confidence is a poor gate for Cantonese — base Whisper is overconfident on
Mandarin-form output and unreliable on `yue`. Cross-model **agreement** is a far better
signal: when independent models converge, the transcript is likely correct; when they
diverge, the segment needs careful human attention.

How agreement drives the pipeline:
- **Filtering**: segments with agreement < 0.80 are not auto-rejected on text grounds,
  but are deprioritised — they reach the human last and must be human-verified before
  use. (Audio of a low-agreement segment can still be perfectly good; only the text is
  uncertain.)
- **Calibration ordering**: `calibrate.sample`/`pipe calibrate serve` reviews lowest-agreement segments first.

**Final gate**: regardless of agreement, every segment that enters the training corpus
must have `text_verified: true`. Agreement only prioritises human effort.

**Expected distribution**: clean broadcast speech → agreement 0.90+; casual/overlapping
or noisy speech → lower.

---

### 5. Language Composition Filters

#### English Ratio

| Parameter | Value |
|-----------|-------|
| Maximum English ratio | ≤ 0.30 |
| Calculation | English alphabetic chars / all alphabetic chars |

**Rationale**: Cantonese speakers code-switch English naturally (e.g., "call 佢" or "email 返 你"). A ratio up to 30% is acceptable because the TTS should learn to handle this. Above 30%, the segment is likely from an English-dominant context unsuitable for Cantonese TTS.

**Note on English tokens in Jyutping**: canto-hk-g2p passes English words through unchanged rather than bracket-placeholder tokenising them (see `CLAUDE.md` "Issue 1 — G2P tool history"). The TTS training procedure must handle raw English tokens mixed into the Jyutping string.

#### Mandarin Ratio

| Parameter | Value |
|-----------|-------|
| Maximum Mandarin ratio | ≤ 0.15 |
| Detection method | Fraction of Mandarin-only characters not in Cantonese vocabulary |

**Rationale**: Mandarin-transcribed audio causes G2P failures because Mandarin characters map to Mandarin pronunciations, not Jyutping. A small fraction is acceptable (some Cantonese-written text uses Mandarin characters by convention).

**Detection approach**:
```python
# Characters that indicate Mandarin transcription (not used in written Cantonese)
MANDARIN_ONLY = set("是没有他她它们这那说")   # non-exhaustive — extend as needed
CANTONESE_PREFERRED = set("係冇佢佢哋呢嗰嚟咁嘅")

def mandarin_ratio(text: str) -> float:
    mandarin_chars = sum(1 for c in text if c in MANDARIN_ONLY)
    cantonese_chars = sum(1 for c in text if c in CANTONESE_PREFERRED)
    cjk_total = sum(1 for c in text if '一' <= c <= '鿿') or 1
    # Simple heuristic: if Mandarin markers exist without Cantonese markers, flag
    if cantonese_chars == 0 and mandarin_chars > 3:
        return mandarin_chars / cjk_total
    return mandarin_chars / max(cjk_total, mandarin_chars + cantonese_chars)
```

**Expected rejection rate**: 5–10% (depends on ASR Mandarin-vs-Cantonese confusion behaviour — see `KNOWN_ISSUES.md §9`).

---

### 6. Text Length Filters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Minimum characters | ≥ 5 CJK chars | After punctuation removal |
| Maximum characters | ≤ 150 chars | |

**Rationale**:
- Min 5 chars: Shorter segments tend to be utterances like "係" or "好" — too short to be useful TTS training data.
- Max 150 chars: ASR hallucination (models sometimes generate repetitive text for music/noise segments) produces abnormally long output. This filter catches those.

---

### 7. Jyutping Validity

Hard Constraint 8 (`CLAUDE.md`): reject if `valid_fraction < 0.95` (i.e., > 5% of a
segment's Jyutping tokens fail the `^[a-z]+[1-6]$` pattern). This is a hard gate, not
an advisory/scored filter — moved out of the old "Secondary Filters" grouping below,
which is advisory-only.

| Threshold | Action |
|-----------|--------|
| ≥ 0.95 | Accept |
| < 0.95 | Reject |

---

## Secondary Filters (Advisory — Do Not Auto-Reject)

These filters generate warnings that are logged for human review but do not automatically reject segments.

### Speaker Confidence
If speaker embedding distance to nearest cluster centroid > 0.85 cosine similarity, flag as "uncertain speaker assignment". Does not reject, but lowers confidence in speaker_id.

### Duration Outlier
Segments with duration > 18 seconds are accepted but flagged. They are rare under VAD-based segmentation; a large number suggests the VAD silence threshold is too permissive.

---

## Filter Application Order

Apply filters in this order (cheapest first to minimise computation):

```
0. Hard gates: sample rate == 48 kHz, single speaker, text_verified
1. Duration (O(1) from file header)
2. Text length (O(n) on verified text)
3. English ratio (O(n) on verified text)
4. Mandarin ratio (O(n) on verified text)
5. ASR agreement (already computed by the `asr.agreement` node)
6. Jyutping validity (already computed by the `g2p` node, runs on verified text)
7. SNR (O(n) audio analysis — fast)
8. DNSMOS (speechmos neural model on 16 kHz copy — most expensive, apply last)
```

---

## Per-Source Expected Pass Rates

These are calibrated estimates. If actual pass rates deviate significantly, investigate immediately.

| Source | Expected pass rate | Common failure mode |
|--------|-------------------|---------------------|
| RTHK broadcast | 70–80% | False: audio confident; failures mainly duration/Mandarin |
| YouTube HK channels | 50–65% | Music, noisy backgrounds, intro/outro |
| HK podcasts | 55–70% | Inconsistent mic quality |
| HK drama | 40–60% | Background music under dialogue |

---

## Changing Thresholds

If you need to change a threshold, follow this process:
1. Run `python -m pipeline.cli run report.build` with the current threshold and note the pass rate.
2. Propose the change with a specific reason in DECISIONS.md.
3. Run `filter.text`/`filter.acoustic` with the new threshold on a `--limit` sample to preview impact.
4. If > 20% swing in pass rate, require human review before applying.
5. Update this document to reflect the new value and rationale.

---

## Acceptance Thresholds for Final Corpus

The corpus is ready for TTS training when all pass:

| Metric | Hard minimum | Target |
|--------|-------------|--------|
| Clean hours total | 100 h | 300 h |
| Unique speakers | 100 | 200+ |
| Sample rate | 48 kHz (100%) | 48 kHz (100%) |
| `text_verified` coverage | 100% | 100% |
| Single-speaker segments | 100% | 100% |
| DNSMOS p50 | ≥ 3.0 | ≥ 3.2 |
| SNR p50 | ≥ 25 dB | ≥ 30 dB |
| Jyutping valid rate | ≥ 98% | ≥ 99% |
| Sources | ≥ 3 | ≥ 4 |
| Domains | ≥ 3 | ≥ 5 |
| Windows paths | 0 | 0 |
