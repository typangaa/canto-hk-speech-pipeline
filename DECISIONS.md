# Decision Log — canto-hk-speech-pipeline

## 2026-06-09 — Project Scope and Data Sources
**Decision**: Build self-owned HK Cantonese dataset from RTHK + YouTube + Podcast + other HK TV. Target 100–500h, 100+ speakers.
**Alternatives considered**: WenetSpeech-Yue (rejected — user wants self-owned data); single-source RTHK only (rejected — too narrow domain and speaker diversity).
**Rationale**: Self-owned data gives full control over quality, licensing, and metadata. Multiple sources ensure domain and speaker diversity.

---

<!-- Subsequent agent sessions append decisions below this line -->

## 2026-06-09 — Audio storage at 48 kHz mono master
**Decision**: Store every segment as a 48 kHz mono WAV master. Generate transient 16 kHz copies only for VAD/diarization/ASR/DNSMOS, never overwriting the master.
**Alternatives considered**: 16 kHz (rejected — telephone-band, unusable for any modern TTS codec; the prior RTHK pipeline's 16 kHz choice was an ASR habit and is a dead-end); 24 kHz (viable for NeuCodec/F5 but caps out MOSS-Nano which is 48 kHz).
**Rationale**: Downsampling is irreversible; 48 kHz keeps every candidate TTS architecture open (NeuCodec 24k, F5-TTS 24k, MOSS-Nano 48k). Storage cost (~3× vs 16 kHz) accepted by user. See KNOWN_ISSUES.md §11.

## 2026-06-09 — Multi-ASR + human calibration
**Decision**: Run several ASR models per segment (Cantonese fine-tuned Whisper + base large-v3 with `language="zh"` + written-Cantonese prompt). Store all candidates + a cross-model agreement score. A human produces the canonical `text` in a dedicated calibration stage (05). G2P and the final manifest consume only verified text.
**Alternatives considered**: single `language="yue"` pass (rejected — causes decoder collapse on large-v3, confirmed Jan 2026); single fine-tuned model auto-trusted (rejected — user explicitly wants multiple references + manual calibration for quality).
**Rationale**: User requirement. Cross-model agreement is a far better quality signal than single-model confidence for Cantonese, and human calibration sets the ground-truth bar. See KNOWN_ISSUES.md §9.

## 2026-06-09 — Speaker diarization before segmentation
**Decision**: Run pyannote (or WhisperX) diarization first; cut VAD clips only within single-speaker turns. Reject overlapping-speech regions. Per-file diarization labels are clustered into global speaker_ids in stage 8.
**Alternatives considered**: VAD-only segmentation (rejected — multi-speaker programs like 城市論壇/鏗鏘集 would produce clips spanning speaker changes, corrupting training and speaker labels).
**Rationale**: TTS training clips must be single-speaker. See KNOWN_ISSUES.md §10.

## 2026-06-09 — DNSMOS via speechmos
**Decision**: Compute DNSMOS with `speechmos` (`dnsmos.run(audio16k, sr=16000)["ovrl_mos"]`), on a 16 kHz copy.
**Alternatives considered**: a fabricated `dnsmos` package (the earlier draft's `from dnsmos import DNSMOS` does not exist); torchmetrics `DeepNoiseSuppressionMeanOpinionScore` (acceptable alternative).
**Rationale**: `speechmos` is the real, verified package. DNSMOS models expect 16 kHz input. See KNOWN_ISSUES.md §4.

## 2026-06-09 — Licensing / usage scope
**Decision**: Treat the corpus as internal research / model-training only; do not redistribute raw source audio. Record source_url for provenance. Prefer public-broadcaster (RTHK) content.
**Rationale**: "Self-owned" means self-collected, not licensed for redistribution. See KNOWN_ISSUES.md §12.

## 2026-06-09 — ASR Model A: simonl0909/whisper-large-v2-cantonese (local ct2)
**Decision**: Use `simonl0909/whisper-large-v2-cantonese` (converted to ctranslate2 format at `data/ct2_models/whisper-large-v2-cantonese`) as the primary Cantonese ASR model. Use `Systran/faster-whisper-large-v3` (cached) as the secondary model with `language="zh"` + Cantonese written-form prompt.
**Alternatives considered**: `khleeloo/whisper-large-v3-cantonese` (not tested — simonl0909 already available); `openai/whisper-large-v3` via HuggingFace (rejected — HF cache has Transformers format, not ctranslate2; use Systran mirror instead).
**Rationale**: simonl0909 model reliably produces authentic HK Cantonese orthography (係、唔係、噉、㗎、喺、佢哋) with no prompting. The large-v3 tends toward formal Mandarin Chinese orthography even with the Cantonese prompt. Agreement scores are often lower (0.6-0.8) due to writing system differences, not transcription errors — human calibration in stage 5 will select the canonical Cantonese form. This is the expected design.

## 2026-06-09 — Sequential model loading (two-pass transcription)
**Decision**: Load ASR models one at a time rather than simultaneously. Pass 1: Cantonese model transcribes all segments and stores results in memory. Pass 2: large-v3 transcribes all segments. Then write all `.transcript.json` files at once.
**Alternatives considered**: Load both simultaneously (rejected — 2 × float16 models ~6-7GB exceeds GPU 1's available 5.4GB); per-segment model switching (rejected — model loading overhead per segment too slow ~15s × 1906).
**Rationale**: GPU 0 occupied by llama-server (root process, ~23GB). GPU 1 has 5.4GB free. int8_float16 quantization allows each model to fit (~2GB each). Two-pass gives full throughput (~3.5 segs/sec).

## 2026-06-09 — DNSMOS filter metric: sig_mos not ovrl_mos
**Decision**: Use `sig_mos` (speech clarity MOS, typical range 3.0–5.0) rather than `ovrl_mos` (overall MOS) as the DNSMOS quality gate. Threshold remains ≥3.0.
**Alternatives considered**: `ovrl_mos ≥ 3.0` (original plan — only 18% of RTHK segments pass because documentary/broadcast audio has background music and ambient sound); `p808_mos` (possible but less commonly used for filtering).
**Rationale**: RTHK 鏗鏘集 documentary has consistent background music and ambient audio, which DNSMOS penalizes heavily in `bak_mos` (background score ~2.3–2.5). But `sig_mos` (speech clarity) is 3.4–3.5, indicating the speech itself is clear. For TTS training, speech clarity is what matters; background can be separated if needed. Using `ovrl_mos ≥ 3.0` would reject ~82% of segments from high-quality broadcast audio, which is counter-productive. Note: `dnsmos_ovrl` is still stored in filter.json for reference.

## 2026-06-29 — Dataset will NOT be released (zero-risk); focus shifts to quality
**Decision**: The corpus will **not** be open-sourced or shared in any form. This supersedes/strengthens the 2026-06-09 "internal research only" decision into an explicit no-publish policy: never release the dataset, manifests, raw/filtered audio, per-segment `source_url`s, or the reconstruction recipe. A canto-tts **model** trained on the data *may* be published later (weights only, never the data). The pipeline **code** stays open source (Apache 2.0). Effort now focuses entirely on improving dataset quality for TTS training.
**Alternatives considered**:
  - Metadata-only "reconstruction recipe" HF release (built + validated as A1–A3): rejected. A3 proved real friction — YouTube needs user cookies + a JS runtime (anti-bot), and ~30% of podcast segments are unreconstructable due to dynamic ad insertion (same timestamp → different audio after re-download). Reconstruction also forces exposing every copyrighted `source_url`.
  - Ship audio directly like VieNeu-TTS (gated + CC-BY-NC, no provenance disclosed): rejected — still carries copyright exposure; owner wants zero risk.
**Rationale**: Owner prefers no risk at all. Private use for model training has the lowest exposure. The release infrastructure (`reconstruct.py`, `scripts/10_enrich_manifest.py`, `metadata/manifest_release.jsonl`, `metadata/excluded_no_url.jsonl`) is **kept dormant** rather than deleted — the no-release decision is not necessarily permanent — but must never be acted on without explicit owner approval. See CLAUDE.md Hard Constraint #9.
**Quality priorities (all selected, TTS-focused)**: (1) music/jingle + overlap detection → tag clean-speech segments; (2) speaker cluster purity audit + per-speaker count/hours stats; (3) loudness normalisation (−23 LUFS) + leading/trailing-silence boundary trim; (4) duration filtering (2–15 s TTS subset) + silver-tier transcript accuracy (WER) estimate.
