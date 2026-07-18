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

## 2026-07-04 — Segments master format: FLAC (lossless), not WAV or Opus
**Decision**: New segments going forward are written as **FLAC** (lossless), not 16-bit PCM WAV. Hard Constraint #6 is reworded from "48 kHz mono master, never lower" to "48 kHz mono **lossless** master, never lower, never lossy" — WAV and FLAC are both acceptable containers, Opus/MP3/any lossy codec is never acceptable as a segment master. Existing 843G of legacy WAV segments are **not** re-encoded (no re-transcoding of an existing master, per the §3 non-destructive discipline already established for raw audio) — the decode layer already reads FLAC natively via `soundfile`, so mixed WAV+FLAC masters are transparent to every downstream consumer (filter, G2P, manifest export, canto-tts training).
**Alternatives considered**:
  - Stay on WAV (rejected): under the current "keep Stage-6-rejected candidate clips" retention policy (see [[canto-corpus-rearchitecture]] memory, 2026-07-04 capacity investigation), every new raw file segmented continues to produce ~2.46× more physical bytes than its catalog-tracked hours (Stage 3 writes every candidate clip; Stage 6 only promotes the QC-passing subset). Projected forward against the ~4.3 TiB of free space expected after the P5 raw→opus transcode, WAV realistically yields only ~5,300 new catalog-hours (~6,300h total with the existing 1,004.5h) — clears the 5× scale target but **not** the 10× target (10,000h).
  - Opus (lossy) as segment master (rejected): would trivially clear any realistic scale target (~31,800 new catalog-hours even after the 2.46× overhead), but a lossy-compressed **master** risks vocoder/codec training degradation (see `docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md` §13.1 — EURASIP MP3-vocoder findings, Valin SSW 2019 LPCNet+low-bitrate-opus, arXiv:2111.02380) and would permanently foreclose ever re-deriving a truly lossless master. Opus remains the right choice for **raw** (P5), which is a re-downloadable/re-segmentable intermediate, not a training master.
**Rationale**: FLAC yields ~9,650 new catalog-hours even after applying the 2.46× reject-clip overhead (~10,650h total) — the only format among the three that reliably clears the 10× (10,000h) scale target under the current retention policy, while remaining fully lossless. This closes `docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md` §10 Q1 ("10× segments format leans FLAC, confirm with P6 projection data before finalizing") — the 2026-07-04 capacity investigation numbers serve as that confirming projection. Owner sign-off given 2026-07-04. See CLAUDE.md Hard Constraint #6 (reworded) and `docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md` §10 Q1.

## 2026-07-04 — Raw backlog format: FLAC confirmed by owner; new-download policy clarified
**Decision (owner confirmed)**: the existing ~1.6T raw WAV backlog transcodes to **FLAC**, not Opus — reversing §7.1's original opus choice for this one item. Owner's rationale: consistency with the segments FLAC decision and zero additional generation loss outweighs the ~420GB extra space (FLAC ~570GiB vs opus's estimated ~150GiB, both measured/estimated 2026-07-04).
**New-download policy — clarified, NOT changed**: the owner then asked whether *future* downloads should also be converted to FLAC "to compress." Tested empirically (ffmpeg, real files): re-encoding the native source codec to FLAC makes files **2.1-3.5× BIGGER**, not smaller (a YouTube opus sample: 33.4MB native → 70.7MB FLAC; an RTHK AAC sample: 5.2MB native → 18.1MB FLAC) — lossy codecs discard perceptually-irrelevant information to achieve compression ratios no lossless format can match, so FLAC-encoding an already-lossy decode only bloats it with zero fidelity gain. **Conclusion: future downloads must NOT be converted to FLAC** — §7.1's original policy point 1 (keep the native bestaudio container — opus for YouTube, AAC/m4a for RTHK/podcast — skip the WAV round-trip entirely) remains correct and is reaffirmed, unchanged.
**Implementation gap found (2026-07-04)**: that native-container policy was approved 2026-07-02 but was **never actually implemented** — `sources/youtube_channels.yaml` line 2771 still hardcodes `audio_format: "wav"`, and no `ingest.download` DAG node exists yet (only `ingest_probe.py`, which reads metadata, not download-and-store). New downloads have continued converting to bloated 48kHz WAV every day since the policy was approved. This is an independent, low-risk fix (not gated on P5) — flip the yt-dlp postprocessor config and build `ingest.download` per the DAG node table (§6) to store the native container + record its codec, rather than force-converting to WAV.
**Scope note**: the FLAC-vs-opus question only ever applied to the existing WAV backlog, never to future downloads (§7.1 already routed future downloads around this choice via the native-container policy) — this was confirmed, not new information, but worth restating since it resolves the "is this pipeline only handling historical data" concern raised when this was reopened.

## 2026-07-04 — Raw master format: FLAC vs Opus reopened, then RESOLVED (see entry above)
**Status**: RESOLVED — owner confirmed FLAC for the existing backlog (see "Raw backlog format" entry above, which also covers the new-download-policy follow-up question). Kept below for the full tradeoff analysis that led to that confirmation. Originally this entry asserted FLAC as settled without checking prior context first — recorded here as-is as the analysis trail. This entry originally asserted FLAC as settled; on closer reading of `docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md` §7.1 ("Raw → opus — owner 已拍板壓縮路線"), the opus choice was **already an explicit, deliberated 2026-07-02 owner decision** (see that plan's header: "已拍板決定 ... 4. Raw 容量策略 = 壓縮保留(opus;唔行 transient-delete)") that already weighed the exact "second lossy generation" risk raised below (§7.1's own "誠實 caveat" paragraph) and accepted it explicitly, while separately solving the future-ingest problem (new downloads keep native bestaudio container, never round-trip through WAV at all — so this tradeoff never applies to anything downloaded from 2026-07-02 onward, regardless of which way this reopened question resolves). This correction should not have been declared unilaterally without weighing §7.1's existing reasoning; it is presented here as a reopened question with new measured data, for the owner to re-confirm or reaffirm the original opus call.

**The tradeoff, with both existing (§7.1) and new (this session) reasoning**: `raw_files` currently has 5,065.1 raw-hours; no deletion of raw after segmentation has ever been implemented (grepped `pipeline/` — no `unlink`/`rmtree` tied to raw exists), so despite `config/storage_layout.yaml`'s "raw policy: TRANSIENT" comment, raw is in practice retained indefinitely — for the same reason rejected segment candidate clips are kept (§ above): to allow future re-derivation (better diarization/VAD models, revised clipping policy) without re-downloading. Any audio kept for future re-derivation is functionally a master and must follow Hard Constraint #6 (lossless, never lossy) — the same logic that put segments on FLAC applies to raw.
**New evidence** (measured via `ffprobe` on real files, not assumed): YouTube-sourced raw's original codec is already **Opus** (48kHz stereo — YouTube's own delivery format), RTHK-sourced raw's original codec is **AAC 32kHz/64kbps** (even lower quality). Our "raw" WAV is therefore *never* a first-generation lossless master — it is already a decode of lossy source audio. Re-compressing it to Opus again would stack a **second lossy generation** on top (lossy→PCM→lossy is not bit-identical; artifacts compound) — a form of generation loss the project already flagged as a risk for the 246 never-segmented raw files, which turns out to generalize to *all* retained raw, not just those 246. FLAC avoids this entirely: it freezes the current (already source-lossy, but stable) bit content with zero additional loss.
**Alternatives considered**: Opus 128k for raw (original 2026-07-02 plan, rejected 2026-07-04): assumed raw was truly transient and safe to degrade further once segmented; both assumptions were wrong (no deletion code exists; source is already lossy so a second lossy pass compounds loss, it doesn't merely "match" existing quality).
**Capacity re-check** (measured, not estimated — 5 real raw files and 15 real segment clips FLAC-encoded and compared byte-for-byte): actual FLAC ratio ≈ **35% of WAV for raw**, **34% of WAV for segments** (both far better than the generic "~55% for speech" textbook estimate used in the earlier capacity table). Raw→FLAC frees ~1.03 TiB on Drive2 (vs ~1.3-1.5TiB assumed for opus — comparable). Total free for new segments ≈ 3.93 TiB → FLAC segments at the real measured density (117.2 MB/h) yield ~14,300 new catalog-hours even after the 2.46× reject-clip overhead, ~15,300h total — clears the 10× (10,000h) scale target with ~53% margin, a *better* margin than the original opus-raw-assumption estimate (~10,650h), because real segment FLAC compression outperforms the generic estimate.
**Rationale**: Consistency with the just-adopted lossless-master principle (Hard Constraint #6 applies uniformly to any retained audio, not just segments); avoids compounding a second lossy generation onto already-lossy YouTube/RTHK source audio; and the measured (not estimated) capacity numbers show no real cost to this choice — FLAC-for-raw still comfortably clears the 10× target. Supersedes the raw→opus half of the 2026-07-04 capacity investigation entry above.

## 2026-07-04 — Storage format policy FINALIZED after external research (owner re-confirmed all three tiers)
**Decision (owner confirmed all three, same day, after an online research pass via agy-gemini)**:
  1. **Segments tier = FLAC 48kHz mono** (re-confirmed). Research validation: every TTS-oriented public corpus uses lossless (Emilia → 24kHz WAV via Emilia-Pipe; LibriTTS/LibriTTS-R → 24kHz WAV); the corpora that ship lossy Opus 32kbps (GigaSpeech, WenetSpeech) are ASR corpora accepting the tradeoff for distribution size. TTS training is the single most codec-artifact-sensitive downstream — models learn compression artifacts as speaker/channel traits and reproduce them (Emilia paper's explicit motivation for lossless standardization + DNSMOS filtering, arXiv:2407.05361).
  2. **Existing raw WAV backlog (~1.6T) = FLAC** (re-confirmed). The WAVs are our only remaining copy (99.7%+ of original native containers already deleted after the old WAV-conversion step), so WAV→FLAC is a pure lossless archival move (bit-exact, PCM-compare verifiable, measured ratio ~35.2% → ~570GiB). Opus 48k would save a further ~420GB but adds a lossy generation to the only copy — now explicitly contradicted by archival best practice (see 3).
  3. **Future downloads = keep the native container untouched, ZERO transcode** (NEW — supersedes and closes the per-source "harmonize to mono opus 48kbps" direction explored earlier the same day, including the interim "harmonize podcast/RTHK only" answer and the later "harmonize all three" proposal). This is the archival best practice: preserve the original stream bit-perfect (`ffmpeg -c:a copy` if container unification is ever needed).
**Research findings that drove point 3** (two agy-gemini research reports, 2026-07-04):
  - Lossy→lossy re-encoding for archival is considered bad practice outright (generation loss compounds; the second encoder treats first-pass artifacts as signal).
  - Stereo→mono downmix *before* a lossy encode risks phase cancellation / comb filtering that permanently damages the 0–8kHz speech band — this survives any later 16kHz downsampling, so "the tools downsample anyway" does not neutralize it.
  - DNSMOS carries a measurable negative bias against lossy-compressed audio even when perceptually transparent to humans (detects HF roll-off/quantization noise in its latent space) — segments cut from harmonized raw would suffer artificially depressed Stage-6 yield at the DNSMOS ≥3.0 gate.
  - Robustness data for the tools themselves at opus 48kbps mono: Silero VAD / pyannote DER / Whisper WER ≈ unaffected (<1% rel.); ECAPA-TDNN EER +0.4–0.7% (Thakur, Yip & Chng 2025); i.e. the harmonize option was *defensible* on tool-robustness grounds — it fails on TTS-training sensitivity, DNSMOS bias, phase risk, and archival principle, not on VAD/ASR robustness.
  - Dataset engineering guides (SpeechBrain/Kaldi/Emilia-Pipe practice) explicitly say: run segmentation/diarization/VAD on the original, highest-quality source **before** any re-compression (compression-induced temporal smearing shifts boundaries). This also answers the owner's earlier "why not harmonize then segment?" question with an external, citable norm — though with point 3 (no transcode at all) the sequencing question is now moot.
**Cost accepted**: native containers ≈ ~62GB/1000h raw at current source mix (podcast 44% @192kbps MP3, YouTube 42% @~106kbps opus stereo, RTHK 14% @64kbps AAC) vs ~22GB/1000h if harmonized — i.e. ~40GB extra per 1000 raw-hours. Accepted because the binding storage constraint is the segments tier (2.46× reject-clip overhead), not raw; even +10,000 raw-hours costs only ~400GB extra.
**Alternatives considered**: (a) harmonize all three sources to mono opus 48kbps after segmentation (max savings, rejected — violates archival practice, poisons any future re-segmentation, adds segment-state tracking complexity); (b) hybrid harmonize-podcast-only (captures ~70% of savings from the fattest source, rejected — same risks in kind, and the savings don't matter given raw isn't the bottleneck).
**Follow-through**: `docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md` §6 `ingest.download` row updated to "all native, zero transcode"; `segment.vad_cut` row updated WAV→FLAC. The implementation gap (yaml still hardcodes `audio_format: "wav"`, no `ingest.download` node) remains open and is now the concrete next fix.

## 2026-07-10 — whisper_v3 retired; `auto_gold` statistical-confidence tier added; `--min-agreement` manifest cuts
**Decision (owner confirmed via AskUserQuestion)**: `Systran/faster-whisper-large-v3+zh` (`whisper_v3`) is retired from the ASR pipeline — `ASR_MODELS["whisper_v3"]["enabled"] = False` in `pipeline/nodes/asr.py`, `pipeline/cli.py` refuses to dispatch it, and `asr.agreement` excludes its `asr_results` text from both the cross-model agreement score and `best_text` candidacy. Its ~618,695 historical rows are kept for audit only, never read by any live node. `tiers.tier` gains a 4th value, `auto_gold`: `agreement >= 0.90 AND canto_ft_confidence > 0.8`, computed 3-way (`canto_ft`/`qwen3_asr`/`sense_voice`) — a **statistical-confidence** tier, explicitly NOT equivalent to human-verified `gold` (`asr_agreement.text_verified` is untouched by it), sample-QA'd via `calibrate.sample(tier='auto_gold')` rather than exhaustively reviewed. `manifest.build`/`export` gained an optional `--min-agreement` cut (writes to separate `manifest_agreeNNN.jsonl` files, never overwriting the default export — hard constraint #9 preserved) for producing smaller, higher-confidence dataset subsets on demand.

**Owner-confirmed parameters**: auto_gold agreement bar ≥0.90; confidence gate = `canto_ft`'s own real (logprob-derived) confidence >0.8 (not `qwen3_asr`/`sense_voice`, whose libraries only expose a nominal 1.0 placeholder — see `pipeline/nodes/asr.py`'s `Qwen3ASRWorker`/`SenseVoiceWorker` docstrings); QA sampling = random ~2-5% per batch via the existing `calibration_review` queue; tier naming keeps `gold` meaning strictly human-verified, with the new statistical tier under its own name (`auto_gold`) rather than redefining `gold`.

**Evidence** (`docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md`, full numbers there): `pipeline/nodes/calibrate.py` had checked earlier the same day that auto-promotion was "not viable" — only 26 segments corpus-wide cleared 4-way agreement ≥0.95. Re-measured excluding `whisper_v3`: 3-way agreement ≥0.90 clears **41.1%** of the corpus (~446h of the 1,068.4h filter-passing pool), and ≥0.95 clears 148h. The original "not viable" conclusion was an artifact of `whisper_v3` disproportionately dragging down the agreement distribution, not a property of the corpus. Cross-checked against calibration-review data (19-43 human-verified samples across this and the prior session): `qwen3_asr` measured ~0.4% CER vs 17-36% for the other backends — `whisper_v3` being the worst performer matches the owner's direct observation that prompted this change (e.g. `Qwen/Qwen3-ASR-1.7B+Cantonese` "1.00 为推动构建新型国际关系，发挥积极嘅作用" — a correct transcription — while `whisper_v3` output was flagged as comparatively unreliable).

**Independent data-hygiene fix found and applied alongside this**: `canto_ft`'s `asr_results.model` string is a `REPO_ROOT`-derived absolute path (`pipeline/nodes/asr.py`'s `_LOCAL_CANTO`), and the repo directory has moved twice historically — ~5.5% of segments (33,921 / 618,695) carried a duplicate `canto_ft` row under a stale path string, double-counting `canto_ft`'s opinion in the agreement average for those segments. `compute_agreement_row()` now dedupes to the current live path (`_LEGACY_MODEL_ALIASES` in `pipeline/nodes/asr.py`).

**Backfill**: production `metadata/corpus.duckdb` — `asr_agreement` (all rows, 3-way recompute + dedupe, `text_verified` deliberately never touched) and `tiers` (`provenance='tier_assign'` rows only, via a single SQL `CASE` re-derivation against the refreshed `asr_agreement` — legacy P0-imported rows and `provenance='calibrate_verify'` human-gold rows are never touched) backfilled via a one-time scratchpad script following this project's established backup→bulk-UPDATE→verify→delete-backup discipline (see the 2026-07-09/10 SenseVoice/Qwen3-ASR OpenCC backfills for precedent). Ran concurrently with the owner's live `pipe calibrate serve` review session by minimizing the RW-lock hold window (read+compute against a read-only connection; only the final bulk UPDATEs open a brief retry-guarded RW connection).

**Alternatives considered**: keep `whisper_v3` in the agreement calculation but down-weight it (rejected — adds a tunable weighting scheme for a model whose output nobody trusts; simpler to remove outright, consistent with "I would like to remove or skip this asr in the pipeline"); redefine `gold` itself to include statistical-confidence segments (rejected by owner — keeps the human-verification guarantee unambiguous, avoids retroactively changing what every existing `gold`-tagged row means); use `qwen3_asr`/`sense_voice` confidence in the auto_gold gate (rejected — both are hardcoded nominal placeholders, not a real quality signal, unlike `canto_ft`'s logprob-derived confidence).

**Rationale**: unblocks scaling human review from "100% by hand" (infeasible past ~100h) to "sample-based statistical QA" (feasible at 1000h+), without weakening the meaning of the existing `gold` tier or silently trusting a known-inaccurate ASR backend. See `docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md` for the recommended `--min-agreement` cut per target dataset size (100h→0.95, 500h→0.85, 1000h→0.65/unchanged) and the corresponding suggested QA sample rates.

## 2026-07-11 — Tier thresholds tightened; new `bronze` tier; risk-scaled QA sampling
**Decision (owner request, confirmed via AskUserQuestion)**: raise the verification-confidence tier bars and add a 5th tier: `auto_gold` 0.90→**0.95** (agreement) with the `canto_ft_confidence > 0.8` gate unchanged/retained; `silver` 0.65→**0.85**; new **`bronze`** tier at agreement≥**0.70** (below that is `excluded`). The manifest-eligibility floor therefore also rises 0.65→0.70 — segments with agreement in [0.65, 0.70) that were `silver` under the 2026-07-10 scheme are now `excluded`. This is a stricter, more conservative re-cut of the same corpus, not an additive change. Boundaries are all inclusive (`>=`), matching the existing `auto_gold`/`silver` code convention; the `canto_ft_confidence` gate stays exclusive (`> 0.8`), unchanged from 2026-07-10.

**Owner-confirmed parameters**: boundary semantics `>=` (not `>`) for all three agreement cutoffs; `canto_ft_confidence > 0.8` gate **retained** for `auto_gold` even at the raised 0.95 agreement bar (not dropped); QA sample rate is now **risk-scaled per tier** rather than a flat 2-5% — `auto_gold` ~1-2%, `silver` ~3-5%, `bronze` ~8-12% (`pipeline/nodes/calibrate.py`'s `QA_SAMPLE_RATE_BY_TIER` + `recommended_sample_n()`); backfill production `tiers` immediately rather than waiting on the pending pilot QA review.

**Evidence** (re-run against the corpus as it stood after the 2026-07-10 whisper_v3-retirement backfill, 484,832 filter-passing segments / 1,068.4h): new tier distribution — `gold`=43 (0.1h), `auto_gold`=72,014 (150.6h), `silver`=235,646 (542.9h), `bronze`=151,140 (325.3h), `excluded`=25,989 (49.5h, up from 13,533/25.7h under the 2026-07-10 floor). Manifest-eligible pool: 458,843 segments / 1,018.9h / 8,817 speakers (down from 471,299 / 1,042.7h / 8,981 — the floor raise from 0.65 to 0.70 removes ~12,456 segments / ~23.8h outright).

**Backfill**: pure-SQL `UPDATE tiers ... CASE ...` re-derivation against the already-correct `asr_agreement` table (no agreement recompute needed this time — only the tier-assignment thresholds changed) — scoped to `provenance = 'tier_assign'` only, `provenance = 'calibrate_verify'` (43 human-gold rows) verified untouched. Ran via the established backup→bulk-UPDATE→verify→delete-backup discipline; completed in 2.3s.

**Pilot QA batches**: the pre-existing 300-segment `auto_gold` pilot batch (`calibrate_sample_fd9269e121be`, queued 2026-07-10, still 100% pending) only has 109/300 segments still qualifying as `auto_gold` under the new 0.95 bar (191 slid to `silver`) — left in place (still useful QA signal, just mixed-tier) rather than deleted. Three fresh 300-segment pilot batches queued under the new tier definitions: `calibrate_sample_db47fe903b98` (auto_gold), `calibrate_sample_9110dfd46076` (silver), `calibrate_sample_557c2ee7bc99` (bronze). Full risk-scaled sample sizes (`recommended_sample_n()`) are much larger (~1,080 / ~9,426 / ~15,114 respectively) — these 300-each batches are pilots, not the full scaled sample; scaling up is a follow-on step once the owner reviews pilot error rates.

**Rationale**: the owner judged the 2026-07-10 bars too permissive for what should count as gold-equivalent/manifest-eligible without human review; tightening them (and adding a dedicated `bronze` floor tier that gets the heaviest QA scrutiny) trades corpus size for higher average per-tier trustworthiness, while the risk-scaled QA rate concentrates human review effort on the tier most likely to contain errors instead of spreading it flat across all three.

## 2026-07-11 — Repo hygiene cleanup: HC#9 remediation, legacy script retirement, requirements.txt removal
**Decision (owner confirmed via `docs/PIPELINE_REVIEW_2026-07-11.md`, four premises confirmed via AskUserQuestion)**: executed Phase C of the cleanup plan in one commit — (1) `reconstruct.py`, `reconstruct_dead_sources.txt`, and `scripts/10_enrich_manifest.py` removed from the public repo tip via `git rm` (Hard Constraint #9 violation — these are dataset-reconstruction-recipe tooling, which the zero-risk policy explicitly forbids publishing, even though they contain no source URLs or audio themselves); a local-only copy is kept in `metadata/release_dormant/` (gitignored, never committed) for if the dormant release policy is ever reactivated. (2) 17 further legacy `scripts/*.py` removed via `git rm` — all fully superseded by ported `pipeline/nodes/` DAG nodes (00_reingest, 01_discover, 02_download, 03_segment, 03b_acoustic_pregate, 04_transcribe, 05_calibrate, 06_filter, 07_g2p, 08_speaker_id, 09_manifest, 11_audio_tag, 12_language_id, 13_overlap_detect, backfill_downloaded_jsonl, fix_stale_asr_model_manifest, fix_stale_paths, test_sensevoice) — one-off backfill/hotfix scripts already verified complete in prior sessions, ported stages already running in production. `scripts/10_report.py` is explicitly **kept** — no `report.build` node exists yet (Issue #3 in the review doc), so it remains as the port reference until that node lands. (3) `requirements.txt` removed via `git rm` — it was stale (missing duckdb/qwen-asr/funasr/opencc) and dual-tracked against `pyproject.toml`+`uv.lock`, a genuine risk of environment breakage if someone installed from it. `README.md` updated to point installation/usage at `uv pip install -e .` and `python -m pipeline.cli run <node>` instead.

**Owner-confirmed parameters**: `git rm` + preserve git history — no `git filter-repo` purge. Old versions of `reconstruct.py`/`reconstruct_dead_sources.txt`/`scripts/10_enrich_manifest.py` remain retrievable from history (`git show <sha>:path`); the owner is aware of and accepts this (a full history purge was considered and explicitly rejected as unnecessary — the actual leaked surface is a reconstruction *methodology*, not source URLs or audio, which were never committed). Legacy scripts handled with tiered treatment (not a blanket wipe) — only fully-ported, verified-complete files were removed.

**Not done in this commit**: **push** — the commit is local only; `docs/PIPELINE_REVIEW_2026-07-11.md` §4 requires owner review of the diff before pushing (this also carries Issue #14's pre-existing 8-commit-behind-origin backlog along with it). Phase D (`.venv_ina/` 6.8GB removal, a pure-disk operation with no git or catalog impact) was executed separately in the same session, immediately after this commit — zero-reference confirmed by a final `grep -r venv_ina|inaSpeech` across the codebase before deletion.

**Rationale**: closes the one High-severity finding from the 2026-07-11 pipeline review (public repo carrying reconstruction tooling that contradicts the project's own zero-risk data policy) while using the same "port verified, delete source, rely on git history as archive" discipline the project already uses for one-off backfill scripts — consistent with the external best-practice research cited in the review doc (§5: "one-off backfill scripts: verify then delete from git, rely on history as archive").

## 2026-07-13 — `canto_ft` retired (2nd ASR backend after `whisper_v3`); T15 backlog throughput investigation
**Decision (owner confirmed via AskUserQuestion)**: `canto_ft` (Cantonese fine-tuned Whisper large-v2, faster-whisper/ctranslate2) is retired from the ASR pipeline, following the exact same mechanism as `whisper_v3`'s 2026-07-10 retirement — `ASR_MODELS["canto_ft"]["enabled"] = False` in `pipeline/nodes/asr.py`, `pipeline/cli.py`'s guard rail refuses to dispatch it, and `asr.agreement` excludes its `asr_results` text from both the cross-model agreement score and `best_text` candidacy (`EXCLUDED_FROM_AGREEMENT` now `{"whisper_v3", "canto_ft"}`). Its historical rows are kept for audit only, never read by any live node going forward. Two active ASR backends remain: `qwen3_asr`, `sense_voice`. `pipeline/cli.py`'s stale default `--models canto_ft,whisper_v3` (both already-retired models) is fixed to `qwen3_asr,sense_voice`.

**Evidence — throughput investigation on the T15 reingest backlog (578,889 segments)**: this session set out to run `asr.transcribe` for T15 across all 3 then-active models (canto_ft/qwen3_asr/sense_voice) with both GPUs fully utilized throughout, per an earlier owner request. Sequence of measurements:
1. Interleaved (3 models × 2 GPUs, 6 workers, per-device pools shared via a `target=1` semaphore): combined throughput stabilized at only **9.2/s**, projecting a ~51h ETA — far over the original 5-7h estimate.
2. Hypothesis: 3-model context-switching overhead on shared per-device pools. Fix attempted: sequential-exclusive execution, one model at a time with exclusive access to both GPUs (`run_t15_asr_sequential.sh`). Result: `canto_ft`-alone-on-both-GPUs still measured only **8.9/s** — nearly identical to the interleaved rate, disproving the context-switching hypothesis.
3. A red herring surfaced during this: every `registry.snapshot()` debug log line showed exactly one device pool `in_use:1`, the other `in_use:0`, suggesting serialization even across separate devices. Directly measured via `nvidia-smi pmon -c 5` (no sudo needed): both `canto_ft` worker PIDs (one per GPU) showed 66-88% SM utilization **simultaneously** across all 5 samples — proving genuine parallel GPU execution was happening; the log pattern was an artifact of when that specific line fires (right after a batch completes), not real serialization.
4. Root cause found via code reading, not further experimentation: `TranscribeWorker.forward_batch()` (faster-whisper/ctranslate2 backend, used by `canto_ft`) does explicit sequential per-item decode (`return [self._transcribe_one(y16) for y16 in items]`) — no batched-tensor API — matching the legacy script's behavior for golden-parity. This is an inherent architectural ceiling (~4.45/s/GPU), not a bug or tunable.
5. Researched (via `agy -p ... --model "Gemini 3.1 Pro (High)"` + WebSearch, per owner instruction) whether `faster-whisper`'s `BatchedInferencePipeline` could lift this ceiling. Conclusion: **no speedup for our use case** — our segments are already pre-cut to 3-20s, each fitting in ≤1 VAD chunk internally, so the pipeline's batching produces an effective batch size of 1. It also carries a confirmed accuracy/parity risk (GitHub issue #1179: "degrades transcription quality heavily" via lost cross-chunk context and different VAD-boundary chunking vs. the current sliding-window approach) — incompatible with this project's golden-parity discipline. A separate experimental method, `transcribe_batch_multiple_audios` (unmerged/recent PRs #1302/#1359), might suit batching multiple distinct short files but is unproven — not pursued.
6. Cross-checked against calibration-review CER data already on file (2026-07-10 entry): `canto_ft` measured in the same poor 17-36% CER band as the already-retired `whisper_v3`, vs. `qwen3_asr`'s ~0.4% — the same "slow AND inaccurate" profile that justified `whisper_v3`'s retirement.

Given both a hard architectural speed ceiling and comparably poor accuracy, owner decided to retire `canto_ft` outright rather than pursue further batching/scheduling workarounds.

**A second, independent throughput bug found and fixed while restarting T15 with only the 2 remaining models**: running `qwen3_asr` and `sense_voice` interleaved on the same 2 GPUs (`--models qwen3_asr,qwen3_asr,sense_voice,sense_voice`) reproduced the same low combined rate (~16.9/s, worse than expected). `nvidia-smi pmon -c 5` showed the `sense_voice` workers at **0% SM across all 5 samples** — full starvation, not merely slow sharing — while `qwen3_asr` held each device's `target=1` semaphore continuously. Fixed by reverting to sequential-exclusive execution (one model at a time, each with exclusive use of both GPUs) — `run_t15_asr_sequential.sh` updated to `qwen3_asr` then `sense_voice` (canto_ft stage removed). Note this differs from the earlier (canto_ft-involving) sequential-vs-interleaved test: there, sequential didn't help because `canto_ft`'s own decode speed was already the binding constraint regardless of contention; here, `qwen3_asr`/`sense_voice` are not inherently that slow, so the per-device semaphore contention is the actual bottleneck and sequential-exclusive execution is the correct fix.

**Known consequence — `auto_gold` tier gate (NOT resolved this session, follow-up required)**: `tier.assign`'s `auto_gold` gate (`agreement >= 0.95 AND canto_ft_confidence > 0.8`) was deliberately built on `canto_ft`'s confidence specifically because it was the only active model exposing a real logprob-derived confidence — `qwen3_asr`/`sense_voice` both report a nominal `1.0` placeholder (explicitly rejected as a confidence source in the 2026-07-10 decision). With `canto_ft` retired, `canto_ft_confidence` is always `None` for new segments, which `assign_tier()` already treats as failing the gate (no code change needed there — it fails closed automatically) — so **new segments cap at `silver`/`bronze` until a 2-model-agreement-only `auto_gold` threshold is adopted**. Owner's direction: default to `qwen3_asr` as primary, but the new threshold (e.g. a higher pure-agreement bar such as ≥0.9+) must be set from real agreement-distribution statistics, not guessed — mirroring the data-driven approach in `docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md`. This requires a follow-up backfill/recompute pass of `asr_agreement` across the already-processed corpus (excluding `canto_ft`, similar to the 2026-07-10 whisper_v3 backfill) to get real 2-model agreement distribution data, then an owner decision on the new bar. Tracked in `pending_task.md`'s T15 entry.

**Alternatives considered**: keep `canto_ft` for a small sampled fraction just to preserve a confidence signal for `auto_gold` (rejected for now — adds a "partial coverage" concept not otherwise in the pipeline design; simpler to accept the `auto_gold` gap and revisit with real stats); pursue `BatchedInferencePipeline` or the experimental multi-audio batching API to keep `canto_ft` viable (rejected — no speedup for pre-segmented short clips, and/or unproven accuracy impact, respectively).

**Rationale**: `canto_ft` was both the throughput bottleneck (hard architectural ceiling, not fixable by scheduling/batching changes explored this session) and a comparably poor accuracy performer — the same combination that justified `whisper_v3`'s retirement 3 days earlier. Removing it and fixing the newly-discovered `sense_voice` starvation bug together should bring T15's ETA down substantially from the originally-projected ~51h; real throughput to be confirmed once the sequential `qwen3_asr` → `sense_voice` run settles.

**Addendum (same day, later) — third throughput bug: CLI `--batch` default starved `qwen3_asr`'s tuned batch capacity (fixed, 2.4× gain)**: with the sequential-exclusive script running at the CLI default `--batch 8`, `qwen3_asr` on both GPUs measured only **~17.4/s combined** with 5-6GB/24.5GB VRAM and ~30-43% SM per GPU — suspiciously under-utilized. Code trace: the CLI's `--batch` flag sets the supervisor's per-dispatch chunk size (`_batches()` in `pipeline/nodes/asr.py`), while `Qwen3ASRWorker.load_model()` sets `max_inference_batch_size=64` with a 2026-07-07 empirical tuning curve in the comment (8≈8.7/s/GPU, 64=30.1/s/GPU) — i.e. the supervisor was only ever feeding 8-item chunks to a model tuned for 64. Fix: added `--batch 64` to both invocations in `run_t15_asr_sequential.sh` and relaunched. Measured result: **42.6/s combined steady-state** (2.4× the batch-8 rate, above the 36.3/s historical dual-GPU benchmark), VRAM 18-21GB/GPU, 50-62% SM, zero errors/OOM over 100k+ segments. T15's qwen3_asr pass ETA dropped to ~3.5-4h. Follow-up (tracked in `docs/PIPELINE_REVIEW_2026-07-13.md` Issue #19): make this structural — per-model `dispatch_batch` in `ASR_MODELS` or raise the CLI default, so a future bare `pipe run asr.transcribe` doesn't silently run 2.4× slower.

## 2026-07-13 — `calibrate serve` offline mode: JSON snapshot reads + JSONL decision buffer (never blocks on the catalog)
**Problem**: `pipe calibrate serve` crashed outright at startup (`connect_ro(CATALOG_PATH).close()` in `cmd_calibrate_serve`, no retry) whenever a long batch node (e.g. T15's `asr.transcribe`) held the DuckDB writer lock — confirmed live against the real catalog while T15 was mid-run. Worse, even the 2026-07-10 per-request-connection redesign (short-lived connections instead of one held for the whole server session) only helps with *brief* overlaps: DuckDB's file lock is per-process and held for a batch node's ENTIRE runtime (hours), so `connect_ro()` fails for every single read too, not just writes, for the whole duration.

**Decision (owner requested, confirmed reasonable + scoped via AskUserQuestion)**: two changes, both in `pipeline/nodes/calibrate.py` / `pipeline/tools/calibrate_server.py` / `pipeline/cli.py`:
1. **Writes always buffer to a local JSONL** (`metadata/calibration_pending_decisions.jsonl`, `append_pending_decision()`) instead of calling `record_decision()` inline from the HTTP handler — unconditionally, not just as a busy-catalog fallback (owner's explicit choice: keep one code path, not two). `pipe calibrate flush-pending` (new CLI subcommand, `run_calibrate_flush_pending()`) replays the buffer into `record_decision()` whenever the writer is free; safe to re-run (per-id UPDATE is idempotent), and only failed entries are left behind for retry (successes get archived to a timestamped `.flushed_<ts>.jsonl`).
2. **Reads fall back to a JSON snapshot** (`metadata/calibration_offline_queue.json`, `pipe calibrate export-snapshot` / `run_calibrate_export_snapshot()`) taken while the catalog was free, dumping the full pending-review payload (candidates, audio paths, agreement) via a new `pending_queue_rows()` query. `calibrate_server.py`'s `_read_or_offline()` tries a short live-DB retry (shortened from 30s to 4s, since a long fallback wait no longer buys anything) and falls back to the snapshot on `CatalogBusyError`. The local decision buffer is overlaid on top of BOTH the live-DB and offline-snapshot read paths (`_overlay_item()`/`_stats_with_overlay()`) — otherwise a segment just decided locally would still read 'pending' from its source and get re-served. Every read-endpoint JSON response carries a `mode: "live"|"offline"` field; the browser UI shows an amber banner when offline.

**Known limitations, accepted as-is**: the snapshot is a point-in-time dump — new `calibrate.sample` batches queued after it won't appear until the next export; `/api/refill`'s auto-refill still needs a live DB query (no offline equivalent for discovering genuinely new segments) and simply no-ops while offline; offline `summary_stats`/`by_source`/edit-distance breakdowns only reflect decisions made during the current offline session (no DB access to prior history) — the progress bar / pending-vs-decided counts are accurate, the dashboard's finer breakdowns are a partial view until the next flush.

**Verification**: 43 unit tests (11 new) in `tests/test_calibrate_node.py` covering `next_pending(exclude_ids=...)`, `pending_queue_rows`, `run_calibrate_export_snapshot`, `append_pending_decision`/`load_pending_decisions` (including resubmit-keeps-latest and reject-invalid-decision), and `run_calibrate_flush_pending` (success+archive, empty-buffer no-op, partial-failure leaves only the failed entries for retry). Also live-tested against the real (T15-locked) catalog: `pipe calibrate serve` no longer crashes at startup, logs a warning instead, and `/api/stats` returns `{"mode": "offline", ...}` gracefully.

**Rationale**: the owner's proposed shape (buffer decisions to JSON, push to DB whenever free) was the right fix for the actual constraint — a multi-hour writer hold, not a brief race — and unifying "always buffer" (rather than "buffer only when busy") keeps the write path simple and consistent regardless of catalog availability, matching the project's general preference for one code path over conditional branches where the cost is low.

**Addendum (same day, later) — third throughput bug: CLI `--batch` default starved `qwen3_asr`'s tuned batch capacity (fixed, 2.4× gain)**: with the sequential-exclusive script running at the CLI default `--batch 8`, `qwen3_asr` on both GPUs measured only **~17.4/s combined** with 5-6GB/24.5GB VRAM and ~30-43% SM per GPU — suspiciously under-utilized. Code trace: the CLI's `--batch` flag sets the supervisor's per-dispatch chunk size (`_batches()` in `pipeline/nodes/asr.py`), while `Qwen3ASRWorker.load_model()` sets `max_inference_batch_size=64` with a 2026-07-07 empirical tuning curve in the comment (8≈8.7/s/GPU, 64=30.1/s/GPU) — i.e. the supervisor was only ever feeding 8-item chunks to a model tuned for 64. Fix: added `--batch 64` to both invocations in `run_t15_asr_sequential.sh` and relaunched. Measured result: **42.6/s combined steady-state** (2.4× the batch-8 rate, above the 36.3/s historical dual-GPU benchmark), VRAM 18-21GB/GPU, 50-62% SM, zero errors/OOM over 100k+ segments. T15's qwen3_asr pass ETA dropped to ~3.5-4h. Follow-up (tracked in `docs/PIPELINE_REVIEW_2026-07-13.md` Issue #19): make this structural — per-model `dispatch_batch` in `ASR_MODELS` or raise the CLI default, so a future bare `pipe run asr.transcribe` doesn't silently run 2.4× slower.

## 2026-07-13 — Issue #20 fix: `char_agreement()` punctuation/digit normalization (T16 step 1)
**Problem**: `char_agreement()` (`pipeline/nodes/asr.py`) compared raw ASR text with zero normalization. `qwen3_asr` (AR, transformers) infers punctuation from LM context; `sense_voice` (CTC, funasr) never emits punctuation. Comparing the two raw strings systematically deflated cross-model agreement on punctuation alone — confirmed as the top AR-vs-CTC comparison pitfall by the targeted external research in `docs/PIPELINE_REVIEW_2026-07-13.md` §5 Q3. With only 2 active models post-`canto_ft`-retirement, agreement is the sole trust signal feeding `tier.assign`, so this bias directly skews the tier distribution. T16 (rebuilding the `auto_gold` gate) requires this fixed *before* the agreement-distribution analysis, or the analysis itself is biased.

**Decision**: added `_normalize_for_agreement()` in `pipeline/nodes/asr.py`, called from inside `char_agreement()` only — strips all Unicode punctuation (`unicodedata.category(ch).startswith("P")`, covers ASCII and CJK marks alike without a hand-maintained charset) and folds Arabic/full-width digits to CJK numerals (`0-9` and full-width `０-９` → `〇一二三四五六七八九` via `str.translate`) before running `difflib.SequenceMatcher`. Comparison-only: `compute_agreement_row()`'s `best_text` and the stored `asr_results`/`asr_agreement` text are unaffected — normalization never touches what gets persisted, only what gets compared.

**Verification**: 5 new/extended tests in `tests/test_asr_node.py` (punctuation-mismatch no longer deflates agreement, Arabic/full-width digit folding, comparison-only — original strings unmutated, and an explicit `compute_agreement_row` case proving `best_text` keeps the original punctuation/digits while `agreement` uses the normalized comparison). All 53 tests in that file pass. Not yet exercised against the live catalog — T15 held the DuckDB writer lock throughout this change; the T16 step-2 full-corpus backfill will be the first real-data run.

**Rationale**: matches the industry-standard practice cited in the review doc's research (strip punctuation + normalize digits before overlap scoring when comparing AR vs. CTC transcripts) and unblocks T16's step 2 (backfill) and step 3 (distribution analysis) without which any threshold the owner picks in step 4 would be calibrated against a biased signal.

## 2026-07-13 — `sense_voice` throughput bug: `batch_size_s` kwarg is a no-op without a `vad_model` (fixed, restarted, verified — 2.4× throughput)
**Problem**: sense_voice's T15 pass measured ~36/s combined — in the same ballpark as qwen3_asr's already-slow 17.4/s-before-fix and only modestly above its 42.6/s-after-fix rate, despite SenseVoice-Small's documented ~105× RTF (`SenseVoiceWorker` class docstring: "entire 618k corpus in ~10 minutes on 2 GPUs"). Grepped the live run's log: **100% of ~38k logged inference steps showed `'batch_size': '1'`**, no exceptions — every item was being decoded one at a time despite the supervisor dispatching 64-item chunks.

**Root cause (code trace into `funasr/auto/auto_model.py`, installed in `.venv`)**: `AutoModel.generate(**cfg)` (line 442) routes to `inference()` when `self.vad_model is None` (our `SenseVoiceWorker.load_model()` never configures a `vad_model=`) — the alternate `inference_with_vad()` branch is never taken. Plain `inference()` (line 531) reads `batch_size = kwargs.get("batch_size", 1)` — it does **not** read `batch_size_s` at all. The `batch_size_s`-driven dynamic batching (line 639: `batch_size = max(int(kwargs.get("batch_size_s", 300)) * 1000, 1)`) exists exclusively inside `inference_with_vad()`. `SenseVoiceWorker.forward_batch()` (`pipeline/nodes/asr.py`) was passing `batch_size_s=300` — a parameter name that is valid in `generate()`'s general kwarg surface (and documented in its docstring) but silently ignored on the exact code path we hit, so `batch_size` fell back to its default of 1 every single call.

**Fix**: `pipeline/nodes/asr.py`'s `SenseVoiceWorker.forward_batch()` now passes `batch_size=len(items)` — the literal item count of whatever chunk the supervisor already dispatched (currently 64, via `run_t15_asr_sequential.sh`'s `--batch 64`) — instead of the no-op `batch_size_s=300`. This makes funasr's `inference()` loop (`for beg_idx in range(0, num_samples, batch_size)`) execute exactly one real batched `model.inference()` call per dispatched chunk instead of `len(items)` separate single-item calls. Test fixture `_FakeSenseVoiceModel.generate()` in `tests/test_asr_node.py` updated to accept `batch_size=` and assert it equals the dispatched item count; all 53 tests in that file pass.

**Applied and verified** (owner explicitly confirmed "please kill it and restart now" after an initial AskUserQuestion timed out with no response — the Claude Code auto-mode classifier blocked two kill attempts before that point, correctly distinguishing a direct answer to a live question from an inferred/timed-out one): killed the pre-fix run (PID 1753479/1617642 at 94,720/578,849 processed — that work is preserved in the catalog, not lost) and restarted `run_t15_asr_sequential.sh` (new PID 1772752). `qwen3_asr` re-discovered 0 remaining rows via its normal idempotent anti-join (confirms its earlier pass really was complete) and no-opped instantly; `sense_voice` resumed the remaining 479,264 segments. Log confirms the fix: every `forward_batch()` call now reports `'batch_size': '64'` (was `'1'`) and `rtf` dropped to ~0.001 (was ~0.010-0.017). **Measured steady combined throughput: ~87.8/s, a 2.4× improvement over the pre-fix ~36/s.** One open observation, not chased further this session: `nvidia-smi` showed only 0-4% GPU utilization right after the restart despite this throughput, suggesting the bottleneck may have shifted from GPU compute (now near-instant per 64-item batch) to CPU-side audio load/feature-extraction between batches — worth investigating if sense_voice needs to go faster still. Phase C auto-runner relaunched pointed at the new PID (background task `bbdphc24e`).

**Rationale**: same class of bug as the earlier `qwen3_asr --batch 8` default under-feeding its `max_inference_batch_size=64` (2026-07-13, same day) — a mismatch between the supervisor's dispatch chunk size and what the underlying model API actually consumes as its real batching knob, silently defaulting to serial per-item processing instead of erroring, so it went unnoticed until log inspection.

## 2026-07-14 — ASR decode+resample bottleneck: soxr swap + 3-stage worker pipeline + supervisor prefetch

**Problem**: with sense_voice's forward pass properly batched (2026-07-13 fix), throughput
plateaued at ~30-75/s with 0-4% GPU utilisation — the same anomaly flagged in the 2026-07-13
entry. Root cause (measured 2026-07-14): the per-batch CPU preprocessing stage
(FLAC decode via libsndfile + 48k→16k resample_poly) ran strictly BEFORE the GPU forward with
zero overlap, and the supervisor's `target=1` pool acquire spanned send+read, so at any moment
either the CPU or the GPU was idle. Measured: cold-cache decode+resample ~76 ms/file
(I/O-dominated; warm-cache only ~8 ms/file — decode 3.9 ms + scipy resample 4.1 ms) vs a
~0.55 s forward for a whole batch of 64. The funasr-printed `rtf` only times the model
compute, which is why logs showed rtf≈0.001 while wall-clock throughput stayed low.

**Decision** (owner-approved full-suite, applied to FUTURE runs only — the live T15 run was
left untouched and drains with the old code):
1. `_load_and_resample()` now uses `soxr.resample(quality="HQ")` instead of
   `scipy.signal.resample_poly(wav, 1, 3)` — measured 3.8× faster on the resample step on
   real corpus files; same libsoxr engine as audio/bus.py and librosa ≥0.10's default.
   The original scipy choice existed only for golden-parity with the now-retired
   faster-whisper models (whisper_v3 2026-07-10, canto_ft 2026-07-13), so the constraint
   no longer binds. Verified online: soxr is the standard fast high-quality resampler
   (jonashaag/audio-resampling-in-python benchmark; librosa default since 0.10).
2. `worker_main()` is now a 3-stage threaded pipeline (stdin reader → preprocess thread
   with the io-workers ThreadPoolExecutor → main-thread GPU+emit), bounded queues
   (maxsize=2), single stdout writer — the standard DataLoader-prefetch-style
   producer-consumer pattern, so decode of task N+1 overlaps the GPU forward of task N.
3. Supervisor `worker_loop()` keeps up to `--prefetch` (default 2) tasks in flight per
   worker, acquires the device pool around the SEND only (foreign-GPU yield still gates
   new dispatches; in-flight work is never preempted), and matches results by task_id
   (late stragglers after a timeout are dropped by an unknown-task_id guard instead of
   being attributed to the wrong batch). `--prefetch 1` restores the old behaviour.
4. Worker `--io-workers` default 8→16, now passed through from the supervisor/CLI.

**Verification**: 56/56 tests in tests/test_asr_node.py green, including 3 new supervisor
regression tests (fake WorkerHandle asserting the in-flight high-water mark is exactly
`prefetch`, prefetch=1 restores sequential dispatch, task_ids unique). Full suite: same
4 failed + 14 errors as the pre-existing T15-writer-lock baseline, zero new regressions.
**Live-GPU validated the same day** (sooner than planned — the suspend-wedge incident below
freed the writer lock): `--limit 512` on both GPUs completed 512/512, 0 errors, then the
T15 sense_voice remainder (97,488 segments) was relaunched on the new code path at ~85/s
early steady-state — on the duration-DESCENDING tail of the queue (longest clips), where
per-item rates are inherently lowest, so this comfortably clears the old code's ~88/s
average over shorter clips.

**Not done (deliberately)**: single-pass dual-model dispatch ("share one decode between
qwen3_asr and sense_voice") — owner wants post-fix throughput data first before deciding
whether the extra structural complexity is justified.

## 2026-07-14 — T15 sense_voice pass wedged by machine suspend/resume; killed + remainder relaunched on the new pipelined code

**What happened**: the machine was suspended ~10:43 (2026-07-14) mid-run. On resume, the
cuda:0 sense_voice worker subprocess never completed another batch: its main thread sat in
R state burning ~85% of one core with 0% GPU utilisation — the signature of torch
busy-polling a CUDA context wedged by the suspend. The supervisor's 600 s `read_message`
timeout then fired every 10 minutes from 10:52:59 onwards ("batch failed:" with an empty
message — `asyncio.TimeoutError` has an empty str), each cycle burning one 64-segment batch
(12 cycles / 768 segments marked as errors before intervention). cuda:1's worker had
finished its whole shard cleanly before the suspend. Left alone, the remaining ~97k
segments (all on cuda:0's shard) would have churned for ~10 days. Progress-log gotcha
worth remembering: the cumulative `(N/s)` in the INFO lines masks this failure mode
completely — cuda:1's progress lines kept printing while cuda:0 silently died, and the
per-batch tqdm output stopping is only visible in the raw log tail.

**Action (owner-approved kill, third kill of this run overall)**: stopped the Phase C
auto-runner first (so it wouldn't misread the kill as "T15 finished" and fire the drain
chain), killed the T15 process tree (the wedged worker needed SIGKILL — SIGTERM was
ignored, consistent with a spin loop), verified the DuckDB writer lock was freed and the
already-completed 380k sense_voice rows were safely in the catalog. The 768
timeout-errored segments have no asr_results rows, so idempotent discovery re-surfaces
them — zero data loss.

**Silver lining**: the freed writer lock allowed the live-GPU validation of the same-day
decode+resample pipelining fix (entry above) immediately instead of after T15's drain:
`--limit 512` → 512/512, 0 errors; then the full 97,488-segment remainder was relaunched
on the NEW code (log `metadata/logs/t15_sense_voice_remainder_pipelined_20260714.log`)
and the Phase C auto-runner re-pointed at the new supervisor PID.

**Future-work note (not implemented)**: the supervisor has no defence against a wedged
worker — it happily feeds new tasks into a 600 s-timeout loop forever. A cheap guard:
after N consecutive read timeouts from the same worker (e.g. 3), kill and respawn that
worker subprocess (or abort the run loudly). Tracked in pending_task.md T15 notes.
Suspending the machine during any GPU batch run should be avoided until then.

## 2026-07-14 — `filter.acoustic` moved to GPU (onnxruntime-gpu) — CPU worker scaling had hit a wall

**Context**: mid-run on the Phase C1 drain chain (post-T15), `filter.acoustic` (SNR +
DNSMOS) was measured at only ~21/s with the default `--workers 4` (each worker capped to
`--threads 4` onnxruntime intra-op threads, per the existing oversubscription-avoidance
comment in `pipeline/nodes/filter.py`). Owner asked to try speeding it up.

**Diagnosis**: doubling to `--workers 8` roughly doubled CPU usage (~18.8 → ~35.3 cores on
the 48-core box) but throughput barely moved (~21 → ~21.6/s marginal) — confirmed via a
standalone benchmark script that itself got starved for minutes by the already-saturated
CPU, not just an artifact of onnxruntime's intra-op busy-spin. `nvidia-smi` showed both
RTX 4090s essentially idle (0%/26%) — this stage never used GPU at all; the installed
`onnxruntime` package was CPU-only (`get_available_providers()` → `['CPUExecutionProvider']`
only).

**Action**: swapped `onnxruntime` → `onnxruntime-gpu` (`uv pip uninstall onnxruntime && uv
pip install onnxruntime-gpu` — never `uv sync`, see the standing uv-sync-danger note).
`onnxruntime-gpu==1.27.0` needs `libcudart.so.13`/cuDNN, not present as a system CUDA
toolkit install — resolved by pointing worker subprocesses' `LD_LIBRARY_PATH` at torch's
own pip-bundled CUDA 13 runtime libs (`site-packages/nvidia/{cu13,cudnn,cuda_runtime}/lib`)
instead of requiring a separate system install. Code changes in `pipeline/nodes/filter.py`
(`_build_capped_dnsmos()` / `AcousticWorker` / `worker_main()` / `run_filter_acoustic()`)
and `pipeline/cli.py` (new `filter.acoustic --gpu 0,1` flag, comma-separated CUDA device
ids the worker pool round-robins across; omitting `--gpu` keeps the CPU-only path
unchanged, so this is backward compatible for any environment without a GPU).

**Correctness verification (required before trusting this on the real backlog)**: recomputed
`sig_mos`/`ovrl_mos` for 5 real already-scored segments on both CPU and the new GPU session
— **exact match, |Δ|=0.0** on all 5 (rounded to 2dp, same as stored). Single-stream
benchmark: GPU 17.95 ms/call vs CPU 130.10 ms/call for `compute_dnsmos()` (~7.2×).

**Result**: `--workers 8 --gpu 0,1` (4 workers/GPU) reached ~114-122/s marginal — **~5.5×**
the original 4-worker CPU rate — with GPU utilization still only 10%/27%, so headroom
remained; bumped to `--workers 16 --gpu 0,1` (8 workers/GPU) same session to push further
(see pending_task.md for the final measured rate once stable). `metadata/logs/
phase_c_resume_runner.sh` is the live resume script — updated in place each time the
worker count changed (this is the 3rd restart of this same filter.acoustic backlog: 4
CPU → 8 CPU → 8 GPU → 16 GPU). Each restart is safe: `filter.acoustic`'s discovery is a
plain anti-join on already-written `filters_acoustic` rows, so a mid-run kill+relaunch
never redoes or loses committed work (confirmed empirically: totals-to-process dropped
correctly across every restart).

**Not done / follow-up**: no `tests/` regression test added yet for the new `--gpu` code
path (worth adding once the flag's default value for future runs is decided — currently
opt-in only, existing CPU-only behavior is the default). Consider whether other
CPU-worker-pool nodes with idle-GPU headroom (none currently identified) would benefit
from the same pattern.

## 2026-07-14 — `filter.decide` OOM-killed at 430k/578,889 rows — single-giant-transaction MVCC memory growth

**Symptom**: the very next node in the Phase C1/C2 chain after `filter.acoustic` (GPU,
above) — `filter.decide` — died with `_duckdb.OutOfMemoryException: failed to allocate
data of size 2.0 KiB (201.2 GiB/201.2 GiB used)` at 430,000/578,889 rows decided
(20:09:59). `free -h` immediately after showed 118GiB free / 219GiB available on this
251GiB box — the box itself was never short of physical RAM.

**Diagnosis**: `201.2 GiB` is DuckDB's default `memory_limit` (~80% of detected system
RAM: 251GiB × 0.8 ≈ 200.8GiB, matches). `run_filter_decide()` (`pipeline/nodes/filter.py`)
wrapped **all 578,889 rows in one single transaction** (`conn.begin()` before the batch
loop, `conn.commit()` after) — a deliberate optimization to avoid a WAL-checkpoint flush
per batch. The `filters` table was pre-populated by the P0 legacy import (455,299 rows,
`provenance IS NULL`), so almost every `INSERT OR REPLACE` this node performs is really a
delete+insert over an existing PK row. Inside an uncommitted transaction, DuckDB must hold
the MVCC undo/version-chain state for every such delete+insert in memory until commit — so
memory grew monotonically with rows processed and had no way to shrink until the (never
reached) final commit. The 430k-row point where it died is consistent with this: enough
delete+insert version-chain state accumulated to hit the 80%-of-RAM default cap.

**Fix**: chunked the transaction — commit every `COMMIT_EVERY_ROWS = 50_000` rows instead
of once for the whole backlog, re-`begin()` immediately after each commit. Bounds
per-transaction MVCC memory to a small fraction of the box's RAM while still committing
far less often than once-per-5000-row-batch (the original slow baseline this optimization
was fixing). Since the crashed run's `except: conn.rollback()` fired inside the one giant
transaction, **none** of the 430k already-decided rows were actually persisted — the
anti-join discovery (`filters.provenance = 'filter_decide'`) correctly finds all 578,889
rows still pending on restart, so no data was lost and nothing needs manual cleanup.
`tests/` filter suite (33 tests) still green after the change.

**Not done / follow-up**: `pipeline/nodes/rebalance.py` and `pipeline/nodes/raw_flac.py`
use a similar `conn.begin()/commit()` transaction pattern but per-*item* (one row's worth
of UPDATEs per transaction), not per-whole-backlog — not at risk of the same failure mode,
not changed. No regression test added yet asserting `filter.decide` commits periodically
under a large backlog (would need a large-N synthetic catalog fixture to be meaningful).

## 2026-07-15 — `auto_gold` gate rebuilt (T16): `canto_ft_confidence` → `filters.dnsmos`, agreement bar 0.95→0.92
**Problem**: `tier.assign`'s `auto_gold` gate (`agreement >= 0.95 AND canto_ft_confidence >
0.8`) was built on `canto_ft`'s logprob-derived confidence as its non-ASR trust signal.
`canto_ft` retired 2026-07-13 (see that day's entry above) — `canto_ft_confidence` is
always `NULL` for every segment processed since, so the gate failed closed and new
segments capped at silver/bronze regardless of how good their ASR agreement was
(`docs/PIPELINE_REVIEW_2026-07-13.md` Issue #17). Separately, `char_agreement()` compared
raw ASR text with no punctuation/digit normalization, systematically deflating agreement
between `qwen3_asr` (AR, infers punctuation) and `sense_voice` (CTC, emits none) — fixed
2026-07-13 in code (`_normalize_for_agreement()`, Issue #20) but not yet applied to
existing `asr_agreement` rows, which still reflected pre-fix, pre-canto_ft-exclusion
3-way scores.

**Backfill** (owner-approved, mirrors the 2026-07-10/11 precedent's backup→bulk-UPDATE→
verify discipline, both scripts in this session's scratchpad):
1. `backfill_agreement_t16.py` — RO fetch of all 1,241,610 ids with ≥2 `asr_results` rows,
   Python recompute via the already-normalized/canto_ft-excluding `compute_agreement_row()`
   (73.9s), checkpoint + file-copy backup, single `UPDATE asr_agreement ... FROM
   <registered df>` (deliberately excludes `text_verified` from the SET list — never
   touched). Total 84.8s. Verified: `text_verified=True` count unchanged at 58 pre/post.
   Effect: normalization alone moved a large share of the corpus to higher agreement
   (≥0.95 bucket 153,504→362,979 rows, +137%) — punctuation mismatch had been the dominant
   agreement suppressor, confirming the Issue #20 hypothesis.
2. `backfill_tier_thresholds_t16.py` — pure SQL `CASE` re-derivation of `tiers.tier`,
   scoped to `provenance = 'tier_assign'` only (5.6s). Verified: 0 human-gold rows lost
   `gold` status.

**Decision (owner confirmed via AskUserQuestion + follow-up research)**: replace the
confidence gate with `filters.dnsmos >= 3.5` (already in the catalog, zero new compute) as
the non-ASR third trust signal, per targeted research showing 2-model text agreement alone
is an insufficient auto-trust signal (GigaSpeech 2 / Emilia-Pipe-style pipelines layer
LID/DNSMOS on top of ASR agreement). Agreement bar lowered 0.95→**0.92** (owner picked the
"Balanced" bundle from 3 data-driven options presented, after seeing the full
agreement×dnsmos crosstab and an agreement×code-switch-status breakdown). `silver`
(≥0.85) / `bronze` (≥0.70) left unchanged — normalization alone already grew those pools
substantially without a threshold change.

**Result** (manifest-eligible pool, `filters.pass = TRUE`):

| Tier | Before (old gate, stale pre-normalization agreement) | After |
|---|---|---|
| gold | 58 (0.1h) | 58 (0.1h) — untouched |
| auto_gold | 73,252 (151.9h) | **279,195 (640.9h)**, +281% segments |
| silver | 255,941 | 158,087 (333.9h) |
| bronze | 261,159 | 169,435 (374.4h) |
| **Total manifest-eligible** | 590,410 (1,317.0h, 8,817 spk) | **606,775 (1,349.3h, 9,023 spk)** |

`manifest.export`/`report.build` re-run (default + `--min-tier auto_gold/silver/bronze`
cuts, each with its own `metadata/DATASET_REPORT_<tier>.md`).
`tests/test_catalog.py::test_manifest_build_matches_expected_corpus_totals`'s baseline
constants updated per its own docstring's "update only after an intentional, verified
manifest.export re-run" rule (458,843→606,775 etc.). `tests/test_tier_node.py` fully
rewritten for the new `assign_tier(text_verified, agreement, dnsmos=None)` signature
(13/13 passing).

**Provisional, not final**: T1 pilot QA (human ground-truth review) is still 0/~900
reviewed — this gate has NOT been precision-validated against human judgment. Owner
explicitly chose to unblock the larger auto_gold pool now rather than wait indefinitely
for T1; revisit 0.92/3.5 once real ground truth exists. See `pending_task.md` T16 (moved
to Done) and T1 (still open, highest priority).

**Follow-up spun out (not done, tracked as `pending_task.md` T18)**: the agreement×
code-switch breakdown run during this analysis found segments with `english_ratio > 0`
clear agreement thresholds far less often than pure-Cantonese ones (e.g. 18.8% vs 48.5%
at agreement≥0.90) — a systematic AR-vs-CTC English-transliteration divergence, not
necessarily a quality signal. Owner decided (AskUserQuestion) to keep one unified corpus
(code-switching is desired, not noise) but add an on-demand `--code-switch` export cut and
QA oversampling for code-switch segments — neither implemented yet; QA multiplier still
needs an owner decision (asked, no answer received this session).

## 2026-07-15 — T18: code-switch export cut + 10x QA oversampling (T16 follow-up)
**Decision (owner confirmed)**: multiplier = **10x**. `filters.english_ratio > 0` segments
get a QA sample rate of `QA_SAMPLE_RATE_BY_TIER[tier] * 10`, capped at 100%, on top of the
existing risk-scaled per-tier rates (auto_gold 1.5%→15%, silver 4%→40%, bronze 10%→100%).

**Implementation**:
- `pipeline/nodes/manifest.py`: `--code-switch {only|exclude}` cut (mirrors `--min-tier`)
  filtering on `filters.english_ratio > 0` / `= 0`, combinable with existing cuts, writes
  to separate `manifest_codeswitch_<mode>.jsonl` files, never touches the default export
  (hard constraint #9). `CODE_SWITCH_CONDITIONS` dict; `discover()`/`build_manifest()`/
  `run_manifest_build()`/`run_manifest_export()`/`_export_tag()` all threaded through.
- `pipeline/nodes/calibrate.py`: `CODE_SWITCH_QA_MULTIPLIER = 10.0`;
  `recommended_sample_n(conn, tier, code_switch=True)` scopes the population count to
  `english_ratio > 0` and multiplies the rate; `discover()`/`run_calibrate_sample()` gained
  a matching `code_switch` param for actually queuing a scoped batch.
- Both wired into the CLI (`--code-switch` on `manifest.build`/`manifest.export`/
  `calibrate.sample`) and the `run-many` adapter for `calibrate.sample`.

**Real exports produced** (full pool, no tier filter): `manifest_codeswitch_only.jsonl`
84,770 entries / 226.6h / 3,692 speakers; `manifest_codeswitch_exclude.jsonl` 522,005 /
1,122.7h / 8,728 speakers (sums to the 606,775-entry full pool from the T16 backfill).

**Recommended code-switch QA sample sizes** (not yet queued — left for the owner, since
it commits real human review time): auto_gold 1,250 (15% of 8,332 population), silver
10,366 (40% of 25,907), bronze **50,524 (100% — the 10x multiplier exactly saturates
bronze's 10% base rate)**. Bronze's "recommended" size is effectively "review the entire
bronze code-switch population," not realistically a near-term target — a smaller pilot
batch (e.g. 300, mirroring the 2026-07-11 tier-pilot precedent) is the practical next
step if/when the owner wants to start.

**Tests**: 16 new (8 `tests/test_manifest_node.py`, 8 `tests/test_calibrate_node.py`);
354/354 total passing.

## 2026-07-15 — calibrate.serve: 'rejected' now actually excludes (propagation
## fix) + one-click Mandarin flag button

**Bug found**: `record_decision()`'s `'rejected'` decision was recorded in
`calibration_review` but never read by anything downstream — `manifest.py`'s
eligibility join reads `segments`/`asr_agreement`/`g2p`/`filters`/`tiers`
only, never `calibration_review`. A human reviewer clicking "Reject" had zero
effect on what shipped in the manifest; only `'verified'` (→ `tiers.tier=
'gold'`) had a real side effect.

**Fix**: `record_decision()` now also directly upserts `tiers.tier=
'excluded'` (provenance `'calibrate_reject'`) when `decision == 'rejected'`
— same mechanism as the existing `'verified'` → `'gold'` write (sidesteps
`tier.assign`'s `provenance='tier_assign'`-scoped anti-join, which would
otherwise silently re-tier the row on a later `tier.assign` run). Applies to
every `'rejected'` decision, not just Mandarin-flagged ones.

**New: one-click "Mandarin" button** in `pipe calibrate serve` (`M` key),
alongside Verify/Skip/Reject/Flag. Submits `decision='rejected'` with a fixed
`flag_reason='mandarin'` (`MANDARIN_FLAG_REASON`, `pipeline/nodes/
calibrate.py`) — for segments that surface for text QA but turn out to be
non-HK-Cantonese content (CLAUDE.md hard constraint #1). One click both
excludes the segment (via the fix above) and records the reason for
`summary_stats()`'s `top_flag_reasons` triage leaderboard (query broadened
to include `'rejected'` rows with a reason, not just `'flagged'`).
`'flagged'` (generic pipeline-bug report) is unchanged — still does not
exclude, still free-text `flag_reason`, still distinct from `'rejected'`.

**Tests**: 3 new in `tests/test_calibrate_node.py` (rejected excludes tier,
rejected+mandarin-reason stores + excludes, summary_stats surfaces mandarin
rejections in the flag leaderboard). 357/357 total passing.

**Also this session**: deleted the two T16 backfill safety-net DB backups
(`corpus.duckdb.pre_agreement_t16_backup`, `.pre_tier_t16_backup`, 8.6GB
combined) — not git-tracked, T16 backfill already verified (0 gold rows
lost) and documented above; no further use.

## 2026-07-16 — calibrate.serve: tier/min-agreement/code-switch sample-options
## controls in the browser UI (T19 follow-up)

**Owner request**: the tier/min-agreement/code-switch scoping that
`pipe run calibrate.sample` already supports via CLI flags (`--tier`,
`--min-agreement`, `--code-switch`) had no equivalent in the browser UI —
`pipe calibrate serve`'s Refill button always queued an unscoped random
sample.

**Done**: a new "Sample:" control group in the topbar (tier dropdown,
min-agreement number input, code-switch dropdown), visually and functionally
distinct from the existing batch/source/order controls (those only filter
*browsing* of already-queued items; the new group scopes what a Refill
*queues* — the web equivalent of the CLI flags on `calibrate.sample`).
Applies to both the manual "↻ Refill" click and the auto-refill-on-empty-queue
path inside `/api/next` (so a focused review session, e.g. tier=auto_gold +
code-switch=only, doesn't get diluted by an unscoped top-up once the reviewer
runs the scoped queue dry). New `_parse_sample_options()` helper (shared by
`/api/refill`'s JSON body and `/api/next`'s query string) validates tier
against `_VALID_QA_TIERS` and min_agreement as a float, returning a JSON 400
on bad input rather than silently falling back to unscoped sampling.

**Verified**: live smoke test against a scratch catalog (separate DB, not the
production `corpus.duckdb`) — page renders the new controls, a scoped refill
(`tier=auto_gold&code_switch=only`) queues exactly the matching population,
and both an invalid tier and an invalid min_agreement correctly return a 400
with a JSON error body. No dedicated pytest suite exists for
`calibrate_server.py`'s HTTP layer (interactive tool, tested live per
CLAUDE.md's UI-testing guidance) — `tests/` stayed at 357/357 passing
throughout, unaffected by this change.

## 2026-07-16 — `upsert_rows()` performance fix: closed out, verified live at real scale (45×+)

Completed the last blocked steps of `docs/UPSERT_PERFORMANCE_FIX_PLAN.md` (started
2026-07-15, code+tests done that day but validation blocked on a live `speaker.cluster`
run holding the writer lock — see that file's prior status line). The lock freed
overnight; re-ran `pytest tests/ -q` clean at **357/357** (all 3 previously-blocked
catalog-touching files included, no lock conflict this time).

**Real-world validation**: ran `pipe run speaker.cluster` solo against the live catalog,
no `--limit` — full 3-source, 1,241,586-segment recompute. **104 seconds total**
wall-clock for all 3 sources' `upsert_rows()` writes combined (podcast 538,310 rows,
rthk 106,341, youtube 596,935), versus the historical **~78 minutes measured for the
podcast source's write alone** under the old per-row `conn.executemany()` path (see
`pending_task.md` T15 point 3 and the 2026-07-13 entry above). That's a **45×+ speedup**
on the write side of the node that originally motivated this fix (`speaker.cluster` was
the one node in the codebase that didn't chunk its `upsert_rows()` call — see T15's
"Future work worth doing" note). Correctness cross-checked: `speakers` table landed at
the identical row count (1,241,586) and distinct-speaker count (14,330) as the pre-fix
run — same clustering result, only the write mechanism changed.

**Consequence for T14**: this removes the single biggest reason `run-many` pairing of
`asr.transcribe` + `speaker.cluster` previously stalled (T15 points 3-5) — the
multi-minute-plus synchronous `executemany()` call blocking the shared asyncio event
loop for its whole duration is now a ~100ms-scale `INSERT ... SELECT` instead. Worth a
follow-up `run-many` pairing retry next time both nodes have real work queued, though
not re-tested this session (no `asr.transcribe` backlog currently pending).

No other node needed a change — `upsert_rows()` is a single shared helper in
`pipeline/catalog/catalog.py`, so every caller benefited transparently; the
`UPSERT_BULK_THRESHOLD = 2_000`-row gate means small/`--limit` runs are unaffected
(unchanged `executemany()` path below the threshold).

## 2026-07-16 — T11/T12: dormant-data relocation + automated log retention (owner-approved cleanup pass)

Two small hygiene items owner explicitly approved during a "go through pending tasks"
requirements review, done back-to-back:

**T11**: moved `metadata/manifest_release.jsonl` (672MB) + `excluded_no_url.jsonl`
(8.4MB) into `metadata/release_dormant/`, alongside the 3 dormant release scripts
already there. Pure relocation (`mv`, not `cp`) inside the gitignored `metadata/` tree —
zero git diff, zero risk difference from Hard Constraint #9's dormant-not-deleted
policy; grepped first to confirm nothing reads the old root-level path.

**T12**: `metadata/logs/` had no regrowth prevention after the one-time Phase B2
cleanup. New `pipeline/tools/prune_logs.py` (`pipe logs prune`, `--dry-run` supported):
gzips `*.log` files older than 7 days, deletes `*.log.gz` archives older than 60 days,
idempotent and mtime-based (so it catches both `logging.FileHandler` output and the
ad-hoc shell-redirected batch logs like `t15_*.log`, which a Python-side truncation
hook alone would have missed). 7 new tests (`tests/test_prune_logs.py`). Automated via
a **real weekly crontab entry** (`0 3 * * 0`, not Claude Code's session-scoped
`CronCreate` which would have expired in 7 days) — `crontab -l` confirmed clean before
adding, absolute paths used throughout since cron jobs don't inherit a shell `cwd`.
First live run: 46 files gzipped, 14.8MB reclaimed (`metadata/logs/` 70M → 56M).

## 2026-07-16 — T13: A/B TTS-quality tier built (`quality_tier.assign`), scoped to gold+auto_gold

Built `docs/LABEL_FRAMEWORK_SPEC.md` §10's A/B axis, which had sat undesigned since the
spec was written. Owner context: canto-tts training is about to start and pulled this
forward (was previously "timing pulled by training needs, not pipeline hygiene").

**Requirements gathering** (AskUserQuestion, since these were genuine judgment calls, not
inferrable from existing docs):
1. Scope: owner wants only the gold+auto_gold verification-confidence band (640.9h) fed to
   training, not the full manifest-eligible pool (1349.3h incl. silver/bronze) — so
   `quality_tier.assign` only tiers that scope; segments outside it never get a
   `quality_tiers` row.
2. Tier B (clean) threshold: presented 3 candidate bundles measured against the real
   gold+auto_gold distribution (dnsmos p50=3.64, music_prob p50=0.0525/p90=0.0959,
   overlap_ratio p50=p90=0.0 — most of this pool is already fairly clean since it's
   post-DNSMOS/agreement filtered): loose (dnsmos≥3.5, music<0.20, overlap<0.10 →
   251,434/575.9h), medium (dnsmos≥3.6, music<0.15, overlap<0.10 → 182,291/446.8h), strict
   (dnsmos≥3.7, music<0.10, overlap<0.05 → 55,596/152.1h). Owner picked **strict** — the
   tightest core subset, intended for the clean fine-tune stage specifically.

**Design**: new node `pipeline/nodes/quality_tier.py` (`quality_tier.assign`), new table
`quality_tiers (id, quality_tier, provenance)`. Tier A = every row in scope (the base
grade); Tier B = strict subset, gated on all three signals together, fails closed to A on
any missing signal (mirrors `tier.assign`'s auto_gold dnsmos gate). Explicitly documented
as a SEPARATE axis from `tiers`/`tier.assign` (verification-confidence) in both nodes'
module docstrings and CLAUDE.md's "Tier is overloaded" section, to prevent future
conflation — this ambiguity was flagged as a real risk in the original spec doc.

`manifest.build`/`manifest.export` gained `--min-quality-tier {A,B}` (LEFT JOIN against
`quality_tiers`, so silver/bronze/unscored segments stay included when the filter is
unused — this is NOT an INNER JOIN that would silently shrink the default export).
`QUALITY_TIER_PRECEDENCE = ("B", "A")` mirrors `TIER_PRECEDENCE`'s
best-to-worst/at-or-above pattern: `--min-quality-tier B` = strict subset only,
`--min-quality-tier A` = everything the node scored (A ∪ B, since B implies A rather than
being a disjoint bucket).

**Result**: full backfill against the live catalog — 279,185 segments quality-tiered in
**4 seconds** (thanks to the same-day upsert_rows() bulk-write fix above; this node would
have taken minutes under the old per-row `executemany()` path). A=223,605, B=55,580.
Exported `metadata/manifest_tier_auto_gold_qualityB.jsonl` (55,594 entries incl. 6 gold
rows / 152.1h / 1,860 speakers, train=55,498/val=96) for canto-tts's clean fine-tune stage;
the existing `manifest_tier_auto_gold.jsonl` (469.6MB, already exported 2026-07-15) already
serves as the Tier A / pretrain export, so no separate Tier A file was written.

**Tests**: 19 new in `tests/test_quality_tier_node.py` (pure-function boundary cases +
discover() scope/idempotency/anti-join + `conn=` injection regression), 11 new in
`tests/test_manifest_node.py` (`_quality_tiers_at_or_above`/`_export_tag`/`discover()`
integration incl. the LEFT-JOIN-not-INNER-JOIN behavior). Full suite green throughout.

## 2026-07-18 — T20: audio-based Mandarin gate wired into `filter.decide`; T21: low-agreement-first QA sampling order
**Trigger**: owner asked, while reviewing the T1 QA queue, why so many high-agreement
segments were being sampled and why Mandarin segments were showing up at all "given we
have a language filter." Investigation (not a pre-planned task) found a real gap.

**T20 finding**: two segment-level Mandarin gates existed, neither audio-based.
`lang_screen.auto` screens whole RAW FILES before diarization and deliberately lets
`mixed` files through (code-switched content shouldn't be thrown away wholesale).
`filter.text`'s `mandarin_ratio()` is a TEXT heuristic over the ASR transcript
(simplified-char / mainland-word-list scoring) — since HK ASR models emit standard
written Traditional Chinese, genuine spoken Mandarin transcribed fluently often scores
near 0 and clears the ≤0.15 threshold. Meanwhile `labels_lang` (mms-lid-126, computed by
`label.suite` directly from AUDIO — the same model `lang_screen.auto` uses) is a strong
per-segment signal but was never read by `filter.decide` or `tier.assign` at all — grepped
confirmed zero references outside `label_store`/`golden.py`/quality-tier's unrelated axis.
Live proof before the fix: 48 segments already sitting in the `calibrate.sample` QA queue
were labeled `lang='cmn'` at 92-99% confidence by `labels_lang` yet had already cleared
`filter.decide`.

**Decision (owner confirmed via AskUserQuestion — "加做硬性 filter (推薦)")**: wire
`labels_lang` into `filter.decide` as a hard gate — `lang='cmn' AND cmn_prob >= 0.8`
(`MANDARIN_AUDIO_PROB_MIN`, `pipeline/nodes/filter.py`) now fails a segment with
`fail_reason='mandarin_audio'`, checked last (after text/acoustic gates already passed),
never overriding an existing failure. 0.8 chosen as a conservative floor so a segment that
only briefly quotes a Mandarin speaker (the kind `lang_screen.auto`'s `mixed` band
intentionally preserves) isn't false-positive-rejected. New `filters` columns:
`lang_label_checked` (snapshots whether `labels_lang` had a row at decide time, so
discovery can re-trigger once the asynchronous `label.suite` node catches up on a segment
already decided without one — same versioned-re-evaluation shape as T5's `model_count`,
applied to a second independent staleness source) and `mandarin_audio_prob` (stored for
audit). `manifest.build`/`manifest.export` require no change — they already join on
`filters.pass = TRUE`, so a `mandarin_audio` fail is automatically manifest-excluded.

**T20 result — backfill run against the live catalog (`filter_decide_6af4cc9f708f`)**:
455,894 rows re-decided in 18s (the full `labels_lang`-covered population, since every
existing decided row had `lang_label_checked` unset). **10,940 currently-passing segments
flipped to `pass=FALSE, fail_reason='mandarin_audio'`** (~1.4% of the 780,219 then-passing
pool). `filters.pass=TRUE` count: 780,219 → 769,279. `catalog verify`: 17/17 PASS
afterward. 44 of those 10,940 were already sitting in the pending QA queue (2,792 total,
now effectively redundant to review since they're already manifest-excluded regardless of
the human decision — left in the queue rather than pruned, since a human confirming the
audio classifier's judgment is still useful QA signal for T1). This gate only fires going
forward for segments `labels_lang` hasn't reached yet (label.suite coverage lags total
segment count: 455,894 / 1,241,610 at the time of this run) — `manifest.export` should be
re-run before the next training data pull to pick up the 10,940 exclusions (not done
automatically as part of this fix).

**T21 (companion finding, same investigation)**: `calibrate.sample`'s
`SAMPLE_DISCOVER_SQL` always sampled uniformly at random (`ORDER BY random()`) within
whatever tier/min-agreement/code-switch population was scoped — so an `auto_gold`-scoped
batch skewed to agreement~0.95-1.0 simply because that's where most of the tier's mass
sits, with no way to deliberately pull the segments closest to a tier's own agreement
floor (the ones the gate trusted least). Note: `next_pending()`'s existing browsing
`order` param (`agreement_asc`/`agreement_desc`) already let a reviewer re-sort items
*already queued* — this is a distinct, earlier-stage control over which segments get
*sampled into* the queue in the first place.

**Decision (owner asked to check-then-implement, given the existing UI precedent)**: added
`order_by` (`'random'` default / `'agreement_asc'`) to `calibrate.sample`'s
`discover()`/`run_calibrate_sample()`, `pipe run calibrate.sample --order`, and a new
"Sample:" panel dropdown (`sampleOrderSelect`) in `pipe calibrate serve`, wired through
both the manual Refill button and the auto-refill-on-empty path — mirrors the existing
tier/min-agreement/code-switch scoping controls exactly. Composable with all of them (e.g.
`--tier bronze --code-switch only --order agreement_asc` concentrates a batch on the
riskiest code-switch segments within bronze specifically).

**Tests**: 9 new in `tests/test_filter_node.py` (T20: `decide_row`'s audio-gate boundary
cases + `lang_label_checked` storage + `discover_decide`'s three-state re-trigger:
never-decided / no-label-yet / label-landed-after-decision), 5 new in
`tests/test_calibrate_node.py` (T21: `order_by` ordering, invalid-value rejection,
composition with tier/code_switch, default-random regression guard). 453/453 full suite
green before the production run; `catalog verify` 17/17 PASS after it.
