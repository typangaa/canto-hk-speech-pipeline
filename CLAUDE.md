# canto-hk-speech-pipeline — HK Cantonese TTS Dataset

## Project in One Sentence

Collect, clean, and annotate **100–500 hours** of **Hong Kong Cantonese** speech from multiple public sources to build a self-owned, high-quality TTS training corpus with **100+ unique speakers**.

> This file describes the **current** system only. All history — model retirements, gate
> rebuilds, threshold changes, one-time migrations — lives in **`DECISIONS.md`**; failure
> modes from prior attempts live in **`docs/KNOWN_ISSUES.md`**. Consult those, don't
> re-document history here.

---

## Session Start Protocol

Every session, in this order before writing any code:

```bash
cat PROGRESS.md                                        # 1. last session's log
cat pending_task.md                                    # 2. task backlog
df -h /mnt/Drive2/ /mnt/Drive3/ /mnt/Drive4/           # 3. disk (3-way segment shard, all active)
nvidia-smi --query-gpu=name,memory.free --format=csv   # 4. GPU availability
ls pipeline/nodes/ && python -m pipeline.cli run --help  # 5. available DAG nodes
python -m pipeline.cli catalog verify                  # 6. catalog state (DuckDB = source of truth)
```

Update `PROGRESS.md` at the end of every session (format at the bottom of this file).

**`pending_task.md`** (repo root, git-tracked) is the living task backlog — tiered
🔴 data-trust-critical → 🟠 functional gaps → 🟡 engineering cleanup → 🟢 optional.
**When a task completes, move it into the file's "Done" section (with date + commit) in
the same session** — don't just delete it; the file is the record of fixed vs. open.
**Done section rotation**: once it holds more than ~2 weeks of entries, move the older
ones verbatim into a new dated `DECISIONS.md` section (nothing lost, just relocated) so
the file stays to the recent working window — same pattern `PROGRESS.md` uses.

If touching pipeline internals, also check the living status docs:
`docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md` §8 (milestones) and `docs/ORCHESTRATOR_PLAN.md`.

---

## Directory Layout

The pipeline is the **`pipeline/` Python package** — a catalog-driven DAG orchestrator.
(The original flat `scripts/01_*.py…` chain is fully retired; `scripts/` is empty,
recoverable from git history.)

```
canto-hk-speech-pipeline/
├── CLAUDE.md                ← you are here (read every session)
├── PROGRESS.md              ← session log (gitignored; read first, update last)
├── pending_task.md          ← tiered task backlog (git-tracked)
├── DECISIONS.md             ← technical decision log — all project history lives here
│
├── docs/
│   ├── REARCHITECTURE_IMPLEMENTATION_PLAN.md  ← milestone status (§8) — most authoritative
│   ├── ORCHESTRATOR_PLAN.md                   ← `pipe run-many` concurrency design + status
│   ├── LABEL_FRAMEWORK_SPEC.md                ← label tables + the A/B TTS-quality tier axis
│   ├── QUALITY_SPEC.md                        ← filtering thresholds and rationale
│   ├── MANIFEST_SCHEMA.md                     ← manifest field definitions + examples
│   ├── KNOWN_ISSUES.md                        ← all known failure modes from prior attempts
│   ├── SOURCE_GUIDE.md                        ← how to research and add new sources
│   ├── (others: PIPELINE_SPEC, G2P_MIGRATION_NOTE, IO_OPTIMIZATION_PLAN …)
│   └── archive/                                ← executed/superseded plan+review docs — historical, not maintained
│
├── sources/                 ← rthk_sources.yaml / youtube_channels.yaml / podcast_sources.yaml
├── config/
│   ├── pipeline.yaml        ← catalog path, golden-set, manifest/label export paths
│   └── storage_layout.yaml  ← SSOT for every data path + the 3-way segment shard map
│
├── pipeline/                ← THE SYSTEM (write nodes here)
│   ├── cli.py               ← `python -m pipeline.cli {catalog|golden|run|run-many|calibrate}`
│   ├── config.py            ← typed accessor for config/pipeline.yaml
│   ├── catalog/             ← DuckDB connect/upsert/verify
│   ├── audio/               ← decode-once bus + LRU resampled-variant cache
│   ├── orchestrator/        ← resource pools (gpu.N/cpu/io.DriveN), foreign-GPU yield sampler, run journal
│   ├── workers/             ← GPU worker-subprocess base class (JSONL-over-stdio)
│   ├── tools/               ← calibrate_server.py (human-review browser UI)
│   └── nodes/               ← one file per DAG stage (table below)
│
└── metadata/
    ├── corpus.duckdb        ← THE catalog — single source of truth for every stage's state
    ├── logs/                ← per-node logs ({node_name}.log)
    ├── manifest.jsonl / train.jsonl / val.jsonl / labels.jsonl
    └── DATASET_REPORT.md    ← regenerated live by `pipe run report.build`
```

Data on disk: **raw** → `/mnt/Drive2/canto-corpus/data/raw/{rthk,youtube,podcast}/` (native
container); **segments** → 3-way sharded `/mnt/Drive{2,3,4}/canto/segments/` (see Hard
Constraint 4). The repo-local `data/` directory is not the SSOT — `config/storage_layout.yaml` is.

---

## Subtask delegation(agy via weir)

Subtask please use **agy through weir**:

```bash
weir -c ~/.config/weir/weir.toml chat agy-gemini "<prompt>"   # online search
weir -c ~/.config/weir/weir.toml chat agy-sonnet "<prompt>"   # coding tasks
```
agy report may appear in `~/.gemini/antigravity-cli/brain/<uuid>/<name>.md`

## Pipeline Architecture

The pipeline is a **catalog-driven DAG**, not a linear script chain. Every node is a
`pipe run <node.name>` CLI call; every node's input/output is a table in
**`metadata/corpus.duckdb`** — no node reads another node's sidecar files directly. Every
node is idempotent: discovery is a SQL anti-join against already-processed rows, so a node
can be killed and re-run without redoing work or duplicating output.

| Node | Reads | Writes | One-line detail |
|------|-------|--------|-----------------|
| `ingest.download` | `sources/*.yaml` | staging JSON | yt-dlp/RSS, native container, zero transcode; holds no DB writer |
| `ingest.commit` | staging JSON | `raw_files` | only `ingest.*` step that opens the DuckDB writer |
| `ingest.probe` | `raw_files` | `raw_probe` | ffprobe metadata + L/R stereo correlation |
| `lang_screen.auto` | `raw_files` | `labels_lang` | raw-level Mandarin-vs-Cantonese pre-filter, runs **before** diarize |
| `segment.diarize` | `raw_files` | `diarization_turns`, `raw_segments` | pyannote 3.1, sidecar-reuse-first |
| `segment.vad_cut` | `diarization_turns` | `segments` | Silero VAD within each turn → single-speaker 3–20s 48kHz FLAC |
| `pregate.snr` | `segments` | `pregate` | fast SNR/DNSMOS pre-gate before spending ASR GPU time |
| `asr.transcribe` | `segments` | `asr_results` | 2 active models (`qwen3_asr`, `sense_voice`) — see ASR strategy below |
| `asr.agreement` | `asr_results` | `asr_agreement` | cross-model char-overlap (punctuation/digit-normalized), active models only |
| `filter.text` | `asr_agreement` | `filters_text` | length/English-ratio/Mandarin-ratio gates — no audio decode |
| `filter.acoustic` | `filters_text` (pass only) | `filters_acoustic` | SNR + DNSMOS — CPU worker-subprocess pool |
| `filter.decide` | both `filters_*` | `filters` | merges into final `pass`/`fail_reason` |
| `g2p` | `asr_agreement` (verified text) | `g2p` | canto-hk-g2p → Jyutping, regex-validated `^[a-z]+[1-6]$` |
| `speaker.embed` | `segments` | `speaker_embeddings` | ECAPA-TDNN d-vector, sidecar-`.npy`-reuse-first |
| `speaker.cluster` | `speaker_embeddings` | `speakers` | cross-file agglomerative clustering, whole-source recompute |
| `tier.assign` | `asr_agreement`+`filters` | `tiers` | verification-confidence tier — see "Two tier axes" below |
| `quality_tier.assign` | `tiers`+`filters`+`labels_*` | `quality_tiers` | A/B acoustic-cleanliness tier, gold+auto_gold scope only — see below |
| `calibrate.sample` | `asr_agreement`+`filters`+`tiers` | `calibration_review` | queues a random sample for human review — see "Human calibration" below |
| `manifest.build` / `.export` | `filters`+`g2p`+`speakers`+`tiers` | `metadata/*.jsonl` | final JSONL + 95/5 split; `--min-tier` / `--min-quality-tier` / `--min-agreement` |
| `report.build` | same join as `manifest.build` | `metadata/DATASET_REPORT.md` | live acceptance-criteria report — see Acceptance Criteria |
| `label.suite` / `.music` / `.prosody` | `segments`/`raw_files` | `labels_*` | decode-once fan-out for the TTS-quality label store |
| `label.calibrate` / `label.store` | label tables | `metadata/labels*.jsonl` | rate/pitch calibration + bucketed label export |
| `recover.orphans` / `rebalance.segments` / `raw.flac` | — | — | one-time utilities, all done — see DECISIONS.md |

Milestone status: **P0–P5 done, P6 (scale readiness) not started** —
`docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md` §8 is authoritative.
For per-node detail beyond one line, read the node file's module docstring.

**Concurrency — `pipe run-many`**: DuckDB's write lock is per-*process*, so two separate
`pipe run X` invocations can never hold the writer at once. `pipe run-many <node> -- <node>`
opens **one** shared connection and gives each node its own `conn.cursor()`, running them
concurrently via `asyncio.gather` (e.g. a GPU node beside a CPU node). All node call sites
accept `conn=` injection. Two measured caveats: (1) don't pair a node with a very large
discovery anti-join or a very large single `upsert_rows()` (e.g. `speaker.cluster`'s 500k-row
`executemany`) with anything else — the long synchronous call starves the sibling; (2) never
interleave two *different* GPU models on the same device — the per-device `target=1`
semaphore is not fair-shared and one model starves completely. Run those
sequentially-exclusive. A background sampler polls `nvidia-smi` and yields GPU pool targets
when a foreign process is using a GPU.

**Legacy-row-collision pattern**: several tables (`filters`, `g2p`, `tiers`, `raw_segments`)
were pre-populated by the one-time P0 import of the old manifest (455,299 rows), so a bare
"does a row exist" anti-join finds zero work forever. Every node must tag its writes with its
own `provenance` value (e.g. `'filter_decide'`) and anti-join on that value, not row existence.

**Audio rate strategy**: store every segment as a **48 kHz mono lossless master** —
downsampling is irreversible, and every modern TTS codec needs ≥24 kHz. VAD / diarization /
ASR / DNSMOS need 16 kHz internally: generate a *transient* 16 kHz copy and discard it.

**Raw storage policy**: `ingest.download` keeps the native-container bestaudio format with
zero transcoding. (The old raw-WAV backlog was losslessly converted to FLAC by `raw.flac`.)

**ASR strategy**: two active backends, run per segment —
- `qwen3_asr` (Qwen3-ASR-1.7B, transformers, `language="Cantonese"`; not Whisper-derived) — **always pass `--batch 64`** (the CLI default of 8 costs ~2.4× throughput).
- `sense_voice` (SenseVoice-Small, funasr, CTC non-autoregressive, `language="yue"`, ~105× RTF; emits Simplified → converted to Traditional HK via OpenCC s2hk; emotion/audio-event tags kept in `asr_results.metadata`, stripped from `text`).

Run each model **sequentially-exclusive across both GPUs** (caveat 2 above). Two
Whisper-family models (`whisper_v3`, `canto_ft`) are retired for accuracy (17–36% CER vs
qwen3's ~0.4% on human-verified samples — DECISIONS.md); their historical `asr_results` rows
are kept but never dispatched or read. `asr.agreement` computes over active models only — an
id becomes eligible once ≥2 active models have landed; a late straggler re-triggers
recompute. **Never auto-trust a single ASR output. Never `language="yue"` on any
Whisper-family model** (decoder collapse, `docs/KNOWN_ISSUES.md` §9 — Qwen3-ASR/SenseVoice
are unaffected). Known limitation: `filter.text`/`filter.decide`/`tier.assign` discovery is
bare row-existence, so segments filtered/tiered before a later model improved their
agreement are **not** automatically re-evaluated.

**Human calibration**: `calibrate.sample` queues segments; `pipe calibrate serve` is the
local browser review UI. A `'verified'` decision flips `asr_agreement.text_verified` +
`tiers.tier='gold'`; a `'rejected'` decision flips `tiers.tier='excluded'` (removes the
segment from manifest eligibility). The **Mandarin** button (`M` key) is a one-click
`rejected` + `flag_reason='mandarin'`. **Offline mode**: the UI never blocks on the catalog —
decisions buffer to `metadata/calibration_pending_decisions.jsonl`; reads fall back to a JSON
snapshot (`pipe calibrate export-snapshot`) when the writer lock is held. Run
`pipe calibrate flush-pending` once the writer is free (safe to re-run). **Never bulk-delete
`pending` review rows without flushing the buffer first** — `pipe calibrate prune-excluded`
(also the UI's 🧹 button) does this safely: flush, then delete pending rows whose segment a
later gate already excluded.

**Two tier axes — ⚠️ same word, different things. Never conflate them:**
- **`tiers` / `tier.assign` = verification confidence**: `gold` (human-verified via
  calibrate serve) · `auto_gold` (statistical only, **not** human-verified: agreement ≥ 0.92
  AND `filters.dnsmos` ≥ 3.5, sample-QA'd) · `silver` (≥ 0.85) · `bronze` (≥ 0.70, lowest
  manifest-eligible, highest QA sample rate) · `excluded` (< 0.70, or human-rejected/gate-excluded).
- **`quality_tiers` / `quality_tier.assign` = A/B acoustic cleanliness**
  (`docs/LABEL_FRAMEWORK_SPEC.md` §10, scoped to gold+auto_gold only): Tier A (pretrain) =
  entire scope; Tier B (clean fine-tune) = `dnsmos≥3.7 AND music_prob<0.10 AND overlap_ratio<0.05`.

Threshold history/rationale: DECISIONS.md + `docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md`.

---

## Data Sources

| Source | Config file | Priority | Est. hours available |
|--------|-------------|----------|----------------------|
| RTHK (expansion) | `sources/rthk_sources.yaml` | High | 200–500h |
| YouTube HK Cantonese | `sources/youtube_channels.yaml` | High | varies |
| HK Cantonese Podcasts | `sources/podcast_sources.yaml` | Medium | 50–200h |
| Other HK TV | `sources/hktv_sources.yaml` (not yet created) | Low | TBD |

Source research guide: `docs/SOURCE_GUIDE.md`. RTHK publishes on YouTube
(`@rthkhongkong` + sub-channels) — target programs with multiple presenters and diverse
styles. (The legacy-corpus re-ingest is long resolved — filename lists archived at
`metadata/legacy_filenames/*.txt`; see DECISIONS.md.)

---

## Hard Constraints

These are non-negotiable. Never override for any technical reason.

1. **HK Cantonese only.** Reject audio that is primarily Mandarin, Guangzhou Cantonese, or overseas Cantonese. Use language detection to filter.

2. **No WenetSpeech-Yue or third-party datasets.** This corpus is self-owned and self-collected. Only publicly accessible web sources downloaded directly.

3. **No code from prior project directories.** Do not `cp` or `import` from:
   ```
   ../cantonese-tts/        ../cantonese-tts-old/    ../CantoNeu/
   ../gemini-hk-canto-tts/  ../gemma-hermes--tts/    ../qwen36-hermes-tts/
   ```
   Their **lessons** are encoded in `docs/KNOWN_ISSUES.md` — read that instead.

4. **Absolute Linux paths only.** Every `audio_path` must be absolute: **raw** on `/mnt/Drive2/canto-corpus/data/raw/`; **segments** 3-way sharded across `/mnt/Drive{2,3,4}/canto/segments/` via `config/storage_layout.py:shard_index()` (deterministic `hash % 3`) — **never hand-pick a shard, always call that function**. No relative paths, no Windows-style paths (`/mnt/d/`, `/mnt/c/Users/`).

5. **Single-speaker, VAD-based segmentation only.** Never hard-cut at a fixed duration. Cut at Silero-VAD pause boundaries *within* diarization-detected single-speaker turns, so no clip spans a speaker change. Target 3–20 s. See `docs/KNOWN_ISSUES.md` §2, §10.

6. **48 kHz mono lossless master, never lower, never lossy.** WAV or FLAC only — never Opus/MP3/any lossy codec as a segment master. Downsampling is irreversible. Transient 16 kHz copies for VAD/ASR/DNSMOS only. New segments are FLAC; legacy WAV segments are not re-encoded. See `docs/KNOWN_ISSUES.md` §11.

7. **Never `language="yue"` in Whisper.** Decoder collapse on large-v3. Use a Cantonese fine-tuned model and/or `language="zh"` + written-Cantonese prompt. Multiple ASR models; a human calibrates the canonical text. See `docs/KNOWN_ISSUES.md` §9.

8. **Validate Jyutping format explicitly.** Every space-separated token must match `^[a-z]+[1-6]$`; reject segments where > 5% of tokens fail. Run G2P on **human-verified** text, never raw single-ASR output. See `docs/KNOWN_ISSUES.md` §1.

9. **NEVER publish the dataset — zero-risk policy.** The corpus is private, for training the owner's own canto-tts model only. Do not publish, upload, or share — anywhere — the dataset/manifests, raw or filtered audio, per-segment `source_url`s, or the reconstruction recipe (`reconstruct.py`, `metadata/manifest_release.jsonl`, `metadata/excluded_no_url.jsonl`). A trained **model** may be released later (weights only, never data); the pipeline *code* stays open source. Release/reconstruction scripts are dormant and must never be acted on without explicit owner approval. See DECISIONS.md 2026-06-29.

---

## Critical Known Issues (Summary)

Full details in `docs/KNOWN_ISSUES.md`. These will silently corrupt your data if ignored:

**Issue 1 — G2P tool history**
`pycantonese` ≥ 4.1.0 concatenates Jyutping syllables without spaces in ~47% of cases — do not use.
`ToJyutping` handles English letter-by-letter (`[S][E][N][D]`), breaking TTS alignment.
→ Use **canto-hk-g2p** (`github.com/typangaa/canto-hk-g2p`): Rust-core, passes English through unchanged, expands numbers/dates. Regex validation still mandatory: `^[a-z]+[1-6]$` per token.

**Issue 2 — Hard audio cut at fixed duration**
Prior pipeline cut at exactly 15 s → 44.5% of segments mid-sentence.
→ Silero VAD. Never cut with `ffmpeg -t`. Verify cuts fall at silence boundaries.

**Issue 3 — Windows absolute paths in manifests**
Old manifests referenced `/mnt/d/`, `/mnt/c/Users/…` — broken on Linux. Resolved for current
data; the ongoing concern is any new node writing `segments.audio_path` bypassing
`config/storage_layout.py:shard_root()`. `grep -c "/mnt/d/" metadata/manifest.jsonl` must return 0.

**Issue 9 — Whisper `yue` decoder collapse**
`language="yue"` on large-v3 → repetition loops / garbage. → See Hard Constraint 7.

**Issue 11 — 16 kHz dead-end**
A 16 kHz corpus cannot train any modern TTS codec (all need ≥24 kHz). → 48 kHz master; downsample only transiently.

---

## Pipeline Node Conventions (`pipeline/nodes/*.py`)

Follow these for any new or edited DAG node:

```python
"""
pipeline/nodes/example.py
example.thing DAG node -- one-line description of what it discovers and writes.
"""

async def run_example_thing(*, conn=None, batch_size: int = 5000, limit: int | None = None) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) — pass one when running
    alongside other nodes under `pipe run-many`. Defaults to a fresh self-managed
    connect() for standalone `pipe run example.thing` usage."""
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()                       # conn injection: never call connect() if one was passed
    rows = discover(conn)                          # SQL anti-join on your OWN provenance tag, not bare row-existence
    if limit:
        rows = rows[:limit]
    run_id = new_run_id("example.thing")
    for batch in chunks(rows, batch_size):
        out = [{"id": r.id, ..., "provenance": "example_thing"} for r in batch]
        upsert_rows(conn, "your_table", out, ["id"])  # INSERT OR REPLACE — never partial-write a shared column set
        record_batch(conn, run_id, "example.thing", [r["id"] for r in out], "ok")
    return {"processed": len(rows), "errors": 0, "run_id": run_id}
```

Checklist:
- **`conn=None` injection**: every `run_*()` takes `conn=None`; body does `conn = conn or connect()`. Register in `RUN_MANY_ADAPTERS` in `pipeline/cli.py`, and add a `tests/test_run_many.py` regression test asserting `connect()` is never called when `conn` is passed.
- **Idempotent discovery via anti-join on `provenance`**, not bare row-existence (see Legacy-row-collision pattern above).
- **Register the CLI subcommand** in `pipeline/cli.py` — verify the exact registered name (e.g. `lang_screen.auto`, not `lang.screen`) before referencing it elsewhere.
- **Logging**: `metadata/logs/{node_name}.log` via the standard `logging.basicConfig` pattern.
- **Never partial-write a shared table** — if two nodes produce parts of one table's columns, keep separate tables and let exactly one downstream node merge them; `upsert_rows()` is `INSERT OR REPLACE`, so partial writes clobber each other.
- **Test with `--limit N` against the real catalog before a full backlog run** — this is how most real bugs here were actually caught, not by unit tests alone.

**Audio I/O**: 48 kHz mono lossless — FLAC for new segments, legacy 16-bit PCM WAV not
re-encoded. Use `soundfile` (reads both); `pipeline/audio/bus.py` falls back to an ffmpeg
pipe for containers `libsndfile` can't open (`.webm`/`.m4a`). Never overwrite the 48 kHz
master with a downsampled or lossy-recompressed version.

**File naming**:
```
raw:      {YYYYMMDD}_{program_slug}_{video_id}.{ext}             (native bestaudio container)
segments: {YYYYMMDD}_{program_slug}_{video_id}_seg{N:05d}.flac   (new) / .wav (legacy)
```
"Filtered" is not a physical copy — `filters.pass` in the catalog is the authoritative
signal; `segments.audio_path` is read directly wherever it lives on the shard.

**Never print sensitive info** (API keys, full file paths in error messages visible to log aggregators).

---

## Quality Thresholds

| Filter | Threshold | Notes |
|--------|-----------|-------|
| Duration | 3–20 seconds | VAD-based, not hard cut |
| DNSMOS P.835 (OVRL) | ≥ 3.0 | `speechmos` → `dnsmos.run(wav16k, sr=16000)['ovrl_mos']`; verify in [1.0, 5.0] |
| SNR | ≥ 25 dB | |
| ASR agreement | ≥ 0.80 char-overlap | low agreement → flag for manual calibration |
| Mandarin ratio | ≤ 0.15 | fraction of Mandarin-only words |
| English ratio | ≤ 0.30 | allow code-switching, reject English-dominant |
| Min characters | ≥ 5 | after punctuation removal |
| Max characters | ≤ 150 | guard against ASR hallucination |
| Single speaker | required | diarization must report exactly 1 speaker per segment |

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
    {"model": "...", "text": "...", "confidence": 0.88}
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
Tier semantics: see "Two tier axes" in Pipeline Architecture above.

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

Check with `python -m pipeline.cli run report.build` — computes fresh from the catalog and
writes `metadata/DATASET_REPORT.md` with PASS/FAIL per criterion (single-speaker is a
pipeline-design guarantee, reported separately — see `pipeline/nodes/report.py`).
`--min-tier gold` scopes the check to the human-verified subset.

**CHECKPOINT**: when all criteria pass, stop and present `DATASET_REPORT.md` for human
review before the corpus is used for TTS model training.

---

## External Dependencies

| Tool | Source | Purpose |
|------|--------|---------|
| DuckDB | PyPI | the catalog — single source of truth for all node state |
| canto-hk-g2p | github.com/typangaa/canto-hk-g2p | `g2p` node (Rust+PyO3) |
| qwen-asr | PyPI | `asr.transcribe` (qwen3_asr) |
| funasr + modelscope | PyPI (non-OSI model license, commercial use permitted) | `asr.transcribe` (sense_voice) |
| opencc-python-reimplemented | PyPI | sense_voice Simplified→Traditional HK (s2hk) |
| pyannote.audio | PyPI + HF model terms | `segment.diarize` |
| speechbrain | PyPI | `speaker.embed` (ECAPA-TDNN) |
| speechmos | PyPI | DNSMOS gate (`pregate.snr` / `filter.acoustic`) |
| yt-dlp | PyPI / system | `ingest.download` |
| faster-whisper | PyPI | unused at runtime (retired models); kept so historical imports work |

⚠️ **Env management: `uv pip install <pkg>`, never `uv sync`** — the `.venv` has GPU torch
installed outside lock tracking; `uv sync` prunes the CUDA libs and breaks torch.

---

## PROGRESS.md Update Format

```markdown
## Session YYYY-MM-DD
**Stage**: [e.g. ASR backlog — youtube source]
**Completed**:
- specific item
**Stats**: X h downloaded, Y h segmented, Z h passed QC, N speakers identified
**Next**:
- specific next action
**Blockers** (human input needed):
- item, or "none"
```
