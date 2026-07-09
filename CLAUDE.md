# canto-hk-speech-pipeline — HK Cantonese TTS Dataset

## Project in One Sentence

Collect, clean, and annotate **100–500 hours** of **Hong Kong Cantonese** speech from multiple public sources to build a self-owned, high-quality TTS training corpus with **100+ unique speakers**.

---

## Session Start Protocol

Every session, in this order before writing any code:

```bash
# 1. Read progress from last session
cat PROGRESS.md

# 2. Check disk space (corpus is 3-way sharded across warm ext4 drives — Drive2/3/4 all in active use since P5-C)
df -h /mnt/Drive2/ /mnt/Drive3/ /mnt/Drive4/

# 3. Check GPU availability (needed for diarization, ASR, speaker embedding, lang screen, music tagging)
nvidia-smi --query-gpu=name,memory.free --format=csv

# 4. Check what DAG nodes already exist (current system — see "Pipeline Architecture" below)
ls pipeline/nodes/
python -m pipeline.cli run --help

# 5. Check catalog state (DuckDB is the source of truth, not per-file sidecar JSON)
python -m pipeline.cli catalog verify
```

Update `PROGRESS.md` at the end of every session using the format at the bottom of this file.
Also check `docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md` §8 (milestone status) and
`docs/ORCHESTRATOR_PLAN.md` (status line at top) if you're touching pipeline internals —
both are living status docs, updated more frequently than this file.

---

## Directory Layout

**As of the 2026-07-02 re-architecture, the pipeline runs as a `pipeline/` Python package —
a catalog-driven DAG orchestrator — not the flat `scripts/01_*.py … 10_*.py` chain this
project started with.** The old scripts are being retired one-by-one as each stage is
ported and golden-set-parity-tested; a few (`04_transcribe.py`, `06_filter.py`, `07_g2p.py`,
`08_speaker_id.py`, `09_manifest.py`, `10_report.py`) still exist on disk purely as
pre-migration reference/fallback and should be treated as **legacy, not where new work goes**.
See "Pipeline Architecture (Current)" below for the real system.

```
canto-hk-speech-pipeline/
├── CLAUDE.md                     ← you are here (read every session)
├── PROGRESS.md                   ← session log (read first, update last)
├── DECISIONS.md                  ← technical decision log
│
├── docs/
│   ├── REARCHITECTURE_IMPLEMENTATION_PLAN.md  ← DAG node list, milestone (P0–P6) status — most authoritative
│   ├── PIPELINE_REARCHITECTURE_PLAN.md        ← original re-architecture vision/rationale
│   ├── ORCHESTRATOR_PLAN.md                   ← `pipe run-many` concurrency design + live status line
│   ├── JOURNAL_FIRST_PLAN.md                  ← run_id / batch provenance journal design
│   ├── LABEL_FRAMEWORK_SPEC.md                ← label tables + the separate A/B TTS-quality tier axis
│   ├── PIPELINE_SPEC.md                       ← legacy full pipeline implementation details
│   ├── QUALITY_SPEC.md                        ← filtering thresholds and rationale
│   ├── MANIFEST_SCHEMA.md                     ← manifest field definitions + examples
│   ├── KNOWN_ISSUES.md                        ← all known failure modes from prior attempts
│   ├── G2P_MIGRATION_NOTE.md                  ← canto-hk-g2p PyPI-wrapper migration note
│   └── SOURCE_GUIDE.md                        ← how to research and add new sources
│
├── sources/
│   ├── rthk_sources.yaml       ← RTHK programs to download
│   ├── youtube_channels.yaml   ← YouTube channels to download
│   └── podcast_sources.yaml    ← Podcast RSS feeds
│
├── config/
│   ├── pipeline.yaml            ← catalog path, golden-set, manifest/label export paths
│   └── storage_layout.yaml      ← SSOT for every data path + the 3-way segment shard map
│
├── pipeline/                    ← THE CURRENT SYSTEM — catalog-driven DAG (you write nodes here)
│   ├── cli.py                   ← `python -m pipeline.cli {catalog|golden|run|run-many}`
│   ├── config.py                ← typed accessor for config/pipeline.yaml
│   ├── catalog/                 ← DuckDB connect/upsert/verify + legacy-jsonl import
│   ├── audio/                   ← decode-once bus + LRU resampled-variant cache
│   ├── orchestrator/             ← resource pools (gpu.N/cpu/io.DriveN), foreign-GPU yield sampler, run journal
│   ├── workers/                  ← GPU worker-subprocess base class (JSONL-over-stdio protocol)
│   └── nodes/                    ← one file per DAG stage — ingest_download, ingest_probe, lang_screen,
│                                    segment, asr, filter, g2p, speaker, tier, label_*, manifest,
│                                    recover_orphans, rebalance, raw_flac (see table below)
│
├── scripts/                      ← LEGACY, being retired — do not extend; port to pipeline/nodes/ instead
│
├── data/                         ← legacy per-source symlink tree; real SSOT is config/storage_layout.yaml
│   ├── raw/       → /mnt/Drive2/canto-corpus/data/raw/{rthk,youtube,podcast}/  (native container, transient)
│   ├── segments/  → per-source symlinks (pre-P5-C; segments are now 3-way sharded, see below)
│   └── filtered/  → /mnt/Drive4/canto/filtered/  (physicalization retired per §7.3, catalog `filters.pass` is authoritative)
│
└── metadata/
    ├── corpus.duckdb            ← THE catalog — single source of truth for every pipeline stage's state
    ├── logs/                    ← per-node log files (`{node_name}.log`)
    ├── manifest.jsonl           ← full manifest (all splits) — written by `manifest.export`
    ├── train.jsonl              ← training split (95%)
    ├── val.jsonl                ← validation split (5%)
    ├── labels.jsonl             ← unified label store (label.store node)
    └── DATASET_REPORT.md        ← human-readable summary (final output; report node not yet ported — see Acceptance Criteria)
```

---

## Pipeline Architecture (Current)

The pipeline is a **catalog-driven DAG**, not a linear script chain. Every node is a
`pipe run <node.name>` CLI call (`python -m pipeline.cli run <node.name>`); every node's
input/output is a table in **`metadata/corpus.duckdb`** — no node reads another node's
sidecar files directly. Every node is idempotent: discovery is a SQL anti-join against
already-processed rows, so a node can be killed and re-run without redoing work or
duplicating output.

| Node | Reads (catalog table) | Writes (catalog table) | Detail |
|------|------------------------|-------------------------|--------|
| `ingest.download` | `sources/*.yaml` | JSON staging file (no DB writer held) | yt-dlp/RSS, **native container, zero transcode** (2026-07-04 policy) |
| `ingest.commit` | staging JSON | `raw_files` | only `ingest.*` step that opens the DuckDB writer |
| `ingest.probe` | `raw_files` | `raw_probe` | ffprobe metadata + L/R stereo correlation |
| `lang_screen.auto` | `raw_files` | `labels_lang` (+ reject decision) | raw-level Mandarin-vs-Cantonese pre-filter, **runs before** `segment.diarize` |
| `segment.diarize` | `raw_files` | `diarization_turns`, `raw_segments` | pyannote 3.1, sidecar-reuse-first, GPU fallback for genuine misses |
| `segment.vad_cut` | `diarization_turns` | `segments` | Silero VAD *within* each turn → single-speaker 3–20s clips, written as **48kHz FLAC** (since 2026-07-05, P5-A) |
| `pregate.snr` | `segments` | `pregate` | fast SNR/DNSMOS pre-gate before spending ASR GPU time |
| `asr.transcribe` | `segments` | `asr_results` | 4 models across 3 backends (faster-whisper canto_ft + whisper_v3, Qwen3-ASR-1.7B added 2026-07-07, SenseVoice-Small added 2026-07-08) — never `language="yue"` for the Whisper models |
| `asr.agreement` | `asr_results` | `asr_agreement` | cross-model char-overlap agreement score |
| `filter.text` | `asr_agreement` | `filters_text` | length/English-ratio/Mandarin-ratio gates — no audio decode |
| `filter.acoustic` | `filters_text` (pass=true only) | `filters_acoustic` | SNR + DNSMOS — CPU worker-subprocess pool |
| `filter.decide` | `filters_text` + `filters_acoustic` | `filters` | merges both into the final `pass`/`fail_reason` |
| `g2p` | `asr_agreement` (verified text only) | `g2p` | canto-hk-g2p → Jyutping; regex-validated `^[a-z]+[1-6]$` |
| `speaker.embed` | `segments` | `speaker_embeddings` | ECAPA-TDNN d-vector, sidecar-`.npy`-reuse-first |
| `speaker.cluster` | `speaker_embeddings` | `speakers` | cross-file agglomerative clustering, whole-source recompute |
| `tier.assign` | `asr_agreement` | `tiers` | **verification-confidence** tier: gold / silver / excluded (see note below — different axis from the label-framework A/B quality tier) |
| `manifest.build` / `manifest.export` | `filters`+`g2p`+`speakers`+`tiers` | `metadata/manifest.jsonl`, `train.jsonl`, `val.jsonl` | final JSONL assembly, 95/5 split |
| `label.suite` / `label.music` / `label.prosody` | `segments`/`raw_files` | `labels_lang`, `labels_overlap`, `labels_music`, `labels_prosody` | decode-once fan-out; feeds the separate TTS-quality label store |
| `label.calibrate` / `label.store` | label tables | `metadata/labels/calibration.json`, `metadata/labels.jsonl` | rate/pitch calibration + bucketed label export |
| `recover.orphans` | disk WAVs missing from catalog | `orphan_segments` | one-time: classify legacy pre-catalog segment WAVs, recover promising ones, queue the rest `pending_delete` (never auto-deletes) |
| `rebalance.segments` | `segments` | (moves files, updates `audio_path`) | one-time: 3-way Drive2/3/4 shard migration (P5-C, done 2026-07-06) |
| `raw.flac` | `raw_files` (`.wav` only) | `raw_flac` | one-time: transcode raw WAV backlog to lossless FLAC, bit-exact-verify before any delete |

Milestone status (`docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md` §8 is authoritative): **P0–P5 done**
(foundations, orchestrator core, decode-once label suite, all heavy-stage node ports, metadata
cutover, storage execution including the 3-way shard). **P6 (scale readiness) not yet started.**
A dataset-statistics report node (the old stage-10 equivalent) has **not** been ported yet —
see Acceptance Criteria below.

**Concurrency layer — `pipe run-many`**: DuckDB's write lock is per-*process*, not
per-transaction, so two separate `pipe run X` invocations can never hold the writer at
the same time. `python -m pipeline.cli run-many <node> -- <node> ...` opens **one**
shared connection and gives each node its own `conn.cursor()`, running them concurrently
via `asyncio.gather` — e.g. a GPU node (`asr.transcribe`) and a CPU node (`filter.acoustic`)
now run side-by-side instead of serializing on the lock. As of this writing **10 of 22**
node functions accept the `conn=` injection needed to join a `run-many` group (see
`docs/ORCHESTRATOR_PLAN.md` for the exact list and what's still mechanical follow-up).
A background sampler polls `nvidia-smi` and yields GPU resource-pool targets when a
*foreign* (non-orchestrator) process is using a GPU, so this coexists with other jobs
on the same machine rather than assuming exclusive access.

**Legacy-row-collision pattern**: several catalog tables (`filters`, `g2p`, `tiers`,
`raw_segments`) were pre-populated by the one-time P0 import of the old `manifest.jsonl`
(455,299 rows). A bare "does a row exist" anti-join would find zero unprocessed work
forever, since every legacy row already has a row. The fix, applied consistently: every
node tags its own writes with a `provenance` column value (e.g. `'filter_decide'`,
`'g2p_node'`, `'tier_assign'`) and discovery anti-joins on that specific value, not on
row existence. Follow this pattern for any new node writing into a table the P0 import
touched.

**Audio rate strategy**: store every segment as a **48 kHz mono master** (downsampling is lossy and irreversible — never store below this). VAD, diarization, ASR and DNSMOS all internally need 16 kHz; generate a *transient* 16 kHz copy for those tools and discard it. The 48 kHz master is what TTS training will consume (NeuCodec=24 kHz, MOSS-Nano=48 kHz, F5-TTS=24 kHz — all need ≥24 kHz).

**Raw storage policy**: new downloads (`ingest.download`) keep the **native container**
bestaudio format with zero transcoding — no conversion step, no quality loss, no wasted
CPU. The pre-2026-07-04 raw WAV backlog (decompressed from lossy AAC/opus/MP3 sources) is
being losslessly transcoded to FLAC by the `raw.flac` node (bit-exact-verified before the
original WAV is ever deleted).

**ASR strategy**: run **several** ASR models per segment across three independent backends — `canto_ft` + `whisper_v3` (Cantonese fine-tuned Whisper + base `large-v3`, both faster-whisper/ctranslate2, `language="zh"` + prompt), `qwen3_asr` (Qwen3-ASR-1.7B, transformers backend, native `language="Cantonese"` support — a distinct architecture, not Whisper-derived, added 2026-07-07), and `sense_voice` (SenseVoice-Small, funasr backend, CTC non-autoregressive, native `language="yue"` support, ~105× RTF, added 2026-07-08 — emits Simplified Chinese converted to Traditional HK via OpenCC s2hk, plus emotion/audio-event tags stored in `asr_results.metadata`, stripped from `text`). Store every candidate transcript in `asr_results`; `asr.agreement` computes N-way cross-model agreement (an id becomes eligible once ≥2 models have landed; a later straggler model re-triggers recompute rather than being ignored), and `filter.decide`/`tier.assign` decide what's trustworthy — there is no separate human-calibration UI stage currently wired into the DAG (the old stage-5 `05_calibrate.py` concept). Never auto-trust a single ASR output, and **never use `language="yue"`** for the Whisper models — it triggers decoder collapse on large-v3 (see `docs/KNOWN_ISSUES.md §9`); Qwen3-ASR and SenseVoice are architecturally distinct from Whisper and unaffected by this bug, using `language="Cantonese"`/`"yue"` directly. Known limitation: `filter.text`/`filter.decide`/`tier.assign` discovery is a bare row-existence anti-join (not provenance-tagged like `asr_agreement.model_count`), so segments already filtered/tiered before a later model (e.g. sense_voice) improved their `asr_agreement.best_text` are **not** automatically re-evaluated — only newly-discovered segments benefit from the improved agreement.

---

## Data Sources

### Existing RTHK Data — resolved (2026-07-02)

`../cantonese-tts-old/` (the prior pipeline's low-quality 22050 Hz RTHK/TVB/LegCo segments) is
**no longer on disk** — it was never reused directly. Instead `scripts/00_reingest.py` read its
filenames only, to recover the YouTube video IDs, then re-downloaded fresh 48 kHz audio through
the normal pipeline (all 6 non-skipped legacy categories completed 2026-06-09; see
`metadata/logs/00_reingest.log`). Per-category filename lists are archived at
`metadata/legacy_filenames/*.txt` in case `tier2_legco` (skipped by default — parliament noise,
poor TTS) ever needs enabling.

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
   (`cantonese-tts-old` and `gemma-hermes--tts` were deleted from disk 2026-07-02 — fully
   consumed/superseded, see `docs/PIPELINE_REARCHITECTURE_PLAN.md` §3 執行結果 — the rule still
   applies to any remaining ones.) Their **lessons** are encoded in `docs/KNOWN_ISSUES.md`. Read
   that instead.

4. **Absolute Linux paths only.** Every `audio_path` in every manifest must be an absolute path under your data root — corpus **raw** lives on `/mnt/Drive2/canto-corpus/data/raw/`; **segments** are now **3-way sharded** across `/mnt/Drive2/canto/segments/`, `/mnt/Drive3/canto/segments/`, and `/mnt/Drive4/canto/segments/` (deterministic `hash(raw_id or id) % 3`, see `config/storage_layout.py:shard_index()` / P5-C, done 2026-07-06 — never hand-pick a shard, always call that function). All three drives are ext4 and in active use; none is "reserved" or empty anymore. No relative paths, no Windows-style paths (`/mnt/d/`, `/mnt/c/Users/`).

5. **Single-speaker, VAD-based segmentation only.** Never hard-cut audio at a fixed duration. Segment at natural speech pause boundaries (Silero VAD) *within* diarization-detected single-speaker turns, so no clip spans a speaker change. Target 3–20 seconds. See `docs/KNOWN_ISSUES.md §2, §10`.

6. **48 kHz mono lossless master, never lower, never lossy (reworded 2026-07-04).** Store every segment at 48 kHz, encoded losslessly — **WAV or FLAC** are both acceptable containers; Opus/MP3/any lossy codec is never acceptable as a segment master. Downsampling is irreversible — a 16 kHz corpus would be unusable for every modern TTS codec. Create transient 16 kHz copies for VAD/ASR/DNSMOS only. New segments are written as **FLAC** (decided 2026-07-04 — see `DECISIONS.md`); existing legacy WAV segments are not re-encoded. See `docs/KNOWN_ISSUES.md §11`.

7. **Never `language="yue"` in Whisper.** It causes decoder collapse on large-v3. Use a Cantonese fine-tuned model and/or `language="zh"` with a written-Cantonese prompt. Run multiple ASR models; a human calibrates the canonical text. See `docs/KNOWN_ISSUES.md §9`.

8. **Validate Jyutping format explicitly.** Every Jyutping output must be validated: each space-separated token must match `^[a-z]+[1-6]$`. Reject segments where > 5% of tokens fail. Run G2P on the **human-verified** text, never raw single-ASR output. See `docs/KNOWN_ISSUES.md §1`.

9. **NEVER publish the dataset — zero-risk policy (decided 2026-06-29).** The corpus is private, for training the owner's own canto-tts model only. Do **not** publish, upload, or share — to Hugging Face or anywhere — any of: the dataset/manifests, the raw or filtered audio, the per-segment `source_url`s, or the reconstruction recipe (`reconstruct.py`, `metadata/manifest_release.jsonl`, `metadata/excluded_no_url.jsonl`). A canto-tts **model** trained on this data *may* be released later (weights only, never the data). The pipeline *code* stays open source. The release/reconstruction scripts are kept **dormant** (the no-release decision is not necessarily permanent) and must never be acted on without explicit owner approval. See `DECISIONS.md 2026-06-29`.

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
→ Always write `/mnt/Drive2/...` (raw) or `/mnt/Drive2|3|4/canto/segments/...` (3-way sharded
segments, per `config/storage_layout.py:shard_root()`) paths — not `/mnt/Drive1/` (moved off
there 2026-07-02, see `docs/PIPELINE_REARCHITECTURE_PLAN.md` §3). Run
`grep -c "/mnt/d/" metadata/manifest.jsonl` before finalising; must return 0. This is resolved
for current data (`rebalance.segments` fixed every path as part of P5-C, 2026-07-06) — the
concern going forward is any new node that writes `segments.audio_path` bypassing
`shard_root()`.

**Issue 9 — Whisper `yue` decoder collapse**
Forcing `language="yue"` on large-v3 produces repetition loops / garbage. → Cantonese fine-tuned model and/or `language="zh"` + prompt; multi-ASR + human calibration.

**Issue 11 — 16 kHz dead-end**
A 16 kHz corpus cannot train any modern TTS codec (all need ≥24 kHz). → Store 48 kHz master; downsample only transiently.

---

## Pipeline Node Conventions (current — `pipeline/nodes/*.py`)

Follow these conventions for any new or edited DAG node:

```python
"""
pipeline/nodes/example.py
example.thing DAG node -- one-line description of what it discovers and writes.
"""

async def run_example_thing(*, conn=None, batch_size: int = 5000, limit: int | None = None) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) — pass one when running
    alongside other nodes under `pipe run-many` (see filter.py's run_filter_acoustic
    docstring for the rationale). Defaults to a fresh self-managed connect() for
    standalone `pipe run example.thing` usage."""
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    conn = conn or connect()                       # conn injection: never call connect() if one was passed
    rows = discover(conn)                            # SQL anti-join on your OWN provenance tag, not bare row-existence
    if limit:
        rows = rows[:limit]
    run_id = new_run_id("example.thing")
    for batch in chunks(rows, batch_size):
        out = [{"id": r.id, ..., "provenance": "example_thing"} for r in batch]
        upsert_rows(conn, "your_table", out, ["id"])  # INSERT OR REPLACE — never partial-write a column set another node also writes
        record_batch(conn, run_id, "example.thing", [r["id"] for r in out], "ok")
    return {"processed": len(rows), "errors": 0, "run_id": run_id}
```

Checklist:
- **`conn=None` injection**: every `run_*()` takes `conn=None`; body does `conn = conn or connect()`. Register it in `RUN_MANY_ADAPTERS` in `pipeline/cli.py` once done, and add a `tests/test_run_many.py` regression test asserting `connect()` is never called when `conn` is passed.
- **Idempotent discovery via anti-join on `provenance`**, not bare row-existence — if the table was ever touched by the P0 legacy import (455,299 pre-populated rows), a bare `WHERE t.id IS NULL` anti-join will find zero work forever. Tag your own writes with a distinct `provenance` value and anti-join on that.
- **Register the CLI subcommand** in `pipeline/cli.py`'s `run_sub.add_parser("your.node", help="...")` — verify the exact registered name (e.g. it's `lang_screen.auto`, not `lang.screen`) before wiring anything that references it elsewhere.
- **Logging**: write to `metadata/logs/{node_name}.log` via the same `logging.basicConfig` pattern as before.
- **Never partial-write a shared table** — if two nodes each produce part of a table's columns (e.g. `filters_text` + `filters_acoustic` feeding `filters`), keep them as separate tables and have exactly one downstream node merge them; `upsert_rows()` is `INSERT OR REPLACE`, so partial writes from two nodes clobber each other's columns.
- **Test with `--limit N` against the real catalog before a full backlog run** — this is how several real bugs (legacy-row-collision, wrong CLI node names, missing sidecar-reuse paths) were actually caught in this project, not by unit tests alone.

**Audio format for stored files**: 48 kHz, mono, lossless — **FLAC** for all new segments (decided 2026-07-04, see `DECISIONS.md`), 16-bit PCM WAV for pre-2026-07-04 legacy segments (not re-encoded). Use `soundfile` for I/O (reads both natively); `pipeline/audio/bus.py` falls back to an ffmpeg pipe for containers `libsndfile` can't open (e.g. raw `.webm`/`.m4a`). For VAD / diarization / ASR / DNSMOS, generate a transient 16 kHz copy in memory or `/tmp` and discard it — never overwrite the 48 kHz master with a downsampled or lossy-recompressed version.

---

## Legacy Script Conventions (`scripts/*.py` — retiring, do not extend)

The handful of `scripts/NN_*.py` files still on disk predate the DAG re-architecture and
are kept only as pre-migration reference. Do not add new stages here — port to
`pipeline/nodes/` instead, following the conventions above.

```python
# --- Legacy file header (historical reference only) ---
#!/usr/bin/env python3
"""
scripts/NN_name.py
One-line description.
Usage: python scripts/NN_name.py --source [rthk|youtube|podcast|all] [--dry-run]
"""
```

**File naming**:
```
raw:      {YYYYMMDD}_{program_slug}_{video_id}.{ext}             (ext = native bestaudio container — webm/m4a/opus, no forced re-encode)
segments: {YYYYMMDD}_{program_slug}_{video_id}_seg{N:05d}.flac   (new, since 2026-07-05, P5-A)
          {YYYYMMDD}_{program_slug}_{video_id}_seg{N:05d}.wav    (legacy, pre-2026-07-05, unchanged)
```
"Filtered" is no longer a separate physical copy — §7.3 of `docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md`
retired physicalization; `filters.pass` in the catalog is the authoritative pass/fail signal, and
`segments.audio_path` (wherever it lives on the 3-way shard) is read directly.

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

⚠️ **"Tier" is overloaded — two different axes share the name.** The `tiers` catalog
table / `tier.assign` node produces a **verification-confidence** tier (`gold` = human
text-verified, `silver` = ASR agreement ≥ 0.65, `excluded`). `docs/LABEL_FRAMEWORK_SPEC.md`
§10 separately proposes an **A/B TTS-quality** tier (pretrain vs. clean) built from the
label store — not yet built. Do not conflate the two when reading/writing either one.

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

Check with: `python -m pipeline.cli catalog verify`, then query `metadata/corpus.duckdb` directly
for the criteria above (`filters`/`tiers`/`speakers`/`segments` tables). **A dedicated report
node has not been ported yet** — the legacy `scripts/10_report.py` (reads `data/filtered/`, not
the catalog) is stale and should not be trusted for current numbers; writing a `report.build`
node against the catalog is open work, not yet scheduled into a milestone.

---

## External Dependencies

| Tool | Source | Purpose |
|------|--------|---------|
| DuckDB | PyPI | The catalog (`metadata/corpus.duckdb`) — single source of truth for all node state |
| canto-hk-g2p | [github.com/typangaa/canto-hk-g2p](https://github.com/typangaa/canto-hk-g2p) | `g2p` node (Rust+PyO3) |
| faster-whisper | PyPI | `asr.transcribe` (canto_ft / whisper_v3) |
| qwen-asr | PyPI | `asr.transcribe` (qwen3_asr, transformers backend) |
| funasr + modelscope | PyPI (ModelScope model license, non-OSI, commercial use permitted) | `asr.transcribe` (sense_voice, SenseVoice-Small) |
| opencc-python-reimplemented | PyPI | `asr.transcribe` (sense_voice's Simplified→Traditional HK s2hk conversion) |
| pyannote.audio | PyPI + HF model terms | `segment.diarize` |
| speechbrain | PyPI | `speaker.embed` (ECAPA-TDNN) |
| speechmos | PyPI | `pregate.snr` / `filter.acoustic` DNSMOS quality gate |
| yt-dlp | PyPI / system | `ingest.download` |

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
