-- P0 catalog schema for speech-data-pipeline. Auto-generated from spec review; do not hand-edit blindly.

-- Tracks import provenance: source file mtimes, import timestamps, and schema version.
CREATE TABLE IF NOT EXISTS catalog_meta (
    key        TEXT        PRIMARY KEY,
    value      TEXT,
    updated_at TIMESTAMP
);

-- From sources/*.yaml; one row per corpus source; may be sparsely populated in P0.
CREATE TABLE IF NOT EXISTS sources (
    source_key TEXT    PRIMARY KEY,
    kind       TEXT,
    program    TEXT,
    domain     TEXT,
    style      TEXT,
    config     JSON
);

-- From metadata/downloaded.jsonl; one row per raw downloaded audio file before segmentation.
CREATE TABLE IF NOT EXISTS raw_files (
    raw_id         TEXT    PRIMARY KEY,
    wav_path       TEXT    NOT NULL,
    source         TEXT,
    source_url     TEXT,
    title          TEXT,
    pub_date       TEXT,
    program        TEXT,
    domain         TEXT,
    style          TEXT,
    language       TEXT,
    duration_sec   DOUBLE,
    sample_rate    INTEGER,
    downloaded_at  DATE
);

-- From metadata/manifest.jsonl core fields; one row per training segment, keyed by 12-hex-char segment id.
CREATE TABLE IF NOT EXISTS segments (
    id           TEXT    PRIMARY KEY,
    audio_path   TEXT    NOT NULL,
    source       TEXT,
    source_url   TEXT,
    program      TEXT,
    domain       TEXT,
    duration_sec DOUBLE,
    sample_rate  INTEGER,
    speaker_id   TEXT,
    gender       TEXT,
    style        TEXT,
    created_at   DATE
);
-- P3 session 4 (pipeline/nodes/segment.py): links a segment back to the raw_files row it
-- was cut from. NULL for all 455,299 P0 legacy-imported rows (manifest.jsonl never
-- recorded this link — segments.audio_path was the only surviving pointer). Populated
-- going forward by segment.vad_cut for every newly-created segment.
ALTER TABLE segments ADD COLUMN IF NOT EXISTS raw_id TEXT;

-- From manifest.jsonl 'asr_candidates' list; one row per candidate ASR model per segment.
CREATE TABLE IF NOT EXISTS asr_results (
    id         TEXT,
    model      TEXT,
    text       TEXT,
    confidence DOUBLE,
    metadata   JSON,  -- backend-specific extras, e.g. sense_voice's emotion/audio_event tags -- NULL for models/rows that don't produce any
    PRIMARY KEY (id, model)
);
ALTER TABLE asr_results ADD COLUMN IF NOT EXISTS metadata JSON;

-- From manifest.jsonl asr_agreement/text/text_verified fields; one row per segment with cross-model agreement stats.
CREATE TABLE IF NOT EXISTS asr_agreement (
    id            TEXT    PRIMARY KEY,
    agreement     DOUBLE,
    best_text     TEXT,
    text_verified BOOLEAN
);
ALTER TABLE asr_agreement ADD COLUMN IF NOT EXISTS model_count INTEGER;
-- canto_ft's own (real, logprob-derived) confidence for this id -- NULL if canto_ft has no
-- active row (see pipeline/nodes/asr.py's compute_agreement_row()). Added 2026-07-10 to gate
-- tier.assign's auto_gold tier (agreement >= 0.95 AND canto_ft_confidence > 0.8, threshold
-- raised 2026-07-11); populated going forward by asr.agreement, backfilled for existing rows --
-- see DECISIONS.md 2026-07-10 / 2026-07-11.
ALTER TABLE asr_agreement ADD COLUMN IF NOT EXISTS canto_ft_confidence DOUBLE;

-- From manifest.jsonl quality fields; one row per segment with acoustic quality filter scores.
-- P3 session 2 (filter.decide node) added mandarin_ratio/dnsmos_ovrl/detected_language/
-- language_confidence/pass/fail_reason — NULL for the 455,299 legacy-imported rows (which
-- predate this node and only ever contained passing segments by construction); populated
-- going forward by filter.decide, which always writes the full row (sole writer of the
-- final merged table, so no partial-column upsert ever clobbers these).
CREATE TABLE IF NOT EXISTS filters (
    id                   TEXT    PRIMARY KEY,
    snr_db               DOUBLE,
    dnsmos               DOUBLE,
    english_ratio        DOUBLE
);
ALTER TABLE filters ADD COLUMN IF NOT EXISTS mandarin_ratio DOUBLE;
ALTER TABLE filters ADD COLUMN IF NOT EXISTS dnsmos_ovrl DOUBLE;
ALTER TABLE filters ADD COLUMN IF NOT EXISTS detected_language TEXT;
ALTER TABLE filters ADD COLUMN IF NOT EXISTS language_confidence DOUBLE;
ALTER TABLE filters ADD COLUMN IF NOT EXISTS pass BOOLEAN;
ALTER TABLE filters ADD COLUMN IF NOT EXISTS fail_reason TEXT;
-- provenance distinguishes the 455,299 P0 legacy-imported rows (provenance IS NULL —
-- see catalog/ingest.py's import_segments_and_manifest_tables, which never sets it)
-- from rows filter.decide itself writes (provenance = 'filter_decide'). Necessary
-- because EVERY segment already has a legacy filters row (manifest.jsonl only ever
-- contained already-passing segments) — an anti-join on bare row-existence would
-- find zero "undone" work forever. filter.decide's discovery instead anti-joins on
-- provenance = 'filter_decide' specifically, so legacy rows are correctly treated as
-- not-yet-decided-by-this-node (relevant once a legacy segment also gains
-- filters_text/filters_acoustic rows) without needing a backfill migration.
ALTER TABLE filters ADD COLUMN IF NOT EXISTS provenance TEXT;
-- T5 (added 2026-07-17): snapshots filters_text.asr_model_count at filter.decide time, so
-- filter.decide's discovery can tell "already decided, filters_text unchanged since" apart
-- from "already decided, but filters_text was re-evaluated under a newer ASR model since" --
-- see filter.py's DECIDE_DISCOVER_SQL and pending_task.md T5.
ALTER TABLE filters ADD COLUMN IF NOT EXISTS text_model_count INTEGER;
-- T20 (added 2026-07-18): snapshots whether labels_lang had a row for this id at
-- filter.decide time, so discovery can re-trigger a re-decide once label.suite (an
-- independent, asynchronous node) lands a label for a segment already decided without
-- one. mandarin_audio_prob is stored for audit/debugging alongside the mandarin_audio
-- fail_reason it can produce -- see filter.py's MANDARIN_AUDIO_PROB_MIN and decide_row().
ALTER TABLE filters ADD COLUMN IF NOT EXISTS lang_label_checked BOOLEAN;
ALTER TABLE filters ADD COLUMN IF NOT EXISTS mandarin_audio_prob DOUBLE;

-- P3 session 2 (pipeline/nodes/filter.py): filter.text's own raw output — gates that need
-- only catalog columns + ASR text, no audio decode (sample_rate/duration/text-length/
-- english_ratio/mandarin_ratio). Kept separate from filters_acoustic (rather than both
-- upserting partial columns into one table) because upsert_rows() does INSERT OR REPLACE,
-- which resets any column not in the write's column list — two nodes partial-writing the
-- same table would clobber each other's columns on every run. filter.decide is the only
-- node that reads both and writes the merged `filters` row.
CREATE TABLE IF NOT EXISTS filters_text (
    id                   TEXT    PRIMARY KEY,
    english_ratio        DOUBLE,
    mandarin_ratio       DOUBLE,
    detected_language    TEXT,
    language_confidence  DOUBLE,
    pass                 BOOLEAN,
    fail_reason          TEXT
);
-- T5 (added 2026-07-17, pending_task.md / Issue #4): snapshots asr_agreement.model_count at
-- filter.text evaluation time. filter.text reads asr_agreement.best_text, so when a later
-- ASR model lands and asr.agreement recomputes best_text for an id (model_count increments --
-- see pipeline/nodes/asr.py), filter.text's own bare row-existence anti-join previously had no
-- way to notice the text underneath it had changed and would never re-evaluate. Discovery now
-- anti-joins on "no row OR asr_model_count != asr_agreement.model_count" instead.
ALTER TABLE filters_text ADD COLUMN IF NOT EXISTS asr_model_count INTEGER;

-- P3 session 2 (pipeline/nodes/filter.py): filter.acoustic's own raw output — SNR + DNSMOS,
-- both requiring an actual audio decode. Discovery only picks up ids where filters_text.pass
-- = TRUE (skips the expensive DNSMOS pass entirely for segments already rejected on text
-- grounds — the same short-circuit scripts/06_filter.py's --use-pregate flag approximated,
-- but item-level rather than a separate shard-parallel hack).
CREATE TABLE IF NOT EXISTS filters_acoustic (
    id           TEXT    PRIMARY KEY,
    snr_db       DOUBLE,
    dnsmos_sig   DOUBLE,
    dnsmos_ovrl  DOUBLE,
    pass         BOOLEAN,
    fail_reason  TEXT
);

-- From manifest.jsonl jyutping field; one row per segment with Cantonese romanisation.
-- valid_fraction added P3 session 2 (pipeline/nodes/g2p.py) — NULL for legacy rows (which,
-- like `filters`, only ever contained already-accepted output, i.e. valid_fraction >= 0.80
-- by construction, just never recorded numerically).
CREATE TABLE IF NOT EXISTS g2p (
    id       TEXT PRIMARY KEY,
    jyutping TEXT
);
ALTER TABLE g2p ADD COLUMN IF NOT EXISTS valid_fraction DOUBLE;
-- Same legacy-row-collision fix as filters.provenance above: all 455,299 P0
-- legacy-imported g2p rows have provenance IS NULL; g2p node writes tag
-- provenance = 'g2p_node' so its own discovery anti-join can tell "never
-- processed by this node" apart from "row exists" (which is otherwise always true).
ALTER TABLE g2p ADD COLUMN IF NOT EXISTS provenance TEXT;

-- From manifest.jsonl tier field; one row per segment recording quality tier.
-- P4 (pipeline/nodes/tier.py): same legacy-row-collision fix as filters.provenance/g2p.provenance —
-- all 455,299 P0 legacy-imported rows have provenance IS NULL (manifest.jsonl only ever contained
-- gold/silver segments, never a "pending" one, so a bare row-existence anti-join finds zero
-- undone work forever). tier.assign tags its own writes provenance = 'tier_assign' and anti-joins
-- on that exact value. NOTE: this verification-confidence tier is a DIFFERENT axis from
-- LABEL_FRAMEWORK_SPEC.md §10's proposed 'A'/'B' (pretrain/clean) TTS-quality tier — that is a
-- separate, not-yet-built consumer of the full label store (needs calibrate+build finished first,
-- plus emotion which is gated on an owner spot-check). Do not conflate the two when extending this.
-- Five values as of 2026-07-11 (DECISIONS.md): 'gold' (text_verified=True, human-reviewed via
-- calibrate_server), 'auto_gold' (agreement>=0.95 AND canto_ft_confidence>0.8, NOT human-reviewed
-- -- a statistical-confidence tier, sample-QA'd via calibrate.sample(tier='auto_gold'), never
-- claims to BE human-verified), 'silver' (agreement>=0.85), 'bronze' (agreement>=0.70, the
-- noisiest manifest-eligible tier -- QA'd at a higher sample rate, see calibrate.py's
-- QA_SAMPLE_RATE_BY_TIER), 'excluded' (agreement<0.70). Precedence gold > auto_gold > silver >
-- bronze > excluded -- see pipeline/nodes/tier.py's assign_tier(). Thresholds raised + bronze
-- added 2026-07-11 (was auto_gold>=0.90 / silver>=0.65 / no bronze, 2026-07-10) -- a stricter,
-- more conservative re-cut of the same corpus, not additive; see DECISIONS.md 2026-07-11.
CREATE TABLE IF NOT EXISTS tiers (
    id   TEXT PRIMARY KEY,
    tier TEXT
);
ALTER TABLE tiers ADD COLUMN IF NOT EXISTS provenance TEXT;
-- T5 (added 2026-07-17, pending_task.md / Issue #4): snapshots asr_agreement.model_count at
-- tier.assign time, so discovery can tell a stale tier (computed before a later ASR model
-- improved this id's agreement score) apart from a current one, and re-tier it. Rows written
-- by a human decision (provenance IN ('calibrate_verify', 'calibrate_reject'), see
-- calibrate.py's record_decision()) are deliberately EXCLUDED from this re-evaluation --
-- text_verified/human-rejected is a terminal, human-made call that a later statistical
-- recompute must never silently overturn (assign_tier()'s own docstring: "wins even if
-- agreement/dnsmos would also qualify for auto_gold" -- that invariant only holds if
-- discovery never revisits these rows at all). See tier.py's TIER_DISCOVER_SQL.
ALTER TABLE tiers ADD COLUMN IF NOT EXISTS asr_model_count INTEGER;

-- T13 (added 2026-07-16): A/B TTS-quality axis (docs/LABEL_FRAMEWORK_SPEC.md section 10),
-- a DIFFERENT axis from `tiers` above -- see pipeline/nodes/quality_tier.py's module
-- docstring. Only populated for the gold/auto_gold verification-confidence scope
-- (owner decision, canto-tts training pull).
CREATE TABLE IF NOT EXISTS quality_tiers (
    id           TEXT PRIMARY KEY,
    quality_tier TEXT,
    provenance   TEXT
);

-- Human calibration review queue (owner decision 2026-07-10: text_verified/gold was
-- structurally dead -- asr.transcribe always writes text_verified=False and no live DAG node
-- ever flips it, the only path that once did was the legacy scripts/05_calibrate.py against
-- data/segments/*.transcript.json sidecars that no longer exist). calibrate.sample
-- (pipeline/nodes/calibrate.py) inserts 'pending' rows for a random sample of filter-passing
-- segments; pipeline/tools/calibrate_server.py's local browser UI turns each into
-- 'verified'/'skipped'/'rejected' and, on 'verified', also flips asr_agreement.text_verified
-- and tiers.tier='gold' for that id. Sample-based by design, not full-corpus -- see
-- pipeline/nodes/calibrate.py's module docstring for why.
CREATE TABLE IF NOT EXISTS calibration_review (
    id            TEXT PRIMARY KEY,
    decision      TEXT,       -- 'pending' | 'verified' | 'skipped' | 'rejected' | 'flagged'
    reviewed_text TEXT,
    sample_batch  TEXT,
    queued_at     TIMESTAMP,
    reviewed_at   TIMESTAMP
);
-- 'flagged' (2026-07-10 UI iteration): a 4th decision for pipeline-bug reports surfaced during
-- review (bad segmentation, non-Cantonese slipping past lang_screen, corrupt audio, etc.) --
-- distinct from 'rejected' (this segment's text just isn't verifiable). Does NOT touch
-- asr_agreement/tiers like 'verified' does; flag_reason is free text for the owner to triage.
ALTER TABLE calibration_review ADD COLUMN IF NOT EXISTS flag_reason TEXT;
-- Snapshot of asr_agreement.best_text taken AT QUEUE TIME, before a 'verified' decision
-- overwrites best_text in place with the human-corrected text. Without this snapshot,
-- summary_stats' "how much did the human need to change" edit-distance metric would always
-- read 0 (comparing the corrected text against itself).
ALTER TABLE calibration_review ADD COLUMN IF NOT EXISTS original_best_text TEXT;

-- From metadata/lang_id.jsonl; per-segment language identification probabilities.
CREATE TABLE IF NOT EXISTS labels_lang (
    id           TEXT    PRIMARY KEY,
    source       TEXT,
    duration_sec DOUBLE,
    lang         TEXT,
    lang_prob    DOUBLE,
    yue_prob     DOUBLE,
    cmn_prob     DOUBLE,
    top3         JSON
);

-- From metadata/overlap.jsonl; per-segment speech/overlap ratio metrics.
CREATE TABLE IF NOT EXISTS labels_overlap (
    id            TEXT    PRIMARY KEY,
    source        TEXT,
    duration_sec  DOUBLE,
    overlap_ratio DOUBLE,
    overlap_sec   DOUBLE,
    speech_ratio  DOUBLE
);

-- From metadata/audio_tags.s0.jsonl + .s1.jsonl + tag_calib.jsonl; per-segment music detection; provenance records source shard ('s0'|'s1'|'tag_calib').
CREATE TABLE IF NOT EXISTS labels_music (
    id           TEXT    PRIMARY KEY,
    source       TEXT,
    duration_sec DOUBLE,
    music_prob   DOUBLE,
    music_tags   JSON,
    provenance   TEXT
);


-- P2 ingest.probe node output: ffprobe (duration/sr/channels/codec) + an L/R correlation
-- sample for stereo-vs-dual-mono discrimination, feeding the LABEL_FRAMEWORK §11 /
-- REARCHITECTURE_IMPLEMENTATION_PLAN §10 Q6 production-audio stereo feasibility report.
-- Keyed by raw_id (raw_files.raw_id), not segment id — probes the source file, not clips.
CREATE TABLE IF NOT EXISTS raw_probe (
    raw_id         TEXT    PRIMARY KEY,
    channels       INTEGER,
    codec          TEXT,
    sample_rate    INTEGER,
    duration_sec   DOUBLE,
    lr_correlation DOUBLE,   -- Pearson r over a sampled window (NULL if channels != 2)
    probed_at      TIMESTAMP
);


-- P3 session 4 (pipeline/nodes/segment.py): segment.diarize's own output — one row per
-- speaker turn found by pyannote (or the single VAD-only-mode "SPEAKER_UNKNOWN" turn
-- spanning the whole file when no HF token / gated-model access is available, exactly
-- matching scripts/03_segment.py's fallback). Consumed by segment.vad_cut, which runs
-- Silero VAD *within* each turn to find single-speaker silence-boundary clips (hard
-- constraint #5). Kept as its own table (not inlined into raw_segments) because the two
-- nodes are separate DAG resources (gpu vs cpu+io) that must be able to pipeline without a
-- barrier — segment.vad_cut discovers work the moment a raw file's turns exist.
CREATE TABLE IF NOT EXISTS diarization_turns (
    raw_id      TEXT,
    turn_idx    INTEGER,
    start_sec   DOUBLE,
    end_sec     DOUBLE,
    speaker_tag TEXT,
    PRIMARY KEY (raw_id, turn_idx)
);

-- P3 session 4 (pipeline/nodes/segment.py): per-raw-file completion marker for the
-- diarize+vad_cut chain. A random sample confirmed every current raw_files row already
-- has a legacy `{stem}_segments.jsonl` sidecar on disk (scripts/03_segment.py's own
-- idempotency marker) — i.e. it was already fully diarized+cut by the old monolithic
-- script. Re-running real pyannote+VAD over all 6,272 raw files would be pure waste (and
-- would duplicate already-existing segment WAVs under new ids). Both segment.diarize and
-- segment.vad_cut anti-join against this table (same hybrid reuse-first shape as
-- speaker_embeddings, see pipeline/nodes/speaker.py): on a sidecar hit, skip straight to
-- writing this marker with provenance='legacy_reused' (no diarization_turns row needed —
-- there is nothing new to hand off to segment.vad_cut); a genuine cache miss (a real new
-- raw file with no legacy sidecar) runs the real pipeline and writes
-- provenance='segment_vad_cut' once cutting finishes (or 'diarize_failed' if pyannote
-- itself errored, so the raw file is not retried forever).
CREATE TABLE IF NOT EXISTS raw_segments (
    raw_id       TEXT PRIMARY KEY,
    n_segments   INTEGER,
    provenance   TEXT,
    segmented_at TIMESTAMP
);

-- P3 session 4 (pipeline/nodes/segment.py, ported from scripts/03b_acoustic_pregate.py):
-- fast SNR(+DNSMOS) pre-gate applied to freshly-cut segments (raw_id IS NOT NULL) BEFORE
-- they ever reach ASR — avoids wasting GPU transcription time on clips filter.acoustic
-- would reject anyway. Deliberately a separate table from filters_acoustic: this runs
-- earlier in the DAG (no ASR text exists yet) and uses 03b's own SNR formula (percentile
-- energy ratio with a sliding hop), not filter.py's non-overlapping-frame variant used by
-- the later, authoritative filter.acoustic gate — the two are intentionally not required
-- to numerically agree, only to agree in spirit (reject the same obviously-bad clips
-- early). Only ever populated for pipeline-cut segments; the 455,299 legacy segments were
-- already filtered by the time they reached the catalog and never pass through this gate.
CREATE TABLE IF NOT EXISTS pregate (
    id          TEXT PRIMARY KEY,
    snr_db      DOUBLE,
    dnsmos      DOUBLE,
    pass        BOOLEAN,
    fail_reason TEXT,
    provenance  TEXT
);

-- Forward-compatible stubs (empty until a later milestone; DDL only, created now so imports never need a migration)

-- Future: per-segment prosody features (speaking rate, pitch, voiced duration, pause gaps).
CREATE TABLE IF NOT EXISTS labels_prosody (
    id           TEXT PRIMARY KEY,
    rate_raw     DOUBLE,
    f0_median_hz DOUBLE,
    f0_z         DOUBLE,
    gaps         JSON,
    voiced_sec   DOUBLE
);

-- Future: per-segment emotion classification probabilities from an emotion model.
CREATE TABLE IF NOT EXISTS labels_emotion (
    id   TEXT PRIMARY KEY,
    top  TEXT,
    conf DOUBLE,
    probs JSON
);

-- P3 session 3 (pipeline/nodes/speaker.py): speaker.embed's own raw output — one ECAPA-TDNN
-- d-vector .npy ref per segment. Kept separate from `speakers` (rather than both nodes
-- partial-writing the same table) for the same upsert-clobbering reason as
-- filters_text/filters_acoustic: speaker.cluster is the sole writer of the final `speakers`
-- row. Unlike filters/g2p, this table was NOT pre-populated by the P0 legacy import (the
-- legacy embedding cache lived as sibling .embed.npy files next to each filtered WAV, never
-- in manifest.jsonl/DuckDB) — a sample of the live corpus found ~100% of current segments
-- already have a matching sidecar .embed.npy from scripts/08_speaker_id.py's prior runs, so
-- speaker.embed's discovery-time policy is: reuse the sidecar file if present (near-zero
-- cost, no GPU), only invoke the GPU ECAPA encoder for genuine cache misses.
CREATE TABLE IF NOT EXISTS speaker_embeddings (
    id            TEXT PRIMARY KEY,
    source        TEXT,
    embedding_ref TEXT,
    provenance    TEXT   -- 'legacy_reused' (sidecar .npy found) | 'speaker_embed_node' (freshly computed on GPU)
);
-- 2026-07-12 (I/O optimization, docs/IO_OPTIMIZATION_PLAN.md Phase 3): the embedding
-- vector itself, stored in-table instead of read via embedding_ref's per-file .npy
-- sidecar -- speaker.cluster's dominant cost was opening hundreds of thousands of tiny
-- sidecar files (ext4 dentry-cache thrashing on the giant flat segments/{source}/ dirs),
-- not the clustering compute itself. embedding_ref is kept during the transition (rollback
-- safety) and is speaker.cluster's fallback for any row where embedding IS NULL.
ALTER TABLE speaker_embeddings ADD COLUMN IF NOT EXISTS embedding FLOAT[192];

-- Per-segment final speaker identity. segments.speaker_id already carries the legacy
-- manifest-derived value for all 455,299 P0 rows (a plain column, not this table — this
-- `speakers` table itself was never populated by the P0 import, so no provenance-collision
-- fix is needed here unlike filters/g2p). speaker.cluster is the sole writer, always writing
-- the full row (id, speaker_id, cluster_id, embedding_ref, gender, provenance) once per run.
CREATE TABLE IF NOT EXISTS speakers (
    id            TEXT PRIMARY KEY,
    speaker_id    TEXT,
    embedding_ref TEXT
);
ALTER TABLE speakers ADD COLUMN IF NOT EXISTS cluster_id INTEGER;
ALTER TABLE speakers ADD COLUMN IF NOT EXISTS gender TEXT;
ALTER TABLE speakers ADD COLUMN IF NOT EXISTS provenance TEXT;

-- Future: human-verified transcripts with verifier identity and timestamp.
CREATE TABLE IF NOT EXISTS verified (
    id          TEXT PRIMARY KEY,
    text        TEXT,
    verified_by TEXT,
    verified_at DATE
);

-- Orchestrator audit log for pipeline node executions, one row per item per run.
-- PK enables the same INSERT OR REPLACE upsert_rows() helper every other table uses.
CREATE TABLE IF NOT EXISTS task_runs (
    run_id   TEXT,
    node     TEXT,
    item_id  TEXT,
    status   TEXT,
    started  TIMESTAMP,
    finished TIMESTAMP,
    error    TEXT,
    metrics  JSON,
    PRIMARY KEY (run_id, node, item_id)
);


-- lang_screen.auto DAG node (raw-level pre-segmentation language pre-filter, added
-- 2026-07-04): coarse Mandarin-vs-Cantonese screen run BEFORE segment.diarize, so a
-- raw file that turns out to be Mandarin-dominant never pays for diarization or (once
-- P5-B lands) FLAC transcoding (raw backlog format decided FLAC, not opus — see
-- DECISIONS.md 2026-07-04 "Storage format policy FINALIZED"). Keyed by raw_id, NOT segment id — this is a whole-file
-- decision, not a per-clip one. Deliberately permissive: this is a coarse sampled
-- pre-filter that only rejects on a high mandarin_ratio_raw, NOT a replacement for
-- the existing fine-grained per-segment lang-id in labels_lang/label.suite, which
-- runs AFTER segmentation and remains the final gate for intra-episode
-- code-switching / low-level Mandarin content. decision is written once by
-- lang_screen.auto and never overwritten
-- by it again (raw_id is anti-joined out of discover() the moment a row exists);
-- human_decision is written only by the separate lang_screen.review human-in-loop CLI
-- and, when present, takes precedence over decision — see docs' COALESCE(human_decision,
-- decision, 'pass') "effective decision" formula used by segment.diarize's discovery
-- query. A raw_id with NO lang_screen row at all (not yet screened, or legacy/pre-dates
-- this node) is treated as 'pass' by that COALESCE — this node is additive and must
-- never retroactively block already-segmented or not-yet-screened raw files.
CREATE TABLE IF NOT EXISTS lang_screen (
    raw_id              TEXT      PRIMARY KEY,
    decision            TEXT,     -- 'pass' | 'mixed' | 'reject' (revised 2026-07-04, twice same day)
                                   -- pass:   cantonese_ratio_raw >= 0.70 AND mandarin_ratio_raw <= 0.20
                                   -- reject: mandarin_ratio_raw  >  0.50 OR  cantonese_ratio_raw <  0.40
                                   -- mixed:  everything else -- effective decision treats 'mixed' the
                                   --         SAME as 'pass' (only 'reject' blocks segment.diarize), the
                                   --         'mixed' value itself is the tag a later stage can join on
    cantonese_ratio_raw DOUBLE,   -- fraction of sampled windows with top-1 lang = 'yue' (gates decision, see above)
    mandarin_ratio_raw  DOUBLE,   -- fraction of sampled windows with top-1 lang = 'cmn' (gates decision, see above)
    n_windows           INTEGER,
    window_starts       JSON,     -- [start_sec, ...] -- kept for provenance even though no review UI reads it anymore
    needs_review        BOOLEAN,  -- ALWAYS false going forward -- human review of 'reject' was removed
                                   -- 2026-07-04 (third revision same day): reject is now trusted
                                   -- automatically. Column kept for the raw_ids reviewed before this
                                   -- change and in case manual spot-checks are ever wanted again.
    human_decision       TEXT,    -- 'pass' | 'reject' | 'mixed' | NULL -- set only by the now-removed
                                   -- review tool; existing values are preserved and still take
                                   -- precedence over decision (see COALESCE formula above)
    reviewed_by          TEXT,
    reviewed_at           TIMESTAMP,
    screened_at           TIMESTAMP,
    provenance            TEXT    -- 'lang_screen_auto' | 'read_failed'
);

-- raw.flac DAG node (P5-B, added 2026-07-05): transcodes the existing ~1.6T WAV raw
-- backlog to lossless FLAC (raw backlog format decided FLAC — see DECISIONS.md
-- 2026-07-04 "Storage format policy FINALIZED"; NOT opus, that direction was
-- reopened and rejected same day). Keyed by raw_id, one row per raw file (whole-file
-- decision, matching lang_screen's own raw-level granularity). Eligibility (see
-- pipeline/nodes/raw_flac.py's discovery SQL) requires EITHER a raw_segments row
-- (already segmented — cut first, so a mid-transcode crash never forces re-decoding
-- the source for segmentation) OR a lang_screen 'reject' decision (a rejected raw
-- file will never enter raw_segments via segment.diarize, so it must be transcoded
-- via this alternate path or its WAV would sit on Drive2 forever). Never targets
-- raw_files whose wav_path is a native (non-.wav) container — those are already
-- lossy-compressed at the source and re-encoding to FLAC would only inflate size
-- with zero fidelity gain (measured 2.1-3.5x bigger, see DECISIONS.md).
CREATE TABLE IF NOT EXISTS raw_flac (
    raw_id         TEXT PRIMARY KEY,
    flac_path      TEXT,
    duration_sec   DOUBLE,     -- measured post-transcode, compared against raw_files.duration_sec for a sanity check
    verified       BOOLEAN,    -- true only after a full PCM bit-exact decode comparison against the original WAV
    wav_deleted_at TIMESTAMP,  -- NULL = original .wav still on disk, non-null = deleted (only after verified=true and a signed-off batch)
    transcoded_at  TIMESTAMP,
    provenance     TEXT        -- 'raw_flac' | 'transcode_failed'
);

-- P5-C (pipeline/nodes/rebalance.py): one-time cross-disk shard rebalance of the
-- `segments` table onto the 3-way hash(coalesce(raw_id, id)) % n_shards layout in
-- config/storage_layout.yaml's `sharding` block. Same two-phase shape as raw_flac:
-- phase 1 (default) copies + byte-verifies into the target shard and writes this
-- row with verified=true, WITHOUT touching segments.audio_path or deleting the
-- original file; phase 2 (--delete-verified) does the transactional catalog
-- update + physical delete of the old copy, only for rows already verified=true.
-- pipeline/nodes/recover_orphans.py (2026-07-06) -- physical segment WAVs found on
-- disk that scripts/03_segment.py (the pre-P0 legacy pipeline) cut as VAD candidates
-- but that never made it into manifest.jsonl / the `segments` table (rejected by the
-- legacy filter stage, or simply never finished processing). Each row is EITHER a
-- recovery (status='recovered' -- backfilled into segments/asr_results/asr_agreement
-- so the normal filter.text/filter.acoustic/filter.decide/tier.assign nodes decide its
-- fate like any other segment) OR a queued-not-deleted low-quality candidate
-- (status='pending_delete' -- flagged for a future, separately-approved cleanup pass;
-- this table never triggers a physical delete by itself).
CREATE TABLE IF NOT EXISTS orphan_segments (
    audio_path    TEXT PRIMARY KEY,
    source        TEXT,
    bucket        TEXT,       -- classification bucket, see recover_orphans.py CLASSIFY
    bytes         BIGINT,
    status        TEXT,       -- 'recovered' | 'pending_delete'
    recovered_id  TEXT,       -- segments.id, set only when status='recovered'
    classified_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS segment_shard_migrations (
    id           TEXT PRIMARY KEY,  -- segments.id
    old_path     TEXT,
    new_path     TEXT,
    target_shard INTEGER,
    verified     BOOLEAN,    -- true only after a full byte-for-byte comparison against the original
    migrated_at  TIMESTAMP,  -- NULL = old copy still on disk, non-null = catalog repointed + old copy deleted
    copied_at    TIMESTAMP,
    provenance   TEXT        -- 'rebalance' | 'copy_failed'
);

-- Indexes for common query patterns

CREATE INDEX IF NOT EXISTS idx_segments_source     ON segments (source);
CREATE INDEX IF NOT EXISTS idx_segments_speaker_id ON segments (speaker_id);
CREATE INDEX IF NOT EXISTS idx_labels_music_prob   ON labels_music (music_prob);
