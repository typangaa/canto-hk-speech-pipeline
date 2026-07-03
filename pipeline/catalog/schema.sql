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

-- From manifest.jsonl 'asr_candidates' list; one row per candidate ASR model per segment.
CREATE TABLE IF NOT EXISTS asr_results (
    id         TEXT,
    model      TEXT,
    text       TEXT,
    confidence DOUBLE,
    PRIMARY KEY (id, model)
);

-- From manifest.jsonl asr_agreement/text/text_verified fields; one row per segment with cross-model agreement stats.
CREATE TABLE IF NOT EXISTS asr_agreement (
    id            TEXT    PRIMARY KEY,
    agreement     DOUBLE,
    best_text     TEXT,
    text_verified BOOLEAN
);

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

-- From manifest.jsonl tier field; one row per segment recording quality tier ('gold'/'silver').
CREATE TABLE IF NOT EXISTS tiers (
    id   TEXT PRIMARY KEY,
    tier TEXT
);

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


-- Indexes for common query patterns

CREATE INDEX IF NOT EXISTS idx_segments_source     ON segments (source);
CREATE INDEX IF NOT EXISTS idx_segments_speaker_id ON segments (speaker_id);
CREATE INDEX IF NOT EXISTS idx_labels_music_prob   ON labels_music (music_prob);
