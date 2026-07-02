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
CREATE TABLE IF NOT EXISTS filters (
    id            TEXT    PRIMARY KEY,
    snr_db        DOUBLE,
    dnsmos        DOUBLE,
    english_ratio DOUBLE
);

-- From manifest.jsonl jyutping field; one row per segment with Cantonese romanisation.
CREATE TABLE IF NOT EXISTS g2p (
    id       TEXT PRIMARY KEY,
    jyutping TEXT
);

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

-- Future: per-segment speaker identity with embedding reference paths.
CREATE TABLE IF NOT EXISTS speakers (
    id            TEXT PRIMARY KEY,
    speaker_id    TEXT,
    embedding_ref TEXT
);

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
