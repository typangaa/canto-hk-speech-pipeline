#!/usr/bin/env python3
"""
pipeline/catalog/ingest.py
P0 legacy importer — populates the DuckDB catalog from the existing
metadata/*.jsonl sidecars and sources/*.yaml. This is the "catalog rebuildable
from legacy jsonl" gate (docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md §12 #6).

Idempotent by design: every import_* function TRUNCATEs its target table(s)
immediately before inserting, so `build_catalog()` always reflects the
on-disk legacy files exactly — re-running it is a full, safe rebuild, not an
incremental append. Uses DuckDB's native read_json_auto() (vectorized) rather
than a Python row loop, since manifest.jsonl alone is 455k rows / 576MB.

Prerequisite: metadata/manifest.jsonl and metadata/downloaded.jsonl must
already have been remapped by scripts/fix_stale_paths.py (audio paths must
resolve on disk — see verify_paths() below, which is the P0 gate check).

Exclusions (deliberate, per owner-approved P0 scope, 2026-07-02):
  - metadata/lang_calib.jsonl    — pure 600-row subset of lang_id.jsonl, skip.
  - metadata/overlap_calib.jsonl — pure 600-row subset of overlap.jsonl, skip.
  - metadata/tag_calib.jsonl     — PARTIALLY included: only the ids NOT already
    covered by audio_tags.s0/s1.jsonl (469 net-new ids) are imported, via an
    anti-join against labels_music after s0+s1 land. Overlapping ids keep
    their s0/s1 (production full-pass) values, never overwritten by the
    calibration-run values.

Usage: python -m pipeline.catalog.ingest [--dry-run]
"""

import argparse
import logging
from pathlib import Path

import duckdb
import yaml

from pipeline.catalog.catalog import connect
from pipeline.config import CATALOG_PATH, LOGS_DIR, REPO_ROOT

METADATA_DIR = REPO_ROOT / "metadata"
SOURCES_DIR = REPO_ROOT / "sources"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "catalog_ingest.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("catalog.ingest")


# ---------------------------------------------------------------------------
# Per-source import functions — each is a full TRUNCATE + re-INSERT.
# ---------------------------------------------------------------------------

def import_raw_files(conn: duckdb.DuckDBPyConnection) -> int:
    """metadata/downloaded.jsonl -> raw_files.

    downloaded.jsonl has ~5% duplicate ids (00_reingest.py logged some raw
    files a second time when recovering legacy filenames — same wav_path /
    duration / downloaded_at, differing only in whether 'program' or
    'legacy_category' is populated). Dedup keeps the row with a non-null
    'program' when both exist.
    """
    src = METADATA_DIR / "downloaded.jsonl"
    if not src.exists():
        log.warning("Not found, skipping: %s", src)
        return 0

    conn.execute("TRUNCATE raw_files")
    conn.execute(f"""
        INSERT INTO raw_files
        SELECT
            id AS raw_id, wav_path, source, source_url, title, pub_date,
            program, domain, style, language, duration_sec, sample_rate,
            CAST(downloaded_at AS DATE) AS downloaded_at
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY id ORDER BY (program IS NOT NULL) DESC
            ) AS rn
            FROM read_json_auto('{src}')
        )
        WHERE rn = 1
    """)
    n = conn.execute("SELECT COUNT(*) FROM raw_files").fetchone()[0]
    log.info("raw_files: %d rows imported from %s", n, src.name)
    return n


def import_segments_and_manifest_tables(conn: duckdb.DuckDBPyConnection) -> dict:
    """metadata/manifest.jsonl -> segments, asr_results, asr_agreement,
    filters, g2p, tiers (one manifest row fans out to all six tables).
    """
    src = METADATA_DIR / "manifest.jsonl"
    if not src.exists():
        log.warning("Not found, skipping: %s", src)
        return {}

    counts = {}

    conn.execute("TRUNCATE segments")
    conn.execute(f"""
        INSERT INTO segments
            (id, audio_path, source, source_url, program, domain,
             duration_sec, sample_rate, speaker_id, gender, style, created_at)
        SELECT
            id, audio_path, source, source_url, program, domain,
            duration_sec, sample_rate, speaker_id, gender, style,
            CAST(created_at AS DATE) AS created_at
        FROM read_json_auto('{src}')
    """)
    counts["segments"] = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]

    conn.execute("TRUNCATE asr_results")
    conn.execute(f"""
        INSERT INTO asr_results
        SELECT m.id, c.model, c.text, c.confidence
        FROM read_json_auto('{src}') m, UNNEST(m.asr_candidates) AS t(c)
    """)
    counts["asr_results"] = conn.execute("SELECT COUNT(*) FROM asr_results").fetchone()[0]

    conn.execute("TRUNCATE asr_agreement")
    conn.execute(f"""
        INSERT INTO asr_agreement
        SELECT id, asr_agreement AS agreement, text AS best_text, text_verified
        FROM read_json_auto('{src}')
    """)
    counts["asr_agreement"] = conn.execute("SELECT COUNT(*) FROM asr_agreement").fetchone()[0]

    conn.execute("TRUNCATE filters")
    conn.execute(f"""
        INSERT INTO filters (id, snr_db, dnsmos, english_ratio)
        SELECT id, snr_db, dnsmos, english_ratio
        FROM read_json_auto('{src}')
    """)
    counts["filters"] = conn.execute("SELECT COUNT(*) FROM filters").fetchone()[0]

    conn.execute("TRUNCATE g2p")
    conn.execute(f"""
        INSERT INTO g2p (id, jyutping)
        SELECT id, jyutping
        FROM read_json_auto('{src}')
    """)
    counts["g2p"] = conn.execute("SELECT COUNT(*) FROM g2p").fetchone()[0]

    conn.execute("TRUNCATE tiers")
    conn.execute(f"""
        INSERT INTO tiers (id, tier)
        SELECT id, tier
        FROM read_json_auto('{src}')
    """)
    counts["tiers"] = conn.execute("SELECT COUNT(*) FROM tiers").fetchone()[0]

    log.info("manifest.jsonl fan-out: %s", counts)
    return counts


def import_labels_lang(conn: duckdb.DuckDBPyConnection) -> int:
    """metadata/lang_id.jsonl -> labels_lang. lang_calib.jsonl is a pure
    600-row subset of lang_id.jsonl (verified 2026-07-02) — excluded."""
    src = METADATA_DIR / "lang_id.jsonl"
    if not src.exists():
        log.warning("Not found, skipping: %s", src)
        return 0

    conn.execute("TRUNCATE labels_lang")
    conn.execute(f"""
        INSERT INTO labels_lang
        SELECT id, source, duration_sec, lang, lang_prob, yue_prob, cmn_prob, top3
        FROM read_json_auto('{src}')
    """)
    n = conn.execute("SELECT COUNT(*) FROM labels_lang").fetchone()[0]
    log.info("labels_lang: %d rows imported from %s (lang_calib.jsonl excluded — pure subset)", n, src.name)
    return n


def import_labels_overlap(conn: duckdb.DuckDBPyConnection) -> int:
    """metadata/overlap.jsonl -> labels_overlap. overlap_calib.jsonl is a
    pure 600-row subset of overlap.jsonl (verified 2026-07-02) — excluded."""
    src = METADATA_DIR / "overlap.jsonl"
    if not src.exists():
        log.warning("Not found, skipping: %s", src)
        return 0

    conn.execute("TRUNCATE labels_overlap")
    conn.execute(f"""
        INSERT INTO labels_overlap
        SELECT id, source, duration_sec, overlap_ratio, overlap_sec, speech_ratio
        FROM read_json_auto('{src}')
    """)
    n = conn.execute("SELECT COUNT(*) FROM labels_overlap").fetchone()[0]
    log.info("labels_overlap: %d rows imported from %s (overlap_calib.jsonl excluded — pure subset)", n, src.name)
    return n


def import_labels_music(conn: duckdb.DuckDBPyConnection) -> int:
    """metadata/audio_tags.s0.jsonl + .s1.jsonl (production full-pass shards,
    no id overlap between them) -> labels_music, provenance='s0'/'s1'.
    Then metadata/tag_calib.jsonl (calibration sample) is anti-joined against
    the ids already inserted — only the 469 net-new ids (owner-approved,
    2026-07-02) are added, provenance='tag_calib'. Overlapping ids keep their
    s0/s1 production values untouched.
    """
    s0 = METADATA_DIR / "audio_tags.s0.jsonl"
    s1 = METADATA_DIR / "audio_tags.s1.jsonl"
    tag_calib = METADATA_DIR / "tag_calib.jsonl"

    conn.execute("TRUNCATE labels_music")

    for shard_path, provenance in ((s0, "s0"), (s1, "s1")):
        if not shard_path.exists():
            log.warning("Not found, skipping: %s", shard_path)
            continue
        conn.execute(f"""
            INSERT INTO labels_music
            SELECT id, source, duration_sec, music_prob, music_tags, '{provenance}' AS provenance
            FROM read_json_auto('{shard_path}')
        """)

    if tag_calib.exists():
        conn.execute(f"""
            INSERT INTO labels_music
            SELECT tc.id, tc.source, tc.duration_sec, tc.music_prob, tc.music_tags, 'tag_calib' AS provenance
            FROM read_json_auto('{tag_calib}') tc
            WHERE tc.id NOT IN (SELECT id FROM labels_music)
        """)
    else:
        log.warning("Not found, skipping: %s", tag_calib)

    n = conn.execute("SELECT COUNT(*) FROM labels_music").fetchone()[0]
    by_prov = conn.execute(
        "SELECT provenance, COUNT(*) FROM labels_music GROUP BY provenance ORDER BY provenance"
    ).fetchall()
    log.info("labels_music: %d rows imported (by provenance: %s)", n, dict(by_prov))
    return n


def import_sources(conn: duckdb.DuckDBPyConnection) -> int:
    """sources/*.yaml -> sources. Low priority for the P0 gate (not row-count
    checked); best-effort parse, tolerant of the optional `download_config:`
    wrapper key some source files use alongside their entry list.
    """
    conn.execute("TRUNCATE sources")
    rows = []
    for yaml_path in sorted(SOURCES_DIR.glob("*.yaml")):
        text = yaml_path.read_text()
        # Some source files append a top-level `download_config:` mapping
        # after the entry list, making the file invalid as a single YAML
        # document (a bare list can't be followed by a mapping key at the
        # same level). scripts/02_download.py already works around this by
        # regex-splitting the config section off before parsing; do the same.
        import re
        config_match = re.search(r"^\s*download_config\s*:", text, re.MULTILINE)
        list_text = text[:config_match.start()] if config_match else text
        try:
            doc = yaml.safe_load(list_text)
        except Exception as exc:
            log.warning("Failed to parse %s: %s", yaml_path, exc)
            continue

        if isinstance(doc, dict):
            entries = doc.get("sources") or doc.get("channels") or doc.get("podcasts") or []
        elif isinstance(doc, list):
            entries = doc
        else:
            entries = []

        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("url"):
                continue
            rows.append({
                "source_key": entry["url"],
                "kind": yaml_path.stem,
                "program": entry.get("name", ""),
                "domain": entry.get("domain", ""),
                "style": entry.get("style", ""),
                "config": entry,
            })

    if not rows:
        log.info("sources: 0 rows (no parseable entries found)")
        return 0

    import json as _json
    conn.executemany(
        "INSERT OR REPLACE INTO sources (source_key, kind, program, domain, style, config) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (r["source_key"], r["kind"], r["program"], r["domain"], r["style"],
             _json.dumps(r["config"], ensure_ascii=False, default=str))
            for r in rows
        ],
    )
    n = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    log.info("sources: %d rows imported from %d yaml file(s)", n,
              len(list(SOURCES_DIR.glob("*.yaml"))))
    return n


# ---------------------------------------------------------------------------
# Path-existence verification (P0 gate)
# ---------------------------------------------------------------------------

def verify_paths(conn: duckdb.DuckDBPyConnection, sample_size: int = 2000) -> dict:
    """Sample `sample_size` rows from segments + raw_files and check
    os.path.exists() on each audio_path / wav_path. Returns a stats dict.
    """
    import os
    import random

    stats = {}
    for table, col in (("segments", "audio_path"), ("raw_files", "wav_path")):
        total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if total == 0:
            stats[table] = {"sampled": 0, "missing": 0}
            continue
        n = min(sample_size, total)
        rows = conn.execute(
            f"SELECT {col} FROM {table} USING SAMPLE {n} ROWS"
        ).fetchall()
        missing = [r[0] for r in rows if not os.path.exists(r[0])]
        stats[table] = {"sampled": len(rows), "missing": len(missing),
                         "missing_examples": missing[:5]}
    return stats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_catalog(dry_run: bool = False) -> dict:
    if dry_run:
        log.info("[DRY-RUN] Would rebuild catalog at %s (no changes made).", CATALOG_PATH)
        return {}

    conn = connect()  # single-writer connection; also runs init_schema()
    try:
        counts = {}
        counts["raw_files"] = import_raw_files(conn)
        counts.update(import_segments_and_manifest_tables(conn))
        counts["labels_lang"] = import_labels_lang(conn)
        counts["labels_overlap"] = import_labels_overlap(conn)
        counts["labels_music"] = import_labels_music(conn)
        counts["sources"] = import_sources(conn)

        conn.execute(
            "INSERT OR REPLACE INTO catalog_meta VALUES ('last_build', ?, CURRENT_TIMESTAMP)",
            [str(counts)],
        )

        path_check = verify_paths(conn)
        log.info("Path-existence sample check: %s", path_check)

        log.info("=== Catalog build complete: %s ===", counts)
        return {"counts": counts, "path_check": path_check}
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = build_catalog(dry_run=args.dry_run)

    if args.dry_run:
        return 0

    total_missing = sum(
        v.get("missing", 0) for v in result.get("path_check", {}).values()
    )
    if total_missing:
        log.error("Path-existence check found %d missing path(s) in sample.", total_missing)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
