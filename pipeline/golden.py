#!/usr/bin/env python3
"""
pipeline/golden.py
P0 golden-set builder: stratified sample (source × duration-bucket × tier) of
~500 segments, plus a frozen snapshot of every legacy sidecar output for those
same ids. The snapshot exists so P3's node-by-node parity tests have a fixed
baseline to diff against — captured now, before uv.lock dependency drift
(pyannote/faster-whisper etc.) could make a later re-run of the legacy
scripts produce different output than what's actually in production today.

Outputs:
  tests/golden/manifest.jsonl          — the sampled segment rows (catalog join)
  tests/golden/legacy_snapshot.jsonl   — per-id sidecar snapshot (see below)

Usage: python -m pipeline.golden
"""

import json
import logging

from pipeline.catalog.catalog import connect_ro
from pipeline.config import GOLDEN_LEGACY_SNAPSHOT, GOLDEN_MANIFEST, GOLDEN_SAMPLE_SIZE, LOGS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOGS_DIR / "golden_build.log"), logging.StreamHandler()],
)
log = logging.getLogger("golden")

# Sidecar json suffixes and which directory (relative to the segment's
# containing dir) each one lives in. .transcript.json is written by Stage 4
# alongside the SEGMENTS wav; the rest are written by Stages 6-8 alongside
# the FILTERED wav (06_filter.py copies the wav but not .transcript.json).
_FILTERED_SUFFIXES = [".filter.json", ".jyutping.json", ".speaker.json"]
_SEGMENTS_SUFFIXES = [".transcript.json"]


def stratified_sample(conn, sample_size: int = GOLDEN_SAMPLE_SIZE) -> list[str]:
    """Proportional stratified sample over (source, duration-bucket, tier).

    Duration buckets split the 3-20s quality-spec range into rough thirds:
    short [3,8), medium [8,14), long [14,20]. Each stratum gets
    round(sample_size * stratum_n / total_n) ids (floor 1, capped at the
    stratum's actual size), so the total is ~sample_size, not exact.
    """
    bucket_expr = (
        "CASE WHEN s.duration_sec < 8 THEN 'short' "
        "WHEN s.duration_sec < 14 THEN 'medium' ELSE 'long' END"
    )
    strata = conn.execute(f"""
        SELECT s.source, {bucket_expr} AS dur_bucket,
               COALESCE(t.tier, 'untiered') AS tier, COUNT(*) AS n
        FROM segments s LEFT JOIN tiers t ON s.id = t.id
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
    """).fetchall()

    total = sum(r[3] for r in strata)
    ids: list[str] = []
    for source, dur_bucket, tier, n in strata:
        target = min(n, max(1, round(sample_size * n / total)))
        rows = conn.execute(f"""
            SELECT s.id FROM segments s LEFT JOIN tiers t ON s.id = t.id
            WHERE s.source = ? AND ({bucket_expr}) = ?
              AND COALESCE(t.tier, 'untiered') = ?
            ORDER BY random() LIMIT ?
        """, [source, dur_bucket, tier, target]).fetchall()
        ids.extend(r[0] for r in rows)

    log.info("Stratified sample: %d strata, %d ids sampled (target ~%d)",
              len(strata), len(ids), sample_size)
    return ids


def write_golden_manifest(conn, ids: list[str]) -> None:
    GOLDEN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    placeholders = ", ".join("?" * len(ids))
    rows = conn.execute(f"""
        SELECT s.id, s.audio_path, s.source, s.source_url, s.program, s.domain,
               s.duration_sec, s.sample_rate, s.speaker_id, s.gender, s.style,
               a.best_text AS text, a.text_verified, a.agreement AS asr_agreement,
               f.snr_db, f.dnsmos, f.english_ratio, g.jyutping, t.tier
        FROM segments s
        LEFT JOIN asr_agreement a ON s.id = a.id
        LEFT JOIN filters f ON s.id = f.id
        LEFT JOIN g2p g ON s.id = g.id
        LEFT JOIN tiers t ON s.id = t.id
        WHERE s.id IN ({placeholders})
    """, ids).fetchall()
    cols = [d[0] for d in conn.description]

    with GOLDEN_MANIFEST.open("w") as f:
        for row in rows:
            f.write(json.dumps(dict(zip(cols, row)), ensure_ascii=False, default=str) + "\n")
    log.info("Wrote golden manifest: %d rows -> %s", len(rows), GOLDEN_MANIFEST)


def _read_json_if_exists(path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            log.warning("Failed to parse %s: %s", path, exc)
    return None


def write_legacy_snapshot(conn, ids: list[str]) -> None:
    """For each sampled id, freeze every currently-available legacy sidecar
    output (per-segment json files + catalog label rows) into one combined
    record. Missing sidecars are omitted (not null-filled) so a future parity
    test can tell "not yet computed" apart from "computed as empty"."""
    GOLDEN_LEGACY_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)

    placeholders = ", ".join("?" * len(ids))
    label_rows = {
        "labels_lang": conn.execute(
            f"SELECT * FROM labels_lang WHERE id IN ({placeholders})", ids
        ).fetchall(),
        "labels_overlap": conn.execute(
            f"SELECT * FROM labels_overlap WHERE id IN ({placeholders})", ids
        ).fetchall(),
        "labels_music": conn.execute(
            f"SELECT * FROM labels_music WHERE id IN ({placeholders})", ids
        ).fetchall(),
    }
    label_cols = {
        name: [d[0] for d in conn.execute(f"SELECT * FROM {name} LIMIT 0").description]
        for name in label_rows
    }
    label_by_id = {name: {} for name in label_rows}
    for name, rows in label_rows.items():
        for row in rows:
            d = dict(zip(label_cols[name], row))
            label_by_id[name][d["id"]] = d

    audio_paths = dict(conn.execute(
        f"SELECT id, audio_path FROM segments WHERE id IN ({placeholders})", ids
    ).fetchall())

    n_missing_sidecar = 0
    with GOLDEN_LEGACY_SNAPSHOT.open("w") as f:
        for seg_id in ids:
            audio_path = audio_paths.get(seg_id)
            if not audio_path:
                continue
            from pathlib import Path
            filtered_path = Path(audio_path)
            filtered_stem = filtered_path.with_suffix("")
            # segments/ dir mirrors filtered/ dir one level up, same relative subpath
            segments_stem = Path(str(filtered_stem).replace("/canto/filtered/", "/canto/segments/"))

            sidecars = {}
            for suffix in _FILTERED_SUFFIXES:
                d = _read_json_if_exists(Path(str(filtered_stem) + suffix))
                if d is not None:
                    sidecars[suffix.strip(".").replace(".json", "")] = d
                else:
                    n_missing_sidecar += 1
            for suffix in _SEGMENTS_SUFFIXES:
                d = _read_json_if_exists(Path(str(segments_stem) + suffix))
                if d is not None:
                    sidecars[suffix.strip(".").replace(".json", "")] = d
                else:
                    n_missing_sidecar += 1

            record = {
                "id": seg_id,
                "audio_path": audio_path,
                "sidecars": sidecars,
                "labels": {
                    name: label_by_id[name].get(seg_id)
                    for name in label_by_id
                    if label_by_id[name].get(seg_id) is not None
                },
            }
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    log.info("Wrote legacy snapshot: %d ids -> %s (%d sidecar files not found)",
              len(ids), GOLDEN_LEGACY_SNAPSHOT, n_missing_sidecar)


def main() -> int:
    conn = connect_ro()
    try:
        ids = stratified_sample(conn)
        write_golden_manifest(conn, ids)
        write_legacy_snapshot(conn, ids)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
