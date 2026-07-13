"""
pipeline/nodes/ingest_download.py
ingest.download DAG node — download audio for rthk/youtube/podcast sources.

Policy (2026-07-04, DECISIONS.md "Storage format policy FINALIZED after
external research"): keep the native bestaudio/RSS-enclosure container
exactly as delivered — YouTube opus (webm/m4a), RTHK AAC (m4a/mp4), podcast
MP3 — with ZERO re-encode at download time. No forced WAV conversion, no
forced FLAC/opus transcode, no stereo->mono downmix. This replaces the
previous behaviour (scripts/02_download.py, still present as the legacy
reference implementation) which force-converted every download to 48kHz
mono PCM WAV via ffmpeg — a policy that was approved 2026-07-02 but never
implemented, then reopened and re-confirmed 2026-07-04 after an online
research pass found: (a) re-encoding an already-lossy source is not
acceptable archival practice regardless of target codec, (b) a stereo-to-
mono downmix ahead of any further lossy pass risks phase cancellation in
the 0-8kHz speech band, (c) DNSMOS penalizes re-compressed audio even when
perceptually transparent, biasing Stage-6 filter yield downward. See
CLAUDE.md's documented raw naming convention (`{date}_{slug}_{id}.webm`),
which already assumed a native container — the WAV-everything behaviour
was the deviation, not the other way around.

duration_sec / sample_rate are intentionally left NULL here: libsndfile
cannot reliably open every native container this node produces (AAC-in-
MP4/m4a in particular), and ingest.probe (pipeline/nodes/ingest_probe.py)
already exists specifically to ffprobe these fields for any raw file
shortly after download — no need to duplicate that logic here.

Two-phase, JSON-staged (2026-07-06 revision): every other P3+ node opens a
single read-write DuckDB connection and holds it for its *entire* runtime
(pipeline/catalog/catalog.py's documented single-writer constraint — only
one process may hold that connection at a time). ingest.download's actual
work is 99% network I/O (yt-dlp / RSS fetch, potentially hours per run)
and only ever needed the connection for two cheap things: an up-front
dedup read and a small upsert after each entry. Holding a writer open for
the whole download run meant it could never overlap with any other
catalog-writing node (e.g. filter.acoustic backfill runs that also take
hours) — a purely architectural collision, not a real resource conflict.

So this node now never opens a DuckDB connection at all during the
download phase:
  - Dedup reads a point-in-time JSON snapshot (KNOWN_IDS_SNAPSHOT) instead
    of `SELECT raw_id FROM raw_files` live, plus whatever this run has
    itself already appended to STAGING_FILE (so re-running before a commit
    never re-downloads the same episode/video twice).
  - Each entry's new rows are appended as JSON lines to STAGING_FILE
    instead of upserted into `raw_files`.
  - `ingest.commit` (run_ingest_commit(), CLI: `pipe run ingest.commit`) is
    the only part of this node that touches the catalog: it opens one
    short-lived connection, upserts everything staged, refreshes
    KNOWN_IDS_SNAPSHOT from the now-current `raw_files`, and archives the
    consumed staging file. Run it any time nothing else holds the writer
    lock — it takes seconds, not hours.

raw_id keeps the legacy id scheme so downstream joins against pre-existing
rows keep working: md5(audio_url)[:8] for RSS episodes, the 11-char
YouTube video id for YouTube.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import re
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests
import yaml

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
SOURCES_DIR = ROOT / "sources"
RAW_DIR = ROOT / "data" / "raw"
METADATA_DIR = ROOT / "metadata"
STAGING_FILE = METADATA_DIR / "ingest_download_staging.jsonl"
KNOWN_IDS_SNAPSHOT = METADATA_DIR / "raw_files_known_ids.json"

SOURCE_FILES = {
    "rthk": SOURCES_DIR / "rthk_sources.yaml",
    "youtube": SOURCES_DIR / "youtube_channels.yaml",
    "podcast": SOURCES_DIR / "podcast_sources.yaml",
}

# Native container extensions this node can produce (RSS enclosures + yt-dlp
# bestaudio, no extraction) — used to snapshot "files already in out_dir"
# before/after a yt-dlp run, since we no longer force a single extension.
AUDIO_EXTS = ("*.webm", "*.m4a", "*.mp4", "*.opus", "*.ogg", "*.mp3", "*.aac", "*.wav")


# ---------------------------------------------------------------------------
# YAML source config loading (same shape as scripts/02_download.py)
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_-]+", "_", text).strip("_")[:40]


def _load_entries(path: Path) -> list[dict]:
    with open(path) as f:
        content = f.read()
    config_match = re.search(r"^\s*download_config\s*:", content, re.MULTILINE)
    list_content = content[: config_match.start()] if config_match else content
    try:
        raw = yaml.safe_load(list_content)
    except yaml.YAMLError:
        return []
    entries: list[dict] = []
    if isinstance(raw, list):
        entries = [e for e in raw if isinstance(e, dict) and "name" in e]
    elif isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list):
                entries.extend([e for e in v if isinstance(e, dict) and "name" in e])
    return entries


def discover_active_entries(source: str) -> list[dict]:
    path = SOURCE_FILES[source]
    if not path.exists():
        return []
    entries = _load_entries(path)
    return [
        e for e in entries
        if e.get("status") not in ("skip", "done", "paused")
        and e.get("priority") not in ("skip",)
        and not str(e.get("url", "")).startswith(("PLACEHOLDER", "SKIP", "SEARCH"))
    ]


# ---------------------------------------------------------------------------
# Catalog dedup — via JSON snapshot + staged-but-uncommitted rows, NEVER a
# live DB read during the download phase (see module docstring).
# ---------------------------------------------------------------------------

def existing_raw_ids(conn, source: str) -> set[str]:
    """Live DB read — used only by run_ingest_commit() to rebuild the
    snapshot after a commit, never during the download phase itself."""
    rows = conn.execute(
        "SELECT raw_id FROM raw_files WHERE source = ?", [source]
    ).fetchall()
    return {r[0] for r in rows}


def load_known_ids_snapshot() -> dict[str, list[str]]:
    """Point-in-time dedup snapshot written by run_ingest_commit(). Missing
    file (e.g. before the very first commit) is treated as empty — downloads
    proceed, worst case a few already-known items get re-staged, which
    ingest.commit's raw_id-keyed upsert absorbs harmlessly as a no-op
    duplicate, not data loss."""
    if not KNOWN_IDS_SNAPSHOT.exists():
        log.warning(
            f"{KNOWN_IDS_SNAPSHOT.name} not found — dedup will miss anything "
            f"downloaded since the last `ingest.commit`. Run `pipe run "
            f"ingest.commit` once to bootstrap it if this is unexpected."
        )
        return {}
    with open(KNOWN_IDS_SNAPSHOT, "r", encoding="utf-8") as f:
        return json.load(f)


def write_known_ids_snapshot(conn) -> None:
    """Rebuild KNOWN_IDS_SNAPSHOT from the live `raw_files` table. Called by
    run_ingest_commit() right after it upserts, so the next `ingest.download`
    run's dedup reflects everything committed so far."""
    snapshot = {
        src: sorted(existing_raw_ids(conn, src)) for src in SOURCE_FILES
    }
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(KNOWN_IDS_SNAPSHOT, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


def load_staged_ids(source: str) -> set[str]:
    """raw_ids already appended to STAGING_FILE (by this or a prior,
    not-yet-committed run) — prevents re-downloading the same item twice
    across multiple staged sessions before `ingest.commit` runs."""
    if not STAGING_FILE.exists():
        return set()
    ids: set[str] = set()
    with open(STAGING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("source") == source:
                ids.add(row["raw_id"])
    return ids


def append_staged_rows(rows: list[dict]) -> None:
    """Append newly-downloaded rows as JSON lines — no DB connection opened.
    Flushed per call so a killed/crashed download run loses at most the
    in-flight entry, never previously-appended ones."""
    if not rows:
        return
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(STAGING_FILE, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


# ---------------------------------------------------------------------------
# RSS download (rthk, podcast) — native container, no ffmpeg conversion
# ---------------------------------------------------------------------------

def _find_audio_url(episode: dict) -> str | None:
    for enc in episode.get("enclosures", []):
        enc_type = enc.get("type", "")
        if "audio" in enc_type or "video" in enc_type or "mp4" in enc_type:
            return enc.get("href") or enc.get("url")
    for link in episode.get("links", []):
        href = link.get("href", "")
        ltype = link.get("type", "")
        if "audio" in ltype or "video" in ltype or href.endswith((".mp3", ".m4a", ".aac", ".mp4")):
            return href
    return None


def _download_rss_episode(
    episode: dict, entry: dict, out_dir: Path, ep_id: str, audio_url: str, dry_run: bool,
) -> dict | None:
    pub = episode.get("published_parsed")
    pub_date = date(*pub[:3]).strftime("%Y%m%d") if pub else "unknown"
    program_slug = _slugify(entry["name"])
    ext = Path(audio_url.split("?")[0]).suffix or ".mp3"
    raw_name = f"{pub_date}_{program_slug}_{ep_id}{ext}"
    raw_path = out_dir / raw_name

    if raw_path.exists():
        return None
    if dry_run:
        log.info(f"[DRY-RUN] would download (native, no convert): {raw_name}")
        return None

    log.info(f"Downloading (native container): {audio_url}")
    try:
        r = requests.get(audio_url, stream=True, timeout=120,
                          headers={"User-Agent": "canto-hk-speech-pipeline/0.1"})
        r.raise_for_status()
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
    except Exception as exc:
        log.error(f"download failed {audio_url}: {exc}")
        raw_path.unlink(missing_ok=True)
        return None

    return {
        "raw_id": ep_id,
        "wav_path": str(raw_path),
        "source_url": audio_url,
        "title": episode.get("title", ""),
        "pub_date": pub_date,
        "program": entry["name"],
        "source": entry.get("source", "podcast"),
        "domain": entry.get("domain", ""),
        "style": entry.get("style", ""),
        "language": entry.get("language", "yue"),
        "duration_sec": None,   # filled in by ingest.probe (ffprobe)
        "sample_rate": None,    # filled in by ingest.probe (ffprobe)
        "downloaded_at": date.today().isoformat(),
    }


def _download_rss_source(
    entry: dict, known_ids: set[str], args: argparse.Namespace,
) -> list[dict]:
    url = entry.get("url") or entry.get("rss_url", "")
    if not url or url.startswith(("PLACEHOLDER", "SKIP", "SEARCH")):
        log.warning(f"Skipping {entry['name']}: no valid URL")
        return []

    source = entry.get("source", "podcast")
    out_dir = RAW_DIR / source

    log.info(f"Fetching RSS: {entry['name']} -> {url}")
    try:
        # feedparser.parse(url) has no socket timeout of its own and can hang
        # indefinitely on a slow/dead feed server; fetch via requests (which does
        # have one) and hand it the raw bytes instead.
        resp = requests.get(url, timeout=30, headers={"User-Agent": "canto-hk-speech-pipeline/0.1"})
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception as exc:
        log.warning(f"RSS fetch failed, skipping {entry['name']}: {exc}")
        return []
    if not feed.entries:
        log.warning(f"Empty feed: {url}")
        return []

    max_age = int(entry.get("max_age_days", 365))
    cutoff = date.today() - timedelta(days=max_age) if max_age > 0 else None

    out_rows: list[dict] = []
    considered = 0
    for ep in feed.entries:
        audio_url = _find_audio_url(ep)
        if not audio_url:
            continue
        ep_id = hashlib.md5(audio_url.encode()).hexdigest()[:8]
        if ep_id in known_ids:
            continue  # already downloaded — free in-memory check, doesn't cost the --limit budget
        pub = ep.get("published_parsed")
        if pub and cutoff and date(*pub[:3]) < cutoff:
            continue
        if args.limit and considered >= args.limit:
            log.info(f"Reached --limit {args.limit}, stopping")
            break
        considered += 1
        row = _download_rss_episode(ep, entry, out_dir, ep_id, audio_url, args.dry_run)
        if row:
            out_rows.append(row)
            known_ids.add(ep_id)
        if not args.dry_run:
            time.sleep(2)

    return out_rows


# ---------------------------------------------------------------------------
# YouTube download — bestaudio, no extraction/re-encode -> native container
# ---------------------------------------------------------------------------

def _extract_yt_id(stem: str) -> str | None:
    tail = stem[-11:]
    return tail if re.match(r"^[A-Za-z0-9_-]{11}$", tail) else None


def _snapshot_audio_files(out_dir: Path) -> set[str]:
    names: set[str] = set()
    for pattern in AUDIO_EXTS:
        names.update(p.name for p in out_dir.glob(pattern))
    return names


def _download_youtube_source(
    entry: dict, known_ids: set[str], args: argparse.Namespace,
) -> list[dict]:
    url = entry.get("url", "")
    if not url or url.startswith(("SKIP", "SEARCH", "PLACEHOLDER")):
        log.warning(f"Skipping {entry['name']}: no valid URL")
        return []

    source = entry.get("source", "youtube")
    out_dir = RAW_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)

    name_slug = _slugify(entry["name"])
    outtmpl = str(out_dir / f"%(upload_date)s_{name_slug}_%(id)s.%(ext)s")
    before = _snapshot_audio_files(out_dir)

    cmd = [
        "yt-dlp",
        "--no-playlist" if entry.get("type") not in ("playlist", "channel") else "--yes-playlist",
        "--format", "bestaudio/best",   # native container, NO --extract-audio/--audio-format
        "--restrict-filenames",
        "--socket-timeout", "30",
        "--sleep-interval", "3", "--max-sleep-interval", "8",
        "--sleep-requests", "2",
        "--retries", "5",
        "--retry-sleep", "exp=1:60",
        "--output", outtmpl,
        "--download-archive", str(ROOT / "metadata" / "yt_archive.txt"),
        "--match-filter", "duration > 120 & duration < 14400",
        "--max-filesize", "500M",
        "--no-warnings", "--quiet", "--ignore-errors", "--no-part",
    ]
    if args.limit:
        cmd += ["--max-downloads", str(args.limit)]
    max_age = entry.get("max_age_days")
    if max_age and int(max_age) > 0:
        cutoff_date = (date.today() - timedelta(days=int(max_age))).strftime("%Y%m%d")
        cmd += ["--dateafter", cutoff_date]
    if args.dry_run:
        cmd += ["--simulate"]
        log.info(f"[DRY-RUN] yt-dlp simulate (native container): {entry['name']}")
    cmd.append(url)

    log.info(f"yt-dlp download (native container): {entry['name']} ({url})")
    result = subprocess.run(cmd)
    if result.returncode not in (0, 1, 101):  # 1=partial skip, 101=max-downloads hit
        log.error(f"yt-dlp failed (exit {result.returncode}): {entry['name']}")
        return []
    if args.dry_run:
        return []

    after = _snapshot_audio_files(out_dir)
    new_names = sorted(after - before)
    out_rows: list[dict] = []
    for name in new_names:
        path = out_dir / name
        vid_id = _extract_yt_id(path.stem)
        raw_id = vid_id or path.stem[-11:]
        if raw_id in known_ids:
            continue
        out_rows.append({
            "raw_id": raw_id,
            "wav_path": str(path),
            "source_url": f"https://www.youtube.com/watch?v={vid_id}" if vid_id else url,
            "title": path.stem,
            "pub_date": path.stem[:8] if path.stem[:8].isdigit() else "",
            "program": entry["name"],
            "source": source,
            "domain": entry.get("domain", ""),
            "style": entry.get("style", ""),
            "language": entry.get("language", "yue"),
            "duration_sec": None,   # filled in by ingest.probe (ffprobe)
            "sample_rate": None,    # filled in by ingest.probe (ffprobe)
            "downloaded_at": date.today().isoformat(),
        })
        known_ids.add(raw_id)

    return out_rows


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

async def run_ingest_download(
    *, source: str = "all", dry_run: bool = False, limit: int | None = None,
) -> dict:
    """Download phase — NEVER opens a DuckDB connection (see module
    docstring). Dedups against KNOWN_IDS_SNAPSHOT + STAGING_FILE, appends new
    rows to STAGING_FILE. Safe to run concurrently with any other
    catalog-writing node; run `ingest.commit` afterward (or anytime the
    writer lock is free) to land the staged rows in `raw_files`."""
    snapshot = load_known_ids_snapshot()
    targets = list(SOURCE_FILES) if source == "all" else [source]

    args = argparse.Namespace(dry_run=dry_run, limit=limit)
    total = 0
    for src_name in targets:
        entries = discover_active_entries(src_name)
        log.info(f"=== {src_name}: {len(entries)} active source(s) ===")
        known_ids = set(snapshot.get(src_name, [])) | load_staged_ids(src_name)

        for entry in entries:
            entry_type = entry.get("type", "")
            if not entry_type and entry.get("rss_url"):
                entry_type = "rss"

            if entry_type == "rss":
                rows = _download_rss_source(entry, known_ids, args)
            elif entry_type in ("channel", "playlist", "search"):
                rows = _download_youtube_source(entry, known_ids, args)
            else:
                log.info(f"Unknown type '{entry_type}' for {entry['name']}, skipping")
                rows = []

            if rows and not dry_run:
                append_staged_rows(rows)
            total += len(rows)
            log.info(f"  {entry['name']}: {len(rows)} new download(s)")

    log.info(
        f"DONE: {total} new raw file(s) staged to {STAGING_FILE.name} "
        f"(run `pipe run ingest.commit` to land them in the catalog)"
    )
    return {"staged": total}


# ---------------------------------------------------------------------------
# Commit phase — the ONLY part of this node that touches DuckDB. Short-lived:
# opens a connection, upserts everything staged, refreshes the dedup
# snapshot, archives the consumed staging file. Run whenever the writer lock
# is free (no need to coordinate with an in-progress download run — they
# never touch the same file at overlapping times: download only appends,
# commit only reads-then-archives-by-rename).
# ---------------------------------------------------------------------------

def _read_staged_rows() -> list[dict]:
    if not STAGING_FILE.exists():
        return []
    rows: list[dict] = []
    with open(STAGING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning(f"skipping malformed staged line: {exc}")
    return rows


async def run_ingest_commit(*, conn=None) -> dict:
    """conn: optional pre-opened DuckDB connection (or cursor) — pass one when
    running alongside other nodes under `pipe run-many` (see filter.py's
    run_filter_acoustic docstring for the rationale). Defaults to a fresh
    self-managed connect() for standalone `pipe run ingest.commit` usage."""
    from pipeline.catalog.catalog import connect, upsert_rows
    from pipeline.orchestrator.journal import new_run_id, record_batch

    staged = _read_staged_rows()
    # last-write-wins by raw_id — harmless if the same item was staged twice
    # across separate (e.g. resumed) download runs before a commit.
    by_id = {row["raw_id"]: row for row in staged if "raw_id" in row}
    rows = list(by_id.values())

    conn = conn or connect()
    run_id = new_run_id("ingest.download")
    if rows:
        upsert_rows(conn, "raw_files", rows, ["raw_id"])
        record_batch(conn, run_id, "ingest.download", list(by_id.keys()), "ok")
    write_known_ids_snapshot(conn)

    archived_to = None
    if STAGING_FILE.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived_to = METADATA_DIR / f"ingest_download_staging.committed-{stamp}.jsonl"
        STAGING_FILE.rename(archived_to)

    log.info(
        f"ingest.commit DONE: {len(rows)} row(s) landed in raw_files, "
        f"snapshot refreshed, run_id={run_id}"
        + (f", staging archived to {archived_to.name}" if archived_to else "")
    )
    return {
        "committed": len(rows),
        "run_id": run_id,
        "archived_to": str(archived_to) if archived_to else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all", choices=["rthk", "youtube", "podcast", "all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    parsed = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_ingest_download(
        source=parsed.source, dry_run=parsed.dry_run, limit=parsed.limit,
    ))
    print(f"\nDone: {result}")
    return 0


def main_commit() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_ingest_commit())
    print(f"\nDone: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
