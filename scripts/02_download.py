#!/usr/bin/env python3
"""
scripts/02_download.py
Download audio from configured sources (RSS podcasts + YouTube).
Usage: python scripts/02_download.py --source [rthk|youtube|podcast|all] [--dry-run] [--limit N]

Output: data/raw/{source}/{date}_{slug}_{id}.{wav|webm}
        metadata/downloaded.jsonl  (one line per completed download)

All audio is saved as 48 kHz mono WAV master (KNOWN_ISSUES §11).
"""

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import feedparser
import requests
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "metadata" / "logs" / "02_download.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

SOURCES_DIR = ROOT / "sources"
RAW_DIR = ROOT / "data" / "raw"
DOWNLOADED_LOG = ROOT / "metadata" / "downloaded.jsonl"
TARGET_SR = 48000


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_-]+", "_", text).strip("_")[:40]


def load_yaml_entries(path: Path) -> tuple[list[dict], dict]:
    with open(path) as f:
        content = f.read()

    config_match = re.search(r"^\s*download_config\s*:", content, re.MULTILINE)
    config: dict = {}
    list_content = content

    if config_match:
        list_content = content[: config_match.start()]
        config_yaml = content[config_match.start() :]
        try:
            parsed_cfg = yaml.safe_load(config_yaml)
            if isinstance(parsed_cfg, dict):
                config = parsed_cfg.get("download_config", parsed_cfg) or {}
        except Exception:
            pass

    try:
        raw = yaml.safe_load(list_content)
    except yaml.YAMLError:
        return [], config

    if isinstance(raw, list):
        entries = [e for e in raw if isinstance(e, dict) and "name" in e]
    elif isinstance(raw, dict):
        entries = []
        for k, v in raw.items():
            if isinstance(v, list):
                entries.extend([e for e in v if isinstance(e, dict) and "name" in e])
    else:
        entries = []

    return entries, config


def load_downloaded_ids() -> set[str]:
    if not DOWNLOADED_LOG.exists():
        return set()
    ids = set()
    with open(DOWNLOADED_LOG) as f:
        for line in f:
            try:
                ids.add(json.loads(line.strip())["id"])
            except Exception:
                pass
    return ids


def append_downloaded(record: dict) -> None:
    DOWNLOADED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DOWNLOADED_LOG, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert_to_wav48(src: Path, dst: Path) -> bool:
    """Convert any audio file to 48 kHz mono PCM WAV using ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-ar", "48000", "-ac", "1", "-sample_fmt", "s16",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        log.error(f"ffmpeg failed: {result.stderr.decode()}")
        return False
    # Verify output
    info = sf.info(str(dst))
    if info.samplerate != TARGET_SR:
        log.error(f"Sample rate mismatch after convert: {info.samplerate}")
        return False
    return True


def download_rss_episode(episode: dict, out_dir: Path, dry_run: bool) -> Optional[dict]:
    """Download one podcast RSS episode and convert to 48 kHz WAV."""
    audio_url = None
    # RTHK RSS uses video/mp4 enclosures even for podcast audio
    for enc in episode.get("enclosures", []):
        enc_type = enc.get("type", "")
        if "audio" in enc_type or "video" in enc_type or "mp4" in enc_type:
            audio_url = enc.get("href") or enc.get("url")
            break
    if not audio_url:
        for link in episode.get("links", []):
            href = link.get("href", "")
            ltype = link.get("type", "")
            if "audio" in ltype or "video" in ltype or href.endswith((".mp3", ".m4a", ".aac", ".mp4")):
                audio_url = href
                break
    if not audio_url:
        log.warning(f"No audio URL for episode: {episode.get('title', '?')}")
        return None

    pub = episode.get("published_parsed")
    pub_date = date(*pub[:3]).strftime("%Y%m%d") if pub else "unknown"
    title_slug = slugify(episode.get("title", "episode"))
    ep_id = hashlib.md5(audio_url.encode()).hexdigest()[:8]
    ext = Path(audio_url.split("?")[0]).suffix or ".mp3"
    raw_name = f"{pub_date}_{title_slug}_{ep_id}{ext}"
    wav_name = f"{pub_date}_{title_slug}_{ep_id}.wav"

    raw_path = out_dir / raw_name
    wav_path = out_dir / wav_name

    if wav_path.exists():
        log.info(f"Skip (exists): {wav_name}")
        return None  # already done

    if dry_run:
        log.info(f"[DRY-RUN] Would download: {wav_name}")
        return None

    log.info(f"Downloading: {audio_url}")
    try:
        r = requests.get(audio_url, stream=True, timeout=120,
                         headers={"User-Agent": "canto-hk-speech-pipeline/0.1"})
        r.raise_for_status()
        with open(raw_path, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
    except Exception as exc:
        log.error(f"Download failed: {exc}")
        raw_path.unlink(missing_ok=True)
        return None

    log.info(f"Converting to 48 kHz WAV: {wav_name}")
    ok = convert_to_wav48(raw_path, wav_path)
    raw_path.unlink(missing_ok=True)  # remove temp file
    if not ok:
        wav_path.unlink(missing_ok=True)
        return None

    info = sf.info(str(wav_path))
    return {
        "id": ep_id,
        "wav_path": str(wav_path),
        "source_url": audio_url,
        "title": episode.get("title", ""),
        "pub_date": pub_date,
        "duration_sec": round(info.duration, 1),
        "sample_rate": info.samplerate,
        "downloaded_at": str(date.today()),
    }


def download_rss_source(entry: dict, done_ids: set[str], args: argparse.Namespace) -> int:
    url = entry.get("url") or entry.get("rss_url", "")
    if not url or url.startswith(("PLACEHOLDER", "SKIP", "SEARCH")):
        log.warning(f"Skipping {entry['name']}: no valid URL")
        return 0

    source = entry.get("source", "podcast")
    out_dir = RAW_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Fetching RSS: {entry['name']} → {url}")
    feed = feedparser.parse(url, request_headers={"User-Agent": "canto-hk-speech-pipeline/0.1"})
    if not feed.entries:
        log.warning(f"Empty feed: {url}")
        return 0

    episodes = feed.entries
    # max_age_days from source config; 0 means no age limit (useful for archived feeds like RTHK)
    max_age = int(entry.get("max_age_days", 365))
    cutoff = None
    if max_age > 0:
        from datetime import timedelta
        cutoff = date.today() - timedelta(days=max_age)

    n = 0
    for ep in episodes:
        ep_id = hashlib.md5((ep.get("enclosures") and ep["enclosures"][0].get("href", "")
                              or ep.get("id", ep.get("title", ""))).encode()).hexdigest()[:8]
        if ep_id in done_ids:
            log.info(f"Skip (already downloaded): {ep.get('title', ep_id)}")
            continue

        pub = ep.get("published_parsed")
        if pub and cutoff:
            ep_date = date(*pub[:3])
            if ep_date < cutoff:
                log.info(f"Skip (too old {ep_date}): {ep.get('title', '')}")
                continue

        if args.limit and n >= args.limit:
            log.info(f"Reached --limit {args.limit}, stopping")
            break

        record = download_rss_episode(ep, out_dir, args.dry_run)
        if record:
            record.update({
                "program": entry["name"],
                "source": source,
                "domain": entry.get("domain", ""),
                "style": entry.get("style", ""),
                "language": entry.get("language", "yue"),
            })
            append_downloaded(record)
            done_ids.add(ep_id)
            n += 1
        time.sleep(2)

    return n


def _extract_yt_id(stem: str) -> Optional[str]:
    """Extract 11-char YouTube video ID from last 11 chars of a filename stem."""
    import re
    tail = stem[-11:]
    return tail if re.match(r"^[A-Za-z0-9_-]{11}$", tail) else None


def download_youtube_source(entry: dict, done_ids: set[str], args: argparse.Namespace) -> int:
    url = entry.get("url", "")
    if not url or url.startswith(("SKIP", "SEARCH", "PLACEHOLDER")):
        log.warning(f"Skipping {entry['name']}: no valid URL")
        return 0

    source = entry.get("source", "youtube")
    out_dir = RAW_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)

    name_slug = slugify(entry["name"])
    outtmpl = str(out_dir / f"%(upload_date)s_{name_slug}_%(id)s.%(ext)s")

    # Snapshot before download so we can record only newly added files
    before = {p.name for p in out_dir.glob("*.wav")}

    cmd = [
        "yt-dlp",
        "--no-playlist" if entry.get("type") not in ("playlist", "channel") else "--yes-playlist",
        "--format", "bestaudio/best",
        "--extract-audio", "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 48000 -ac 1 -sample_fmt s16",
        "--restrict-filenames",
        "--sleep-interval", "3", "--max-sleep-interval", "8",
        "--sleep-requests", "2",      # 2s between ALL HTTP requests (prevents rate-limit)
        "--retries", "5",
        "--retry-sleep", "exp=1:60",  # exponential backoff on retries (1s → 60s cap)
        "--output", outtmpl,
        "--download-archive", str(ROOT / "metadata" / "yt_archive.txt"),
        "--match-filter", "duration > 120 & duration < 14400",
        "--max-filesize", "500M",
        "--no-warnings",
        "--quiet",
        "--ignore-errors",  # skip private/unavailable videos, continue rest of playlist
        "--no-part",        # write directly to final filename, avoids .part rename errors on symlinked paths
    ]

    if args.limit:
        cmd += ["--max-downloads", str(args.limit)]

    # Per-entry age filter: --dateafter YYYYMMDD
    max_age = entry.get("max_age_days")
    if max_age and int(max_age) > 0:
        from datetime import timedelta
        cutoff_date = (date.today() - timedelta(days=int(max_age))).strftime("%Y%m%d")
        cmd += ["--dateafter", cutoff_date]

    if args.dry_run:
        cmd += ["--simulate"]
        log.info(f"[DRY-RUN] yt-dlp simulate: {entry['name']}")

    cmd.append(url)
    log.info(f"yt-dlp download: {entry['name']} ({url})")

    result = subprocess.run(cmd)
    if result.returncode not in (0, 1, 101):  # 1 = partial (private/unavailable vids skipped); 101 = max-downloads
        log.error(f"yt-dlp failed (exit {result.returncode}): {entry['name']}")
        return 0
    if result.returncode == 1:
        log.warning(f"yt-dlp partial (some videos skipped/private): {entry['name']}")

    log.info(f"yt-dlp done for {entry['name']}")

    if args.dry_run:
        return 0

    # Record newly downloaded WAV files in downloaded.jsonl
    after = {p for p in out_dir.glob("*.wav")}
    new_wavs = [p for p in after if p.name not in before]
    for wav_path in sorted(new_wavs):
        try:
            info = sf.info(str(wav_path))
            vid_id = _extract_yt_id(wav_path.stem)
            record = {
                "id": vid_id or wav_path.stem[-8:],
                "wav_path": str(wav_path),
                "source_url": f"https://www.youtube.com/watch?v={vid_id}" if vid_id else url,
                "title": wav_path.stem,
                "pub_date": wav_path.stem[:8] if wav_path.stem[:8].isdigit() else "",
                "duration_sec": round(info.duration, 1),
                "sample_rate": info.samplerate,
                "downloaded_at": str(date.today()),
                "program": entry["name"],
                "source": source,
                "domain": entry.get("domain", ""),
                "style": entry.get("style", ""),
                "language": entry.get("language", "yue"),
            }
            append_downloaded(record)
            if vid_id:
                done_ids.add(vid_id)
        except Exception as exc:
            log.warning(f"Could not record {wav_path.name}: {exc}")

    return len(new_wavs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all",
                        choices=["rthk", "youtube", "podcast", "all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max episodes/videos per source entry")
    args = parser.parse_args()

    file_map = {
        "rthk":    SOURCES_DIR / "rthk_sources.yaml",
        "youtube": SOURCES_DIR / "youtube_channels.yaml",
        "podcast": SOURCES_DIR / "podcast_sources.yaml",
    }

    if args.source == "all":
        targets = list(file_map.items())
    else:
        targets = [(args.source, file_map[args.source])]

    done_ids = load_downloaded_ids()
    log.info(f"Already downloaded: {len(done_ids)} entries")

    total = 0
    for src_name, src_file in targets:
        if not src_file.exists():
            log.warning(f"Source file not found: {src_file}")
            continue

        entries, _ = load_yaml_entries(src_file)
        active = [e for e in entries
                  if e.get("status") not in ("skip", "done", "paused")
                  and e.get("priority") not in ("skip",)]
        log.info(f"\n=== {src_file.name}: {len(active)} active sources ===")

        for entry in active:
            entry_type = entry.get("type", "")
            # Infer type from fields when not explicit
            if not entry_type and entry.get("rss_url"):
                entry_type = "rss"
            if entry_type == "rss":
                n = download_rss_source(entry, done_ids, args)
            elif entry_type in ("channel", "playlist", "search"):
                n = download_youtube_source(entry, done_ids, args)
            else:
                log.info(f"Unknown type '{entry_type}' for {entry['name']}, skipping")
                n = 0
            total += n
            log.info(f"  {entry['name']}: {n} new downloads")

    print(f"\nDone: {total} new files downloaded")
    print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
