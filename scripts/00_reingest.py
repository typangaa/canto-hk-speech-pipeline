#!/usr/bin/env python3
"""
scripts/00_reingest.py
Re-download legacy audio from cantonese-tts-old at 48 kHz using YouTube video
IDs embedded in the original filenames. Legacy files are 22050 Hz — unusable
as pipeline masters. This script re-fetches them at best available quality.

Usage:
  python scripts/00_reingest.py --list-ids --category tier2_tvb_news
  python scripts/00_reingest.py --category tier2_tvb_news --dry-run
  python scripts/00_reingest.py --category tier2_tvb_news
  python scripts/00_reingest.py --category all

Category sizes (video count):
  rthk_documentaries  17  鏗鏘集 (overlap with RSS — deduped via yt_archive.txt)
  rthk_culture        88  優遊記 and cultural shows
  rthk_heritage      101  精靈一點 and heritage programmes
  rthk_news          410  千禧年代 and news/talk radio
  tier2_tvb_news     535  TVB news clips (formal Cantonese, clear speech)
  rthk              1779  創科新里程 tech documentaries
  tier2_legco        262  立法會 (skipped by default — parliament noise, poor TTS)
"""

import argparse
import json
import logging
import re
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "metadata" / "logs" / "00_reingest.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

LEGACY_BASE = Path("/mnt/Drive3/Development/AI-ML/cantonese-tts-old/data/raw")
RAW_DIR = ROOT / "data" / "raw"
DOWNLOADED_LOG = ROOT / "metadata" / "downloaded.jsonl"
YT_ARCHIVE = ROOT / "metadata" / "yt_archive.txt"
TARGET_SR = 48000

# Map legacy category → pipeline metadata
CATEGORY_META = {
    "rthk_documentaries": {
        "source": "rthk", "out_subdir": "rthk",
        "domain": "documentary", "style": "interview",
    },
    "rthk_culture": {
        "source": "rthk", "out_subdir": "rthk",
        "domain": "documentary", "style": "casual",
    },
    "rthk_heritage": {
        "source": "rthk", "out_subdir": "rthk",
        "domain": "educational", "style": "interview",
    },
    "rthk_news": {
        "source": "rthk", "out_subdir": "rthk",
        "domain": "talk_show", "style": "casual",
    },
    "tier2_tvb_news": {
        "source": "youtube", "out_subdir": "youtube",
        "domain": "news", "style": "formal",
    },
    "rthk": {
        "source": "rthk", "out_subdir": "rthk",
        "domain": "documentary", "style": "narration",
    },
    "tier2_legco": {
        "source": "youtube", "out_subdir": "youtube",
        "domain": "other", "style": "formal",
    },
}

YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Categories skipped when --category all is used
SKIP_IN_ALL = {"tier2_legco"}


def extract_video_id(stem: str) -> str | None:
    """Extract 11-char YouTube video ID from last 11 chars of a filename stem."""
    if len(stem) < 11:
        return None
    tail = stem[-11:]
    return tail if YT_ID_RE.match(tail) else None


def extract_program_hint(stem: str) -> str:
    """Heuristically extract programme name segment from filename stem."""
    # Remove video ID + preceding underscore
    body = stem[:-12] if len(stem) > 12 and stem[-12] == "_" else stem[:-11]
    # Remove leading YYYYMMDD_ date prefix
    body = re.sub(r"^\d{8}_", "", body)
    # Remove category prefix like "legco_", "tvbnews_"
    body = re.sub(r"^[a-z]+_", "", body)
    # Take first meaningful segment
    parts = [p.strip() for p in re.split(r"[_：:｜—]", body) if p.strip()]
    return parts[0][:30] if parts else ""


def collect_video_ids(category: str) -> list[tuple[str, str]]:
    """Return list of (video_id, program_hint) for a legacy category directory."""
    cat_dir = LEGACY_BASE / category
    if not cat_dir.exists():
        log.error(f"Legacy directory not found: {cat_dir}")
        return []
    results = []
    for wav in sorted(cat_dir.glob("*.wav")):
        vid_id = extract_video_id(wav.stem)
        if vid_id:
            results.append((vid_id, extract_program_hint(wav.stem)))
        else:
            log.debug(f"No video ID in: {wav.name[:60]}")
    return results


def load_yt_archive_ids() -> set[str]:
    """IDs already downloaded per yt-dlp archive — skip these."""
    ids: set[str] = set()
    if YT_ARCHIVE.exists():
        for line in YT_ARCHIVE.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                ids.add(parts[1])
    return ids


def load_downloaded_paths() -> set[str]:
    """WAV paths already in downloaded.jsonl."""
    paths: set[str] = set()
    if DOWNLOADED_LOG.exists():
        with open(DOWNLOADED_LOG) as f:
            for line in f:
                try:
                    paths.add(json.loads(line.strip()).get("wav_path", ""))
                except Exception:
                    pass
    return paths


def download_video_batch(
    video_ids: list[str],
    out_dir: Path,
    name_slug: str,
    dry_run: bool,
) -> set[str]:
    """Download a batch of YouTube video IDs at 48 kHz WAV.

    Returns the set of video IDs that yt-dlp processed (downloaded or skipped-as-archived).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(out_dir / f"%(upload_date)s_{name_slug}_%(id)s.%(ext)s")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        tf.write("\n".join(f"https://www.youtube.com/watch?v={v}" for v in video_ids) + "\n")
        url_file = tf.name

    cmd = [
        "yt-dlp",
        "--batch-file", url_file,
        "--format", "bestaudio/best",
        "--extract-audio", "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 48000 -ac 1 -sample_fmt s16",
        "--restrict-filenames",
        "--sleep-interval", "3", "--max-sleep-interval", "8",
        "--retries", "5",
        "--output", outtmpl,
        "--download-archive", str(YT_ARCHIVE),
        "--match-filter", "duration > 60",
        "--no-warnings",
    ]
    if dry_run:
        cmd += ["--simulate", "--quiet"]
        log.info(f"[DRY-RUN] Would attempt {len(video_ids)} videos")
    else:
        cmd += ["--quiet"]
        log.info(f"Downloading {len(video_ids)} videos → {out_dir}")

    subprocess.run(cmd)
    Path(url_file).unlink(missing_ok=True)
    return set(video_ids)


def record_new_downloads(
    out_dir: Path,
    batch_ids: set[str],
    category: str,
    meta: dict,
    existing_paths: set[str],
) -> int:
    """Scan out_dir for new WAV files from this batch and append to downloaded.jsonl."""
    n = 0
    for wav_path in sorted(out_dir.glob("*.wav")):
        if str(wav_path) in existing_paths:
            continue
        vid_id = extract_video_id(wav_path.stem)
        if vid_id not in batch_ids:
            continue
        try:
            info = sf.info(str(wav_path))
            if info.samplerate != TARGET_SR:
                log.warning(f"Unexpected {info.samplerate} Hz (expected 48000): {wav_path.name}")
                continue
            record = {
                "id": vid_id,
                "wav_path": str(wav_path),
                "source_url": f"https://www.youtube.com/watch?v={vid_id}",
                "title": wav_path.stem,
                "pub_date": wav_path.stem[:8] if wav_path.stem[:8].isdigit() else "",
                "duration_sec": round(info.duration, 1),
                "sample_rate": info.samplerate,
                "downloaded_at": str(date.today()),
                "source": meta["source"],
                "domain": meta["domain"],
                "style": meta["style"],
                "legacy_category": category,
            }
            DOWNLOADED_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(DOWNLOADED_LOG, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing_paths.add(str(wav_path))
            n += 1
        except Exception as exc:
            log.error(f"Error recording {wav_path.name}: {exc}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category", default="tier2_tvb_news",
        choices=list(CATEGORY_META.keys()) + ["all"],
        help="Which legacy category to re-download (default: tier2_tvb_news)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate downloads; do not write files")
    parser.add_argument("--list-ids", action="store_true",
                        help="Print video IDs and exit; no downloads")
    args = parser.parse_args()

    if args.category == "all":
        categories = [c for c in CATEGORY_META if c not in SKIP_IN_ALL]
        log.info(f"Processing all categories (skipping: {', '.join(SKIP_IN_ALL)})")
    else:
        categories = [args.category]

    archive_ids = load_yt_archive_ids()
    log.info(f"yt-dlp archive: {len(archive_ids)} IDs already downloaded")

    if not args.dry_run and not args.list_ids:
        existing_paths = load_downloaded_paths()
    else:
        existing_paths = set()

    total_new = 0

    for category in categories:
        meta = CATEGORY_META[category]
        all_ids = collect_video_ids(category)
        new_ids = [(vid, prog) for vid, prog in all_ids if vid not in archive_ids]

        log.info(
            f"\n=== {category}: {len(all_ids)} files in legacy, "
            f"{len(new_ids)} not yet downloaded ==="
        )

        if args.list_ids:
            for vid, prog in new_ids[:30]:
                print(f"  {vid}  {prog}")
            if len(new_ids) > 30:
                print(f"  ... and {len(new_ids) - 30} more")
            continue

        if not new_ids:
            log.info(f"  All {category} videos already in archive — skipping")
            continue

        out_dir = RAW_DIR / meta["out_subdir"]
        name_slug = category.replace("_", "-")

        batch_ids = download_video_batch(
            [vid for vid, _ in new_ids],
            out_dir,
            name_slug,
            args.dry_run,
        )

        if not args.dry_run:
            recorded = record_new_downloads(out_dir, batch_ids, category, meta, existing_paths)
            log.info(f"  Recorded {recorded} new entries in downloaded.jsonl")
            total_new += recorded

    if not args.list_ids:
        print(f"\nDone: {total_new} new files recorded in downloaded.jsonl")
        print(f"Log: {LOG_PATH}")
        if not args.dry_run:
            print(f"Next: run 03_segment.py --source rthk (or youtube) to segment new audio")


if __name__ == "__main__":
    main()
