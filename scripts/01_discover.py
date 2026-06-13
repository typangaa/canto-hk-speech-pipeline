#!/usr/bin/env python3
"""
scripts/01_discover.py
Survey all configured sources and estimate available audio hours.
Usage: python scripts/01_discover.py [--source rthk|youtube|podcast|all] [--dry-run]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import feedparser
import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "metadata" / "logs" / "01_discover.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

SOURCES_DIR = ROOT / "sources"
DISCOVERY_LOG = ROOT / "metadata" / "discovery.json"


def load_yaml(path: Path) -> list[dict]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    # Some files have a top-level download_config key mixed with the list
    if isinstance(raw, dict):
        return []
    return []


def load_yaml_entries(path: Path) -> tuple[list[dict], dict]:
    """Return (entries list, download_config dict).

    Handles YAML files that mix a top-level list with a trailing 'download_config:'
    mapping key — invalid pure YAML but used in our source files.
    """
    with open(path) as f:
        content = f.read()

    # Split out any top-level download_config block (at column 0, no leading dash)
    # to produce valid YAML for the list portion.
    import re as _re
    config_match = _re.search(r"^\s*download_config\s*:", content, _re.MULTILINE)
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


def probe_rss(url: str, timeout: int = 15) -> dict:
    """Fetch an RSS feed and extract episode count + duration estimate."""
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "canto-hk-speech-pipeline/0.1"})
        if not feed.entries:
            err = str(feed.bozo_exception) if feed.bozo else "empty feed"
            return {"ok": False, "error": err}
        n = len(feed.entries)
        title = feed.feed.get("title", "")
        total_secs = 0
        known = 0
        for e in feed.entries:
            dur = e.get("itunes_duration", "")
            if isinstance(dur, int):
                total_secs += dur
                known += 1
            elif isinstance(dur, str) and dur:
                parts = dur.split(":")
                try:
                    secs = sum(int(p) * 60 ** i for i, p in enumerate(reversed(parts)))
                    total_secs += secs
                    known += 1
                except ValueError:
                    pass
        avg_secs = total_secs / known if known else 0
        est_hours = n * avg_secs / 3600 if avg_secs else 0
        return {
            "ok": True,
            "title": title,
            "episodes": n,
            "known_duration": known,
            "avg_min": round(avg_secs / 60, 1),
            "est_hours": round(est_hours, 1),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def check_url(url: str, timeout: int = 10) -> bool:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True,
                          headers={"User-Agent": "canto-hk-speech-pipeline/0.1"})
        return r.status_code < 400
    except Exception:
        return False


def discover_source_file(path: Path, args: argparse.Namespace) -> list[dict]:
    entries, _ = load_yaml_entries(path)
    results = []
    for e in entries:
        if e.get("status") in ("skip", "done") or e.get("priority") in ("skip",):
            continue
        url = e.get("url") or e.get("rss_url", "")
        src_type = e.get("type") or ("rss" if e.get("rss_url") else "unknown")
        name = e.get("name", "?")

        row = {
            "name": name,
            "source": e.get("source", ""),
            "type": src_type,
            "domain": e.get("domain", ""),
            "priority": e.get("priority", ""),
            "status": e.get("status", ""),
            "estimated_hours": e.get("estimated_hours", 0),
            "url": url,
        }

        if args.dry_run:
            row["probe"] = {"ok": "skipped (dry-run)"}
            results.append(row)
            continue

        if src_type == "rss" and url and not url.startswith(("PLACEHOLDER", "SKIP", "SEARCH")):
            log.info(f"  Probing RSS: {name}")
            row["probe"] = probe_rss(url)
        elif src_type in ("channel", "playlist", "search"):
            accessible = check_url(url) if url and not url.startswith(("SKIP", "SEARCH")) else None
            row["probe"] = {"ok": accessible, "note": "YouTube URL — requires yt-dlp to enumerate"}
        else:
            row["probe"] = {"ok": None, "note": "placeholder / manual"}

        results.append(row)
    return results


def print_table(results: list[dict]) -> None:
    print()
    print(f"{'Name':<35} {'Type':<10} {'Domain':<13} {'Priority':<9} {'Est h':>6} {'Probe'}")
    print("-" * 100)
    for r in results:
        probe = r.get("probe", {})
        if isinstance(probe, dict):
            if probe.get("ok") is True:
                ep = probe.get("episodes", "")
                hrs = probe.get("est_hours", "")
                note = f"OK  {ep} eps ~{hrs}h" if ep else "OK"
            elif probe.get("ok") is False:
                note = f"FAIL {probe.get('error','')[:30]}"
            else:
                note = probe.get("note", str(probe.get("ok", "")))
        else:
            note = str(probe)
        print(
            f"{r['name']:<35} {r['type']:<10} {r['domain']:<13} {r['priority']:<9} "
            f"{r['estimated_hours']:>6} {note}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="all",
                        choices=["rthk", "youtube", "podcast", "all"])
    parser.add_argument("--dry-run", action="store_true")
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

    all_results = []
    for src_name, src_file in targets:
        if not src_file.exists():
            log.warning(f"Source file not found: {src_file}")
            continue
        log.info(f"Scanning {src_file.name} ...")
        results = discover_source_file(src_file, args)
        log.info(f"  Found {len(results)} active entries")
        all_results.extend(results)

    print_table(all_results)

    rss_ok = [r for r in all_results if r.get("probe", {}).get("ok") is True]
    total_est = sum(r.get("estimated_hours", 0) for r in all_results)
    rss_actual = sum(r["probe"].get("est_hours", 0) for r in rss_ok)

    print(f"Summary: {len(all_results)} active sources | "
          f"Config-estimated total: ~{total_est}h | "
          f"RSS-probed actual: ~{rss_actual:.1f}h from {len(rss_ok)} feeds")
    print()

    if not args.dry_run:
        DISCOVERY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(DISCOVERY_LOG, "w") as f:
            json.dump({"sources": all_results}, f, ensure_ascii=False, indent=2)
        log.info(f"Discovery log written: {DISCOVERY_LOG}")

    log.info(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
