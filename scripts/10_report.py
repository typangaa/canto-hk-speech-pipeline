#!/usr/bin/env python3
"""
scripts/10_report.py
Generate DATASET_REPORT.md with corpus statistics and acceptance criteria check.
Usage: python scripts/10_report.py [--manifest metadata/manifest.jsonl]
"""

import argparse
import json
import logging
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "metadata" / "logs" / "10_report.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

REPORT_PATH = ROOT / "metadata" / "DATASET_REPORT.md"

# Acceptance criteria from CLAUDE.md
ACCEPT = {
    "total_hours_min": 100.0,
    "speakers_min": 100,
    "sample_rate_required": 48000,
    "text_verified_pct": 1.0,
    "single_speaker_pct": 1.0,
    "dnsmos_p50_min": 3.0,
    "snr_p50_min": 25.0,
    "jyutping_valid_rate_min": 0.98,
    "sources_min": 3,
    "domains_min": 3,
    "windows_paths_max": 0,
}


def load_manifest(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * p / 100)
    return sorted_v[min(idx, len(sorted_v) - 1)]


def check(label: str, value, threshold, fmt="{:.1f}", good_if=">=") -> tuple[bool, str]:
    if good_if == ">=":
        ok = value >= threshold
    elif good_if == "<=":
        ok = value <= threshold
    elif good_if == "==":
        ok = value == threshold
    else:
        ok = False
    status = "PASS" if ok else "FAIL"
    val_str = fmt.format(value) if isinstance(value, float) else str(value)
    thr_str = fmt.format(threshold) if isinstance(threshold, float) else str(threshold)
    return ok, f"  [{status}] {label}: {val_str} (min {thr_str})" if good_if == ">=" \
           else f"  [{status}] {label}: {val_str} (required {thr_str})"


def generate_report(entries: list[dict]) -> str:
    n = len(entries)
    if n == 0:
        return "# DATASET_REPORT\n\nNo entries in manifest.\n"

    total_hours = sum(e["duration_sec"] for e in entries) / 3600
    speakers = set(e["speaker_id"] for e in entries)
    sources = set(e["source"] for e in entries)
    domains = set(e["domain"] for e in entries)

    dnsmos_vals = [e["dnsmos"] for e in entries if e.get("dnsmos", 0) > 0]
    snr_vals = [e["snr_db"] for e in entries if e.get("snr_db", 0) > 0]
    dur_vals = [e["duration_sec"] for e in entries]
    agr_vals = [e["asr_agreement"] for e in entries]

    n_verified = sum(1 for e in entries if e.get("text_verified"))
    n_48k = sum(1 for e in entries if e.get("sample_rate") == 48000)
    n_win = sum(1 for e in entries if not e.get("audio_path", "").startswith("/mnt/Drive3/"))

    source_counts = Counter(e["source"] for e in entries)
    domain_counts = Counter(e["domain"] for e in entries)
    gender_counts = Counter(e.get("gender", "unknown") for e in entries)

    lines = [
        f"# Dataset Report — canto-hk-speech-pipeline",
        f"Generated: {date.today()}",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total entries | {n:,} |",
        f"| Total hours | {total_hours:.1f}h |",
        f"| Unique speakers | {len(speakers)} |",
        f"| Unique sources | {len(sources)}: {', '.join(sorted(sources))} |",
        f"| Unique domains | {len(domains)}: {', '.join(sorted(domains))} |",
        f"| text_verified | {n_verified}/{n} ({100*n_verified/n:.1f}%) |",
        f"| 48 kHz WAV | {n_48k}/{n} |",
        f"| DNSMOS p50 | {percentile(dnsmos_vals, 50):.2f} |",
        f"| SNR p50 | {percentile(snr_vals, 50):.1f} dB |",
        f"| Duration p5/p50/p95 | {percentile(dur_vals,5):.1f}s / {percentile(dur_vals,50):.1f}s / {percentile(dur_vals,95):.1f}s |",
        f"| ASR agreement p50 | {percentile(agr_vals,50):.2f} |",
        f"",
        f"## Acceptance Criteria",
        f"",
    ]

    checks = [
        check("Total hours", total_hours, ACCEPT["total_hours_min"]),
        check("Unique speakers", len(speakers), ACCEPT["speakers_min"], fmt="{:d}"),
        check("text_verified pct", n_verified/n, ACCEPT["text_verified_pct"], fmt="{:.3f}"),
        check("48 kHz pct", n_48k/n, 1.0, fmt="{:.3f}"),
        check("DNSMOS p50", percentile(dnsmos_vals, 50), ACCEPT["dnsmos_p50_min"]),
        check("SNR p50 (dB)", percentile(snr_vals, 50), ACCEPT["snr_p50_min"]),
        check("Sources count", len(sources), ACCEPT["sources_min"], fmt="{:d}"),
        check("Domains count", len(domains), ACCEPT["domains_min"], fmt="{:d}"),
        check("Windows paths", n_win, ACCEPT["windows_paths_max"], fmt="{:d}", good_if="<="),
    ]
    all_pass = all(ok for ok, _ in checks)
    for _, msg in checks:
        lines.append(msg)

    lines += [
        f"",
        f"**Overall: {'READY FOR TRAINING' if all_pass else 'NOT YET READY'}**",
        f"",
        f"## Source Breakdown",
        f"",
        f"| Source | Segments | Hours |",
        f"|--------|----------|-------|",
    ]
    for src in sorted(source_counts, key=source_counts.get, reverse=True):
        src_entries = [e for e in entries if e["source"] == src]
        src_hrs = sum(e["duration_sec"] for e in src_entries) / 3600
        lines.append(f"| {src} | {source_counts[src]:,} | {src_hrs:.1f}h |")

    lines += [
        f"",
        f"## Domain Breakdown",
        f"",
        f"| Domain | Segments | % |",
        f"|--------|----------|---|",
    ]
    for dom in sorted(domain_counts, key=domain_counts.get, reverse=True):
        lines.append(f"| {dom} | {domain_counts[dom]:,} | {100*domain_counts[dom]/n:.1f}% |")

    lines += [
        f"",
        f"## Gender Distribution",
        f"",
    ]
    for g, c in gender_counts.most_common():
        lines.append(f"- {g}: {c} ({100*c/n:.1f}%)")

    lines += [
        f"",
        f"---",
        f"*Run `python scripts/10_report.py` to regenerate.*",
    ]

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "metadata" / "manifest.jsonl"))
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error(f"Manifest not found: {manifest_path}")
        log.error("Run 09_manifest.py first.")
        return

    entries = load_manifest(manifest_path)
    log.info(f"Loaded {len(entries)} entries")

    report_md = generate_report(entries)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report_md)

    print(report_md)
    print(f"Written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
