"""
pipeline/nodes/report.py
report.build DAG node -- ports scripts/10_report.py's dataset-statistics report
onto a live catalog read, replacing its two stale inputs: (1) a hand-maintained
ACCEPT dict that drifted out of sync with CLAUDE.md's Acceptance Criteria table
(DNSMOS/SNR bars were tightened since the legacy script was written), and
(2) reading metadata/manifest.jsonl off disk, which only reflects whatever the
LAST `manifest.export` run captured -- stale the moment any catalog table
changes after that export.

This node instead calls pipeline.nodes.manifest.run_manifest_build() directly,
which re-runs the exact same "manifest-eligible" catalog join fresh, in memory,
on every call -- so the numbers here are always live, never off by however
long it has been since the last manifest.export. No file is read off disk
except config; no catalog writer connection is ever opened (run_manifest_build
uses connect_ro() internally) -- this node is read-only against the catalog
and only ever writes metadata/DATASET_REPORT.md.

12 acceptance criteria (CLAUDE.md "Acceptance Criteria" section) are checked
here, not the legacy script's 9 -- the legacy ACCEPT dict defined
single_speaker_pct and jyutping_valid_rate_min but never actually added either
to its own `checks` list (a real bug: 2 of the 11 declared thresholds were
silently never evaluated). This version checks all 12 named in CLAUDE.md, with
two deliberately reported as non-boolean/non-PASS-FAIL "honest" notes rather
than faked results:

  - text_verified: the current manifest-eligible pool intentionally includes
    non-human-verified statistical tiers (auto_gold/silver/bronze -- see
    pipeline/nodes/tier.py's module docstring), so this criterion legitimately
    reads well under 100% right now. It is still reported as a real
    percentage with a PASS/FAIL line against the 100% bar (expected to FAIL
    until the corpus is fully hand-verified, or the report is generated with
    min_tier='gold') -- not fudged to pass.
  - single-speaker segments: guaranteed by pipeline DESIGN (VAD-cut
    segmentation always runs *within* a single diarization-detected speaker
    turn -- CLAUDE.md hard constraint #5), not re-verified per-segment from
    catalog data at report time -- there is no per-segment "how many speakers
    are actually in this clip" column to check independently. Reported as a
    design-guarantee note, not a PASS/FAIL line, so it is never confused with
    the 11 criteria that ARE computed from live data.

Jyutping valid-rate aggregation: computed as
(total valid tokens across every manifest-eligible entry) / (total tokens
across every manifest-eligible entry) -- a token-weighted corpus-wide rate,
not a mean-of-per-entry-rates (which would let many tiny entries skew the
average away from where most of the actual syllables live). Each entry's
`jyutping` string is re-split and re-validated here against the same
`^[a-z]+[1-6]$` regex as pipeline/nodes/g2p.py's JYUTPING_TOKEN (duplicated
here as a plain constant, not imported, specifically to avoid loading
g2p.py's module-level `canto_hk_g2p.Pipeline()` instantiation -- a real,
non-trivial model load -- just to generate a report).

min_tier passthrough: matches manifest.build/manifest.export's existing
--min-tier convention (pipeline/nodes/manifest.py's TIER_PRECEDENCE /
_tiers_at_or_above()) -- e.g. a caller can run report.build with
min_tier='gold' to see whether the STRICTLY human-verified subset alone would
already clear the acceptance bar, without needing a separate manifest export
first. Defaults to None (the full manifest-eligible pool), matching
DATASET_REPORT.md's historical scope.

Logging: matches every other ported node's cli.py handler -- a bare
`logging.basicConfig(...)` call in cmd_run_report_build, no dedicated
FileHandler (see manifest.py's / label_store.py's cli.py handlers for the
identical pattern) -- operators redirect stdout to metadata/logs/*.log
themselves for persistence, the same as every other `pipe run` invocation in
this project.
"""

import logging
import re
import time
from collections import Counter
from datetime import date

log = logging.getLogger(__name__)

# Duplicated from pipeline/nodes/g2p.py's JYUTPING_TOKEN (CLAUDE.md hard
# constraint #8) rather than imported -- see module docstring above.
JYUTPING_TOKEN = re.compile(r"^[a-z]+[1-6]$")

# Acceptance criteria thresholds -- CLAUDE.md's "Acceptance Criteria" table is
# the single source of truth; keep this dict in sync with it, not the other
# way around (the legacy scripts/10_report.py's ACCEPT dict is what drifted
# out of sync with CLAUDE.md and is why this node exists).
ACCEPT = {
    "total_hours_min": 100.0,
    "speakers_min": 100,
    "sample_rate_pct_min": 1.0,
    "text_verified_pct_min": 1.0,
    "dnsmos_p50_min": 3.2,
    "snr_p50_min": 30.0,
    "jyutping_valid_rate_min": 0.99,
    "sources_min": 3,
    "domains_min": 3,
    "duration_in_range_pct_min": 1.0,
    "windows_paths_max": 0,
}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * p / 100)
    return sorted_v[min(idx, len(sorted_v) - 1)]


def _check(label: str, value, threshold, *, fmt: str = "{:.1f}", good_if: str = ">=") -> dict:
    """Returns a dict (not the legacy tuple) so generate_report can both render
    a markdown line and hand back a structured pass/fail list to callers."""
    if good_if == ">=":
        ok = value >= threshold
    elif good_if == "<=":
        ok = value <= threshold
    else:
        raise ValueError(f"unsupported good_if: {good_if!r}")

    status = "PASS" if ok else "FAIL"
    val_str = fmt.format(value) if isinstance(value, float) else str(value)
    thr_str = fmt.format(threshold) if isinstance(threshold, float) else str(threshold)
    qualifier = "min" if good_if == ">=" else "max"
    line = f"- [{status}] {label}: {val_str} ({qualifier} {thr_str})"
    return {"label": label, "value": value, "threshold": threshold, "passed": ok, "line": line}


def _jyutping_valid_rate(entries: list[dict]) -> tuple[float, int, int]:
    """Token-weighted corpus-wide valid rate -- see module docstring. Returns
    (rate, valid_tokens, total_tokens); rate is 0.0 (not NaN/ZeroDivisionError)
    when there are no tokens at all."""
    total = 0
    valid = 0
    for e in entries:
        jyutping = e.get("jyutping") or ""
        tokens = jyutping.split()
        total += len(tokens)
        valid += sum(1 for t in tokens if JYUTPING_TOKEN.match(t))
    rate = valid / total if total > 0 else 0.0
    return rate, valid, total


def generate_report(entries: list[dict], *, min_tier: str | None = None) -> tuple[str, list[dict]]:
    """Pure function: manifest-eligible entries -> (markdown string, criteria list).
    criteria is the list of dicts _check() produces for the 11 boolean
    acceptance criteria (single-speaker is reported separately as a design
    note, not included in this list -- see module docstring)."""
    n = len(entries)
    generated_note = (
        f"*Generated by `report.build` (pipeline/nodes/report.py) reading the LIVE catalog "
        f"(`metadata/corpus.duckdb`) via `manifest.build`'s eligibility join -- not a stale "
        f"`metadata/manifest.jsonl` file. Supersedes the retired `scripts/10_report.py`.*"
    )
    if min_tier is not None:
        generated_note += f"\n\n*Scoped to `min_tier={min_tier}` (not the full manifest-eligible pool).*"

    if n == 0:
        md = (
            f"# Dataset Report -- canto-hk-speech-pipeline\n"
            f"Generated: {date.today()}\n\n"
            f"{generated_note}\n\n"
            f"No manifest-eligible entries found in the catalog.\n"
        )
        return md, []

    total_hours = sum(e["duration_sec"] for e in entries) / 3600
    speakers = {e["speaker_id"] for e in entries}
    sources = {e["source"] for e in entries}
    domains = {e["domain"] for e in entries}

    dnsmos_vals = [e["dnsmos"] for e in entries if e.get("dnsmos", 0) > 0]
    snr_vals = [e["snr_db"] for e in entries if e.get("snr_db", 0) > 0]
    dur_vals = [e["duration_sec"] for e in entries]
    agr_vals = [e["asr_agreement"] for e in entries]

    n_verified = sum(1 for e in entries if e.get("text_verified"))
    n_48k = sum(1 for e in entries if e.get("sample_rate") == 48000)
    n_in_range = sum(1 for e in entries if 3.0 <= e["duration_sec"] <= 20.0)
    # Genuine Windows-style paths (KNOWN_ISSUES.md §3) -- not a check against
    # any particular ext4 drive letter, since which Drive hosts segments
    # changes across storage migrations (config/storage_layout.py).
    n_win = sum(
        1 for e in entries
        if e.get("audio_path", "").lower().startswith(("/mnt/c/", "/mnt/d/"))
    )
    jyutping_rate, jyutping_valid_tok, jyutping_total_tok = _jyutping_valid_rate(entries)

    source_counts = Counter(e["source"] for e in entries)
    domain_counts = Counter(e["domain"] for e in entries)
    gender_counts = Counter(e.get("gender", "unknown") for e in entries)

    lines = [
        "# Dataset Report -- canto-hk-speech-pipeline",
        f"Generated: {date.today()}",
        "",
        generated_note,
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total entries | {n:,} |",
        f"| Total hours | {total_hours:.1f}h |",
        f"| Unique speakers | {len(speakers)} |",
        f"| Unique sources | {len(sources)}: {', '.join(sorted(sources))} |",
        f"| Unique domains | {len(domains)}: {', '.join(sorted(domains))} |",
        f"| text_verified | {n_verified}/{n} ({100*n_verified/n:.1f}%) |",
        f"| 48 kHz | {n_48k}/{n} ({100*n_48k/n:.1f}%) |",
        f"| DNSMOS p50 | {percentile(dnsmos_vals, 50):.2f} |",
        f"| SNR p50 | {percentile(snr_vals, 50):.1f} dB |",
        f"| Duration p5/p50/p95 | {percentile(dur_vals,5):.1f}s / {percentile(dur_vals,50):.1f}s / {percentile(dur_vals,95):.1f}s |",
        f"| ASR agreement p50 | {percentile(agr_vals,50):.2f} |",
        f"| Jyutping valid rate | {jyutping_rate:.4f} ({jyutping_valid_tok:,}/{jyutping_total_tok:,} tokens) |",
        "",
        "## Acceptance Criteria",
        "",
        "(CLAUDE.md \"Acceptance Criteria\" table is the source of truth for these 12 rows;"
        " see this node's module docstring for how each is computed.)",
        "",
    ]

    criteria = [
        _check("Total clean hours", total_hours, ACCEPT["total_hours_min"]),
        _check("Unique speakers", len(speakers), ACCEPT["speakers_min"], fmt="{:d}"),
        _check("Sample rate 48000Hz pct", n_48k / n, ACCEPT["sample_rate_pct_min"], fmt="{:.3f}"),
        _check("text_verified pct", n_verified / n, ACCEPT["text_verified_pct_min"], fmt="{:.3f}"),
        _check("DNSMOS p50", percentile(dnsmos_vals, 50), ACCEPT["dnsmos_p50_min"]),
        _check("SNR p50 (dB)", percentile(snr_vals, 50), ACCEPT["snr_p50_min"]),
        _check("Jyutping valid rate", jyutping_rate, ACCEPT["jyutping_valid_rate_min"], fmt="{:.4f}"),
        _check("Distinct sources", len(sources), ACCEPT["sources_min"], fmt="{:d}"),
        _check("Distinct domains", len(domains), ACCEPT["domains_min"], fmt="{:d}"),
        _check("Duration in 3-20s range pct", n_in_range / n, ACCEPT["duration_in_range_pct_min"], fmt="{:.3f}"),
        _check("Windows paths in manifest", n_win, ACCEPT["windows_paths_max"], fmt="{:d}", good_if="<="),
    ]
    for c in criteria:
        lines.append(c["line"])

    lines.append(
        "- [GUARANTEED BY DESIGN -- not independently re-verified per-segment by this report] "
        "Single-speaker segments: VAD-cut segmentation always runs within a single "
        "diarization-detected speaker turn (CLAUDE.md hard constraint #5); there is no "
        "per-segment speaker-count column to check independently at report time."
    )

    all_pass = all(c["passed"] for c in criteria)
    lines += [
        "",
        f"**Overall (11 computed criteria; single-speaker is a design guarantee, see above): "
        f"{'READY FOR TRAINING' if all_pass else 'NOT YET READY'}**",
        "",
        "## Source Breakdown",
        "",
        "| Source | Segments | Hours |",
        "|--------|----------|-------|",
    ]
    for src in sorted(source_counts, key=source_counts.get, reverse=True):
        src_entries = [e for e in entries if e["source"] == src]
        src_hrs = sum(e["duration_sec"] for e in src_entries) / 3600
        lines.append(f"| {src} | {source_counts[src]:,} | {src_hrs:.1f}h |")

    lines += [
        "",
        "## Domain Breakdown",
        "",
        "| Domain | Segments | % |",
        "|--------|----------|---|",
    ]
    for dom in sorted(domain_counts, key=domain_counts.get, reverse=True):
        lines.append(f"| {dom} | {domain_counts[dom]:,} | {100*domain_counts[dom]/n:.1f}% |")

    lines += [
        "",
        "## Gender Distribution",
        "",
    ]
    for g, c in gender_counts.most_common():
        lines.append(f"- {g}: {c} ({100*c/n:.1f}%)")

    lines += [
        "",
        "---",
        "*Run `python -m pipeline.cli run report.build` to regenerate (add `--min-tier` to scope the check).*",
    ]

    return "\n".join(lines) + "\n", criteria


def run_report_build(*, min_tier: str | None = None) -> dict:
    """Synchronous, CLI-style entrypoint -- mirrors manifest.py's
    run_manifest_build() / label_store.py's run_label_store() shape (no async,
    no orchestrator conn= injection: this is a cheap, read-mostly, file-writing
    node, not a GPU batch node). Rebuilds the manifest-eligible pool fresh from
    the catalog on every call (via run_manifest_build), computes the 12
    CLAUDE.md acceptance criteria against it, and writes
    metadata/DATASET_REPORT.md.

    min_tier: optional passthrough to manifest.build's tier-cut convention
    (see TIER_PRECEDENCE / _tiers_at_or_above in pipeline/nodes/manifest.py) --
    e.g. min_tier='gold' scopes the report to the strictly human-verified
    subset only. None (default) = the full manifest-eligible pool, matching
    historical DATASET_REPORT.md scope.

    Returns a summary dict: path written, entry/hour/speaker counts, the full
    criteria list (each a dict with label/value/threshold/passed/line), and
    overall_pass (True iff all 11 computed criteria pass -- the single-speaker
    design guarantee is not a boolean and is not included in this rollup).
    """
    from pipeline.config import REPORT_PATH
    from pipeline.nodes.manifest import run_manifest_build

    t0 = time.time()
    manifest_result = run_manifest_build(min_tier=min_tier)
    entries = manifest_result["entries"]

    report_md, criteria = generate_report(entries, min_tier=min_tier)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_md, encoding="utf-8")

    overall_pass = all(c["passed"] for c in criteria) if criteria else False
    elapsed = time.time() - t0

    log.info(
        f"report.build: {manifest_result['count']} entries "
        f"({manifest_result['total_hours']}h, {manifest_result['n_speakers']} speakers) "
        f"-- {sum(1 for c in criteria if c['passed'])}/{len(criteria)} criteria passed "
        f"-- overall={'READY' if overall_pass else 'NOT READY'} "
        f"in {elapsed:.1f}s -> {REPORT_PATH}"
        + (f" [min_tier={min_tier}]" if min_tier is not None else "")
    )

    return {
        "path": str(REPORT_PATH),
        "count": manifest_result["count"],
        "total_hours": manifest_result["total_hours"],
        "n_speakers": manifest_result["n_speakers"],
        "tier_counts": manifest_result["tier_counts"],
        "criteria": criteria,
        "overall_pass": overall_pass,
        "min_tier": min_tier,
    }
