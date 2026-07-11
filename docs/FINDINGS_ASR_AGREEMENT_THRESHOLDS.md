# Findings: ASR Cross-Model Agreement vs. Corpus Hours (2026-07-10)

## Why this exists

`pipeline/nodes/calibrate.py`'s module docstring recorded (2026-07-10, before this analysis):
> "Checked 2026-07-10: auto-promoting via a high cross-model-agreement threshold instead of
> human review is NOT viable -- only 26 filter-passing segments clear agreement >= 0.95
> corpus-wide (median agreement is 0.71)."

That check used the **4-way** agreement (`canto_ft` + `whisper_v3` + `qwen3_asr` + `sense_voice`).
Later the same day, the owner flagged `whisper_v3` (`Systran/faster-whisper-large-v3+zh`) as
"highly inaccurate" from direct observation and asked to remove it from the pipeline. This
document re-runs the same measurement **excluding `whisper_v3`** and shows the "not viable"
conclusion was an artifact of including a disproportionately-disagreeing model, not a property
of the corpus itself. This is the evidence trail behind the `auto_gold` tier decision — see
`DECISIONS.md` 2026-07-10 entry and `pipeline/nodes/tier.py`.

A separate, independent data-hygiene issue was found and fixed alongside this: `canto_ft`'s
`asr_results.model` string is an absolute path built from `REPO_ROOT` at run time
(`pipeline/nodes/asr.py`'s `_LOCAL_CANTO`), and the repo has moved directories twice historically.
~5.5% of segments (33,921 / 618,695) therefore carry a duplicate `canto_ft` row under a stale
path string. Uncorrected, this double-counts `canto_ft`'s opinion in the agreement average for
those segments. The numbers below dedupe to the current live path.

## Method

- **Sample-based check** (20,000 segments, `filters.pass = TRUE`, all 4 models present):
  fast, used to explore threshold shape.
- **Full-corpus check** (all 484,832 filter-passing segments, 1,068.4h total): used for the
  final hour numbers below — every segment in the current filter-passing pool, not a sample.
- Agreement = `char_agreement()`'s existing definition (average pairwise
  `difflib.SequenceMatcher(None, a, b).ratio()` across all model-text combinations) — unchanged
  logic, just computed over a 3-model set instead of 4, with `canto_ft` deduped to its current
  path.
- "3-way + canto_ft conf>0.8" = the actual proposed `auto_gold` gate: 3-way agreement AND
  `canto_ft`'s own (real, logprob-derived) confidence > 0.8.

## Result 1 — % of segments cleared, by threshold (20,000-segment sample)

| Agreement threshold | 4-way (incl. whisper_v3) | 3-way (excl. whisper_v3) | 3-way + canto_ft conf>0.8 |
|---|---|---|---|
| ≥0.65 | 94.6% | 97.2% | 92.2% |
| ≥0.70 | 89.6% | 94.6% | 90.3% |
| ≥0.75 | 79.4% | 89.6% | 86.2% |
| ≥0.80 | 62.2% | 79.3% | 77.0% |
| ≥0.85 | 39.1% | 63.5% | 62.2% |
| ≥0.90 | 15.7% | 41.1% | 40.6% |
| ≥0.95 | 2.2% | 14.3% | 14.2% |

## Result 2 — corpus hours cleared, by threshold (full 484,832-segment / 1,068.4h pool)

| Agreement threshold | 4-way hrs | 3-way hrs | 3-way + canto_ft conf>0.8 hrs |
|---|---|---|---|
| ≥0.60 | 1,041.1h (97.5%) | 1,055.0h (98.7%) | 1,014.8h (95.0%) |
| ≥0.65 (current silver bar) | 1,014.4h (94.9%) | 1,042.6h (97.6%) | 1,005.4h (94.1%) |
| ≥0.70 | 961.1h (90.0%) | 1,018.5h (95.3%) | 986.5h (92.3%) |
| ≥0.75 | 849.7h (79.5%) | 965.6h (90.4%) | 940.6h (88.0%) |
| ≥0.80 | 666.4h (62.4%) | 862.4h (80.7%) | 846.0h (79.2%) |
| ≥0.85 | 411.6h (38.5%) | 688.1h (64.4%) | 679.6h (63.6%) |
| ≥0.90 (`auto_gold` bar) | 150.6h (14.1%) | 448.8h (42.0%) | 446.2h (41.8%) |
| ≥0.95 | 15.7h (1.5%) | 148.3h (13.9%) | 148.0h (13.9%) |

Average segment duration ≈ 7.9s. Existing true human-verified `gold` = 43 segments (< 1h,
negligible against these totals).

## Interpretation

- The "only 26 segments clear 0.95" finding was real but **whisper_v3-specific**: it single-handedly
  drags 4-way agreement down across the whole corpus. 3-way ≥0.95 clears 148h — over 5,600×
  more coverage than the 4-way check implied.
- `canto_ft confidence > 0.8` barely changes the picture at high agreement (≥0.85: -1.3pp;
  ≥0.90: -0.2pp) — segments that already agree strongly across 3 independent models tend to
  also have a confident `canto_ft` transcript. It matters more at the low end (≥0.65: -3.5pp;
  ≥0.70: -3.0pp), where it's doing real filtering work.
- The gap between ≥0.70 and ≥0.80 (3-way) is 15.3 percentage points / ~156h — this is the
  "medium agreement, not yet trustworthy alone" band most of `filter.pass`'s current silver
  tier actually lives in.

## Recommendation — thresholds by target dataset size

| Target | Suggested `--min-agreement` | Pool at that cut | Suggested QA sample rate |
|---|---|---|---|
| 100h (cleanest / pilot) | ≥0.95 | 148h (~67,400 segments) | ~5-8% (~3,400-5,400 segments) |
| 500h (main training set) | ≥0.85 | 688.1h (~313,700 segments) | ~2-3% (~6,300-9,400 segments) |
| 1000h (full pretrain pool) | ≥0.65 (today's silver bar, unchanged) | 1,042.6h (~475,000 segments) | ~0.5-1% (~2,400-4,750 segments) |

Rationale: at 1000h scale, the target is essentially "the whole current filter-passing pool" —
QA there is about monitoring the overall error rate, not gatekeeping individual segments. At
100h scale, the absolute segment count is small enough that a stricter bar *and* a higher QA
percentage are both affordable, and the corpus doesn't need everything the low bar would allow
in, so there's no reason to accept lower-confidence data.

See `pipeline/nodes/manifest.py`'s `--min-agreement` export flag and
`pipeline/nodes/calibrate.py`'s `min_agreement`-scoped sampling for the implementation.
