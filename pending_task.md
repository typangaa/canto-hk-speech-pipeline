# Pending Tasks

> **Maintenance rule**: this file must be updated whenever a task here is completed —
> move it to the "Done" section (with the completion date + commit/ref if applicable)
> instead of deleting it, and update `docs/PIPELINE_REVIEW_2026-07-11.md` §6 disposition
> table if the task closes one of that doc's numbered issues. Keep this file's Tier
> ordering current if priorities shift. See `CLAUDE.md` for the standing instruction to
> keep this file in sync.

Source: round-2 post-execution review of `docs/PIPELINE_REVIEW_2026-07-11.md` §6,
2026-07-11. Re-derive priorities from that doc if this file and it ever disagree.

---

## 🔴 Tier 1 — data-trust-critical, do first

### T1. Pilot QA batch review (Issue #15)
- **What**: 3 queued 300-segment pilot batches (auto_gold / silver / bronze) in
  `calibrate_review` — none reviewed yet.
- **How**: work through them via the live `pipe calibrate serve` browser UI.
- **Why first**: this is the only way to validate whether the 2026-07-11 tightened tier
  thresholds (0.95 / 0.85 / 0.70 + `canto_ft` confidence gate) actually hold up. Every
  downstream export and training decision rests on these thresholds.
- **Effort**: manual, ~900 segments, a few hours (can split across sessions).
- **Depends on**: nothing. `calibrate serve` is already running (PID 971786 at review time).
- **Owner**: human (not delegable to Claude).

---

## 🟠 Tier 2 — functional gaps, close before P6

### T5. Filter/tier re-evaluation mechanism (Issue #4)
- **What**: `filter.text`/`filter.decide`/`tier.assign` discovery is a bare
  row-existence anti-join — a later ASR model improving `asr_agreement.best_text`
  does NOT trigger re-evaluation of already-filtered/tiered segments. Two full backfills
  have papered over current data, but the gap is structural and will recur.
- **How**: add an `agreement_version` column (or track `asr_agreement`'s last-modified
  marker) on `filters_text`/`tiers`; change discovery to "no row **or** version stale".
- **Effort**: medium, a few hours (discovery SQL + migration + tests across 3 nodes).
- **Depends on**: nothing, but **must land before the next new ASR model** is added, or
  another manual backfill will be needed.

### T6. Re-export default manifest (Issue #2 + N3)
- **What**: `metadata/manifest.jsonl`/`train.jsonl`/`val.jsonl` stuck at 2026-07-09 —
  predates the whisper_v3 retirement backfill and the tier-tightening backfill.
- **How**: `pipe run manifest.export`.
- **Timing**: two `ingest.download` jobs (podcast + youtube) were running at review time.
  Wait for the full downstream chain to drain (see T7) before exporting, so this doesn't
  go stale again immediately.
- **Effort**: one command, but gated on T7.
- **Depends on**: T7. Trigger a fresh T4 report afterward.

### T7. Drain this ingest round's downstream backlog (N3)
- **What**: two `ingest.download` jobs (podcast + youtube, started 13:55 on 2026-07-11)
  are landing new raw files that need to flow through the full DAG before they're
  manifest-eligible.
- **How**: once ingest finishes, run in order (or via `run-many` where the node supports
  `conn=`): `ingest.commit` → `ingest.probe` → `lang_screen.auto` → `segment.diarize` →
  `segment.vad_cut` → `pregate.snr` → `asr.transcribe` (×3 backends) → `asr.agreement` →
  `filter.*` → `g2p` → `speaker.embed`/`speaker.cluster` → `tier.assign`. All idempotent.
- **Effort**: mostly GPU wall-clock, human-supervised; use `nohup` + `disown` for long jobs.
- **Depends on**: ingest completing; GPU availability.

---

## 🟡 Tier 3 — engineering cleanup, schedule into P6

### T8. Finish `conn=` injection on remaining 12/22 nodes (Issue #5)
- **What**: `run-many` concurrency needs `conn=` injection; only 10/22 nodes have it.
  Exact list in `docs/ORCHESTRATOR_PLAN.md`.
- **How**: mechanical per-node follow-up — add `conn=None` param + register in
  `RUN_MANY_ADAPTERS` + add a `tests/test_run_many.py` regression case.
- **Effort**: mechanical but broad, ~half a day; can be done in batches.

### T9. Speaker-centroid cosine pruning (§5 external best-practice gap)
- **What**: industry-standard practice (embedding + centroid cosine < 0.75 pruned) is
  missing — `speaker.cluster` has no per-segment purity check, which directly affects
  the "single-speaker segments 100%" acceptance criterion's real trustworthiness.
- **How**: add a step after `speaker.cluster` (new node or built into clustering) that
  computes each segment's cosine to its own speaker centroid; demote/exclude low-cosine
  segments. Calibrate the threshold against T1's QA ground truth.
- **Effort**: medium; **depends on T1** (need QA ground truth to set the threshold).

### T10. Content-hash linkage (§5 external best-practice gap)
- **What**: audio↔metadata linkage currently relies on absolute paths + `shard_index()`
  discipline; another drive migration would require a full P5-C-style rebalance again.
- **How**: add a content-hash column on `segments`; compute on new writes, backfill the
  rest slowly (io-bound, can run in background).
- **Effort**: small design, long backlog fill (618k+ files); low priority.

---

## 🟢 Tier 4 — optional / nice-to-have

### T11. Relocate dormant release data (Issue #16)
- Move `metadata/manifest_release.jsonl` (672MB) + `excluded_no_url.jsonl` into
  `metadata/release_dormant/`, alongside the 3 dormant scripts already there, for clarity.
  Zero risk difference — purely cosmetic. Two `mv` commands.

### T12. Automate log retention (Issue #11 residual)
- Phase B2 cleaned `metadata/logs/` (1.7GB → 17M) but there's no mechanism preventing
  regrowth. Add simple logrotate config or per-node startup truncation for oversized logs.
  Small effort.

### T13. A/B TTS-quality tier axis (`docs/LABEL_FRAMEWORK_SPEC.md` §10)
- Label store (lang/overlap/music/prosody) is complete, but the "pretrain vs. clean"
  A/B quality tier is not built. Part of the staged-training strategy for
  [[canto-tts-project]] — timing should be pulled by training needs, not pipeline hygiene.

---

## Suggested execution order

```
Do now (independent):      T1 (human QA) + T2/T3 (one commit) + T11 (opportunistic)
Once ingest drains:        T7 (downstream chain) → T6 (re-export)
Parallel track:            T4 (report.build) — pair with T6 for a fresh report
Before next new ASR model: T5 (re-eval mechanism)
Schedule into P6 proper:   T8, T9 (depends on T1), T10, T12
Pulled by training needs:  T13
```

---

## Done

### T2. Re-baseline `catalog verify` row_count checks (Issue N1) — done 2026-07-11
`pipeline/catalog/verify.py`'s `check_row_counts()` now does floor-only (or floor+ceiling
for the three tables that are 1:1-with-segments: `asr_agreement`/`filters`/`tiers`) checks
against live-queried 2026-07-11 baselines, replacing the old exact-match `EXPECTED` dict —
same pattern already established in `tests/test_catalog.py`'s `*_monotonic_growth` tests.
Verified: `pipe catalog verify` now shows 17/17 PASS (was 10/17 FAIL). See
`docs/PIPELINE_REVIEW_2026-07-11.md` §6 Issue N1 disposition.

### T3. Fix flaky snapshot test (Issue N2) — done 2026-07-11
`tests/test_catalog.py::test_manifest_build_matches_expected_corpus_totals` converted
from hardcoded `==` to tolerant floor/ceiling/tolerance-window assertions (count/n_speakers:
floor + generous ceiling; gold: floor only, expected to trend up; auto_gold/silver/bronze:
±1,000-row tolerance window, since they deplete as rows promote to gold). Landed in the
same batch as T2. Verified: full suite green (304 passed), including this test, while a
live `calibrate serve` review session was actively drifting `gold` in the background.

### T4. Port `report.build` node (Issue #3) — done 2026-07-11
New `pipeline/nodes/report.py`: `run_report_build(*, min_tier=None)` reuses
`manifest.py`'s `run_manifest_build()` to read the manifest-eligible pool LIVE from the
catalog on every call (never a stale file), computes all 12 CLAUDE.md Acceptance Criteria
(fixing a legacy bug where the old script silently never checked 2 of its 11 declared
thresholds), and writes `metadata/DATASET_REPORT.md`. `text_verified` and single-speaker
are reported honestly (not faked to pass) — see the node's module docstring. Registered as
`pipe run report.build` (`--min-tier` scoping, matching `manifest.build`/`manifest.export`'s
convention). `scripts/10_report.py` retired via `git rm` per its own documented condition —
`scripts/` is now empty. Live run against the real catalog: 458,844 entries / 1018.9h /
8,817 speakers, 10/11 criteria PASS (only `text_verified` fails, correctly — see T1).
CLAUDE.md/README.md updated to match. See `docs/PIPELINE_REVIEW_2026-07-11.md` §6 Issue #3
disposition.
