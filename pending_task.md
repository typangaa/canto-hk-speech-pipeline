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
- **What**: **corrected 2026-07-17** — actual queue is **10 sample batches, 2,400 segments
  total** in `calibration_review` (the "3 queued 300-segment pilot batches (~900)" figure
  below was stale/undercounted; batches accumulated across several sessions via repeated
  `calibrate.sample` calls). **58 already reviewed** (all `verified`, all in one batch,
  flipped to `gold`) — **2,342 still pending**. Tier split of the pending pool: auto_gold
  ~1,030, silver ~382, bronze ~917, excluded ~13 (drifted post-queue, e.g. a since-rejected
  id). Run `pipe calibrate progress` anytime for a live breakdown (see tool below).
- **Owner decision 2026-07-17**: pure-Cantonese segments are assumed already-adequate
  quality — **review priority is code-switch segments** (`filters.english_ratio > 0`), not
  a flat pass over the whole queue. Of the 2,342 pending, 787 are already code-switch by
  incidental distribution (not deliberately oversampled); T18's `--code-switch only` sample
  flag (built 2026-07-15, never yet used to queue a batch) is available to queue a
  dedicated, oversampled code-switch batch on top of that if/when wanted — see T18 in Done
  for `recommended_sample_n(..., code_switch=True)`'s suggested sizes (auto_gold 1,250,
  silver 10,366, bronze capped at 100%/50,524 — those are full-population figures, a
  smaller pilot slice is the practical near-term ask, not a full pass).
- **Code-switch pilot batch queued 2026-07-17**: owner chose a small dedicated pilot
  (150/tier, not T18's full oversampled recommendation) — `pipe run calibrate.sample --tier
  {auto_gold,silver,bronze} --code-switch only --n 150` ×3, run_ids
  `calibrate_sample_888979e8434c` (auto_gold), `calibrate_sample_c215b57afdff` (silver),
  `calibrate_sample_aea932ef4a8d` (bronze). Queue is now **2,850 total / 2,792 pending**,
  code-switch pending up to 1,237 (from 787).
- **New tool (2026-07-17)**: `pipe calibrate progress` — read-only CLI command
  (`pipeline/nodes/calibrate.py::progress_report()` /  `run_calibrate_progress()`, CLI
  wiring in `pipeline/cli.py::cmd_calibrate_progress`) that breaks the whole review queue
  down by tier x code-switch status x decision, so the backlog can be checked anytime
  without hand-writing a query. 3 new tests in `tests/test_calibrate_node.py` (57/57
  passing). Built specifically so the owner can track progress against the code-switch
  focus above without me touching the review process itself (owner explicitly kept T1
  review itself human-only, not delegable).
- **How**: work through them via the live `pipe calibrate serve` browser UI.
- **Why first**: this is the only way to validate whether the 2026-07-11 tightened tier
  thresholds (0.95 / 0.85 / 0.70 + `canto_ft` confidence gate) actually hold up. Every
  downstream export and training decision rests on these thresholds.
- **Effort**: manual, owner-paced — **owner decision 2026-07-17: no fixed completion
  target, stop whenever** (superseded the earlier "~900 segments, a few hours" estimate,
  which was based on the stale count anyway).
- **Depends on**: nothing. `calibrate serve` is already running (PID 971786 at review time).
- **Owner**: human (not delegable to Claude).
- **Update 2026-07-13**: `calibrate serve` no longer blocks while a long batch node (T15's
  `asr.transcribe`) holds the DuckDB writer lock — it falls back to a JSON snapshot for reads
  and buffers every decision to a local JSONL, replayed via `pipe calibrate flush-pending` once
  the writer frees up (see DECISIONS.md 2026-07-13). To review DURING a long batch run: first
  run `pipe calibrate export-snapshot` while the catalog is free (or use whatever snapshot is
  already on disk), then `pipe calibrate serve` works throughout the run; run
  `pipe calibrate flush-pending` afterward to land the decisions.

---

## 🟠 Tier 2 — functional gaps, close before P6

(none currently — T5 done 2026-07-17, see Done section)

---

## 🟡 Tier 3 — engineering cleanup, schedule into P6

### T9. Speaker-centroid cosine pruning (§5 external best-practice gap)
- **What**: industry-standard practice (embedding + centroid cosine < 0.75 pruned) is
  missing — `speaker.cluster` has no per-segment purity check, which directly affects
  the "single-speaker segments 100%" acceptance criterion's real trustworthiness.
- **How**: add a step after `speaker.cluster` (new node or built into clustering) that
  computes each segment's cosine to its own speaker centroid; demote/exclude low-cosine
  segments. Calibrate the threshold against T1's QA ground truth.
- **Effort**: medium; **depends on T1** (need QA ground truth to set the threshold).
- **2026-07-16 requirements discussion**: this actually requires a NEW QA signal, not
  just T1's existing text-verification ground truth — `pipe calibrate serve` only lets a
  human judge text correctness today, with no way to flag "this segment isn't actually
  single-speaker" or "this segment is single-speaker but got clustered under the wrong
  speaker_id" (two DIFFERENT failure modes: the first violates CLAUDE.md Hard Constraint
  #5 and should always exclude; the second is a harmless metadata mislabel, audio is
  fine). Advised the owner: keep Hard Constraint #5 as-is (it's foundational to what a
  TTS training clip means, not worth relaxing), but the UI needs a dedicated
  speaker-purity label — proposed either one non-exclusionary "Speaker issue" button, one
  auto-excluding button (conflates the two failure modes), or two separate buttons (most
  correct, more UI surface). **Owner wants to investigate further before deciding** —
  parked, no UI change made this session. Revisit once the owner has a direction.
- **Owner decision 2026-07-17**: two separate buttons (the "most correct" option above).
  **UI built same session** — even though T9 proper (the actual cosine-pruning threshold
  calibration) stays parked until T1 produces enough QA ground truth, the two buttons were
  added now so the T1 review sessions already underway (owner is actively reviewing the
  2,850-segment queue) start collecting this ground truth immediately instead of needing a
  second review pass later:
  - `pipeline/nodes/calibrate.py`: `NOT_SINGLE_SPEAKER_FLAG_REASON = "not_single_speaker"`
    (submitted as `decision='rejected'` — a real audio defect, Hard Constraint #5
    violation, excludes from the manifest, same mechanism as the existing Mandarin button)
    and `WRONG_SPEAKER_ID_FLAG_REASON = "wrong_speaker_id"` (submitted as
    `decision='flagged'` — harmless metadata mislabel, audio is fine, does NOT exclude,
    does NOT touch `tiers`/`asr_agreement`). No `record_decision()` code changes needed —
    both ride its existing generic `'rejected'`/`'flagged'` handling.
  - `pipeline/tools/calibrate_server.py`: two new buttons ("Multi-speaker (N)" /
    "Wrong speaker ID (W)") + matching keyboard shortcuts, styled like the existing
    Mandarin button. `summary_stats()`'s `top_flag_reasons` leaderboard picks up both
    automatically (already generic over any `flag_reason` string).
  - 2 new tests in `tests/test_calibrate_node.py` (61/61 passing) verifying the exclude vs.
    non-exclude behavior for each button; live-smoke-tested that `_build_app()` still
    constructs and both button ids/flag reason strings are present in the served page.
  - **Still not done**: the actual cosine-pruning threshold/node itself (T9 proper) —
    stays parked on T1 ground truth as before. This was purely the data-collection
    prerequisite.

### T10. Content-hash linkage (§5 external best-practice gap)
- **Owner decision 2026-07-16**: defer until canto-tts training actually starts consuming
  the corpus (not just "training is coming up" — the trigger is the training run itself).
- **What**: audio↔metadata linkage currently relies on absolute paths + `shard_index()`
  discipline; another drive migration would require a full P5-C-style rebalance again.
- **How**: add a content-hash column on `segments`; compute on new writes, backfill the
  rest slowly (io-bound, can run in background).
- **Effort**: small design, long backlog fill (618k+ files); low priority.

(T15 — see Done section: fully drained, 2026-07-17)

### T14. Full CPU+GPU utilization during chained node runs (found 2026-07-12)
- **Owner decision 2026-07-16**: wants levers (3)+(4) done — approved.
- **Progress 2026-07-17 — levers (3) and (4) both built, (3) live-validated, (4)
  unit-tested but not yet live-validated against a real GPU stage:**
  1. **Re-test of the original stall case (owner's recommended first step) —
     CONFIRMED FIXED.** `pipe run-many ingest.probe -- speaker.cluster` (real
     backlog: 4,086-row `ingest.probe` backlog from the same-day download round +
     a full 3-source `speaker.cluster` recompute, 1,241,586 segments → 14,330
     speakers) completed in **212s total, zero stalling** — both nodes' log lines
     interleaved throughout, confirming genuine concurrency, not serialization.
     This is the same class of pairing that stalled 30+ minutes in T15 points 3-5
     before the 2026-07-16 `upsert_rows()` bulk-write fix. Lever (3)'s premise is
     now viable.
  2. **Lever (3) built**: `pipeline/tools/chain_runner.py` (`pipe chain run`) —
     a committed replacement for the ad-hoc `run_t7_chain.sh`-style scripts that
     were hand-written and thrown away each time. Codifies the full
     ingest→quality_tier DAG as 12 ordered rounds; two rounds pair genuinely
     independent nodes via `run-many` instead of a strict waterfall: round 2
     (`ingest.probe` + `lang_screen.auto`, both read `raw_files` only, write
     disjoint tables) and round 11 (`g2p` + `tier.assign` + `speaker.cluster`,
     the direct stand-in for the historically-stalled pairing — all three read
     already-landed tables and write disjoint ones). `--only`/`--skip` (comma
     round numbers), `--devices` (threaded to the 3 GPU rounds only), `--dry-run`.
     Every round's discovery is an idempotent anti-join, so re-running the full
     chain when a round has nothing to do just no-ops fast — safe default.
     18 new tests in `tests/test_chain_runner.py` (command construction, ordering,
     only/skip filtering, failure short-circuit, log file). Live-validated: full
     `--dry-run` round plan correct, `--only 1` (`ingest.commit`) ran for real.
  3. **Lever (4) built, NOT yet live-validated against a real GPU stage**:
     `pipeline/tools/stream_drain.py` (`pipe chain stream`) — backgrounds the
     upstream GPU node (e.g. `asr.transcribe`) via `Popen`, then re-invokes the
     downstream node(s) (solo `pipe run` or `run-many` if more than one) on a
     poll interval (default 300s) while the upstream process is still alive,
     plus one final drain pass after it exits. Relies on the same idempotent-
     anti-join property lever (3) does — no coordination between the two
     processes beyond the catalog itself. 9 new tests in `tests/test_stream_drain.py`
     (`FakePopen`-based, injectable `sleep_fn`, no real wall-clock sleeps) —
     command construction and poll/drain sequencing verified, but this has
     **not been run against a real long-running `asr.transcribe` pass yet** —
     no large ASR backlog was queued this session (the new download round
     hadn't reached segmentation/ASR by session end). **Live-validate the next
     time a real multi-hour `asr.transcribe` pass is queued** — pair with
     `asr.agreement` (or `asr.agreement g2p` once g2p has backlog again) and
     confirm the poll loop actually drains mid-run, not just at the end.
  - Historical context retained below (original problem statement + lever menu
    + the lever (2) `filter.acoustic` tuning result, already concluded — leave
    as-is, don't change).
- **What**: measured during the T7 chain resume run — `asr.agreement` (CPU-only stage)
  used only 471% CPU (4.7 of 48 cores, `load average` 7.06/48); during the preceding
  `asr.transcribe` stage each GPU sat at pool `target=1` using only ~2.7GB/24GB VRAM per
  model. The `run_t7_chain.sh`-style orchestration also runs every stage as a strict
  waterfall (`ingest.probe` → ... → `tier.assign`, each fully blocking the next), so GPUs
  are 100% idle during every CPU-only stage and CPU cores are ~90% idle during every
  GPU-only stage — despite `pipe run-many` (all 23/23 node call sites have `conn=`
  injection, confirmed done 2026-07-07 per `docs/ORCHESTRATOR_PLAN.md` line 3 — **T8 above
  was stale, already fully done, moved to Done section below**) already providing the
  mechanism to run independent nodes concurrently under one shared DuckDB connection.
- **How** (four independent levers, can be landed separately):
  1. **GPU pool `target` bump**: `asr.transcribe`/`segment.diarize`/`speaker.embed` VRAM
     headroom (~11% used per model on a 24GB card) suggests 2+ concurrent workers per GPU
     is safe — try e.g. `--devices cuda:0,cuda:0,cuda:1,cuda:1` (same device listed twice)
     and measure real throughput gain vs. VRAM/compute contention.
  2. **CPU-stage internal parallelism**: `filter.acoustic` already has `--workers` (mind
     the DNSMOS onnxruntime thread-oversubscription trap from earlier sessions — set
     `OMP_NUM_THREADS=1` per worker); `asr.agreement`, `g2p`, `filter.text`,
     `speaker.cluster`, `tier.assign` appear to run as a single process with no
     `--workers`/multiprocessing option — audit each and add worker-pool parallelism
     where the per-item work is independent (most of these are pure-Python/DuckDB CPU
     loops, good multiprocessing candidates).
  3. **Overlap independent stages via `run-many` in the chain script**: rewrite
     `run_t7_chain.sh`'s strict waterfall to pair genuinely independent nodes in the same
     `run-many` invocation when their discovery sets don't conflict — e.g. next round's
     `ingest.probe`/`lang_screen.auto` can run alongside this round's `asr.transcribe`
     (disjoint catalog tables, no lock contention).
  4. **(Bigger lift) Streaming pipeline via poll loop**: because every node's discovery
     is an idempotent anti-join, downstream CPU stages (`asr.agreement` → `filter.*` →
     `g2p`) could in principle drain newly-landed rows continuously while an upstream GPU
     stage (`asr.transcribe`) is still running, instead of waiting for the entire GPU
     stage to finish first. Needs a wrapper that re-invokes `run-many` for the downstream
     nodes on an interval until the upstream stage's process exits, then a final drain
     pass. Biggest potential wall-clock win (GPU+CPU already-idle window observed was a
     large fraction of total chain time), but needs script engineering + testing to avoid
     redundant no-op invocations spamming the log.
- **Effort**: (1)+(2) are quick, low-risk, testable independently against the real
  catalog with `--limit`; (3) is a script-only change; (4) is a half-day-plus design+build.
- **Depends on**: nothing blocking; safe to pick off individually. Do (1)/(2) first since
  they're cheapest to validate.
- **Lever (2) tested live 2026-07-12 — CLI default is already the best config found,
  do not change it.** Measured 3 configs against the real T7 `filter.acoustic` backlog,
  each read via a proper steady-state sample (several deltas 40s+ after worker startup,
  not the first minute — an early reading looked like +25% and was actually just
  worker-startup transient, corrected after longer sampling):
  - `--workers 4 --threads 4` (CLI default): steady-state **~38-43/s**, ~1993% CPU
    (~20 cores), load average ~7-23 depending on what else is running. **Best of the 3.**
  - `--workers 8 --threads 3`: steady-state **~17-19/s** (worse) — fewer threads/worker
    slows each DNSMOS inference enough that 8 workers don't compensate; total CPU stayed
    flat at ~19-20 cores (same budget, split into smaller/less-efficient units).
  - `--workers 8 --threads 4`: steady-state **~11-12/s** (worst) — CPU shot up to
    ~3459% (~35 cores), `load average` hit 42/48 — classic oversubscription/thrashing,
    8 concurrent ONNX sessions contending for L3 cache/memory bandwidth. More cores used,
    less work done.
  - **Conclusion**: `filter.acoustic`'s bottleneck is not raw core count — it's
    per-worker ONNX inference efficiency, which degrades if threads-per-worker drops
    below the default, and CPU/cache contention, which appears fast if worker count
    rises without a matching drop in total nominal threads. Leave `filter.acoustic` at
    its CLI default; do not spend more time tuning this specific stage's own
    workers/threads knobs. The bigger win is still lever (3)/(4) (stage-waterfall idle
    time) — worth remembering as a general lesson before tuning any other CPU-pool node
    in this codebase (`asr.agreement`, `g2p`, etc. if multiprocessing is added there
    later): always verify with a steady-state sample, not the first 1-2 minutes.

---

## 🟢 Tier 4 — optional / nice-to-have

(none currently — T9/T10/T14 remaining levers are Tier 3, see above; T13 moved to Done)

---

## Suggested execution order

```
Do now (independent):      T1 (human QA, owner-paced, code-switch focus — 2026-07-17)
Schedule into P6 proper:   T9 (depends on T1), T10, T14 levers (3)+(4) live-validation
Pulled by training needs:  T13
```
(updated 2026-07-17 — T5/T6/T16/T15-remaining-chain all moved to Done since this list was
last written; see Done section for what actually landed and when.)

---

## Done

### T5. Filter/tier re-evaluation mechanism (Issue #4) — done 2026-07-17
- **What**: `filter.text`/`filter.decide`/`tier.assign` discovery was a bare row-existence
  anti-join — a later ASR model improving `asr_agreement.best_text` did NOT trigger
  re-evaluation of already-filtered/tiered segments. Owner chose to do this now (full
  3-node scope) rather than defer to the next new ASR model.
- **How (built)**: rather than a separate version/timestamp column, reused
  `asr_agreement.model_count` (already increments whenever a new active ASR model lands
  for an id — see `asr.agreement`) as the version signal, snapshotted at evaluation time:
  - `filters_text.asr_model_count` (schema.sql), compared against `asr_agreement.model_count`
    in `TEXT_DISCOVER_SQL` — re-evaluates on "no row OR stale count".
  - `filters.text_model_count`, compared against `filters_text.asr_model_count` in
    `DECIDE_DISCOVER_SQL` — catches filter.decide going stale when filter.text re-evaluates
    underneath it, even if filter.decide itself was never touched by a new ASR model directly.
  - `tiers.asr_model_count`, compared against `asr_agreement.model_count` in
    `TIER_DISCOVER_SQL`.
  - `filter.acoustic` deliberately untouched — it only reads `segments.audio_path` (never
    `asr_agreement`), so its output can never go stale from an ASR model change; its existing
    `fa.id IS NULL` discovery was already correct.
- **Bug found while implementing (data-trust-relevant)**: `tier.assign`'s OLD discovery
  anti-joined via `LEFT JOIN tiers t ON a.id = t.id AND t.provenance = 'tier_assign'` — a
  human review decision written by `calibrate.py`'s `record_decision()` carries a DIFFERENT
  provenance (`'calibrate_verify'`/`'calibrate_reject'`), so it always looked like "not yet
  tiered by this node" and got silently re-processed by the very next `tier.assign` run.
  Confirmed live against the real catalog: all 58 currently-`verified` rows had already had
  their provenance silently overwritten `calibrate_verify` → `tier_assign` by a prior
  full-backlog run (harmless here since `text_verified=True` deterministically re-computes
  `'gold'` either way) — but a `'rejected'` row would NOT have been harmless: `assign_tier
  (False, ...)` recomputes from agreement/dnsmos alone and could silently promote a
  human-rejected segment straight back into the manifest-eligible pool, undoing T19's
  'rejected' propagation fix. No `'rejected'` decisions exist in the catalog yet (0 recorded),
  so this hadn't visibly corrupted data, but was a live landmine for the T1 code-switch pilot
  batch queued this same session. Fixed as part of the same discovery-SQL rewrite —
  `TIER_DISCOVER_SQL` now unconditionally excludes `provenance IN ('calibrate_verify',
  'calibrate_reject')` from re-discovery, regardless of `model_count`, making a human decision
  permanently terminal (matches `assign_tier()`'s own documented invariant).
- **Migration + backfill**: `pipeline/catalog/schema.sql` gained 3 `ALTER TABLE ... ADD
  COLUMN IF NOT EXISTS` (idempotent, applied automatically via `init_schema()` on next
  `connect()`). Ran a one-time backfill against the real catalog immediately after migrating
  (all 1,241,610 rows in `filters_text`/`filters`/`tiers` would otherwise have `NULL` version
  columns and look simultaneously "stale" — verified this WOULD have triggered an
  unintended full-corpus reprocess before backfilling, confirmed 0-row discovery backlogs
  after). `catalog verify` re-run clean, 17/17 PASS.
- **Tests**: 14 new (`tests/test_filter_node.py` — discover_text/discover_decide
  re-evaluation + legacy-NULL cases; `tests/test_tier_node.py` — discover() re-evaluation +
  both human-decision-protection regression tests). 437/437 total passing.
- **Not done / left as-is**: no attempt to backfill or re-run the actual filter/tier logic
  for any segment (nothing changed operationally this session — no new ASR model landed —
  this was purely the mechanism + a preventative correctness fix + the version-column
  backfill needed to ship it safely).

### T13. A/B TTS-quality tier axis (`docs/LABEL_FRAMEWORK_SPEC.md` §10) — done 2026-07-16
- Pulled forward from Tier 4 by owner decision — canto-tts training is about to start,
  scope narrowed to gold+auto_gold only (not the full manifest-eligible pool).
- **Done**: new node `pipeline/nodes/quality_tier.py` (`quality_tier.assign`) + table
  `quality_tiers (id, quality_tier, provenance)`. Tier A = full gold+auto_gold scope
  (223,605 segs); Tier B (clean, strict bundle owner-picked after a 3-way loose/medium/
  strict comparison against the real distribution) = `dnsmos>=3.7 AND music_prob<0.10
  AND overlap_ratio<0.05` (55,580 segs / 152.1h). Explicitly a SEPARATE axis from
  `tiers`/`tier.assign` — documented in both nodes' docstrings + CLAUDE.md's "Tier is
  overloaded" section to prevent conflation.
- `manifest.build`/`manifest.export` gained `--min-quality-tier {A,B}` (LEFT JOIN, so
  silver/bronze/unscored rows stay included when unused). Exported
  `metadata/manifest_tier_auto_gold_qualityB.jsonl` (55,594 entries/152.1h/1,860 speakers)
  for the clean fine-tune stage; Tier A already covered by the existing
  `manifest_tier_auto_gold.jsonl`.
- Full backfill: 279,185 segments in 4s (validates the same-day upsert_rows() fix again).
- **Tests**: 19 new in `tests/test_quality_tier_node.py`, 11 new in `tests/test_manifest_node.py`.
  Full writeup: DECISIONS.md 2026-07-16.
- **Not done**: no Tier A-only export file written (redundant with the existing
  `manifest_tier_auto_gold.jsonl`); label coverage gap (~3-5% of the gold+auto_gold scope
  has no `labels_music`/`labels_overlap` row yet) means a handful of segments fail closed
  to Tier A that a full label.suite backfill might upgrade to Tier B later — not
  re-triggered automatically (same structural gap as T5).

### T12. Automate log retention (Issue #11 residual) — done 2026-07-16
- Phase B2 cleaned `metadata/logs/` (1.7GB → 17M) but there's no mechanism preventing
  regrowth. Most growth is ad-hoc shell-redirected batch logs (`t15_*.log` etc.), not
  just the handful of nodes using `logging.FileHandler` directly — a Python-side
  truncation hook wouldn't catch those, so went with a standalone prune script instead.
- **Done**: `pipeline/tools/prune_logs.py` (`prune_logs()`) — gzips `*.log` older than
  `--gzip-after-days` (default 7), deletes `*.log.gz` archives older than
  `--delete-after-days` (default 60). Idempotent (already-gzipped files skipped,
  operates on mtime). Wired as `pipe logs prune` (`--dry-run` supported) in
  `pipeline/cli.py`. 7 new tests in `tests/test_prune_logs.py`.
- **Automated**: added a real (not Claude-session-scoped) weekly crontab entry —
  `0 3 * * 0` (Sun 3am) — running `pipe logs prune >> metadata/logs/prune_cron.log`.
  Its own log is subject to the same pruning, self-limiting.
- **Result**: first real run (not dry-run) gzipped 46 files, reclaimed 14.8MB
  (`metadata/logs/` 70M → 56M — most of the remaining size is a handful of files
  younger than the 7-day gzip threshold, e.g. the T15 batch64 run log, which will
  compress on their next scheduled pass).

### T11. Relocate dormant release data (Issue #16) — done 2026-07-16
Moved `metadata/manifest_release.jsonl` (672MB) + `excluded_no_url.jsonl` (8.4MB) into
`metadata/release_dormant/`, alongside the 3 dormant scripts already there. Zero risk
difference (both `mv`d, not copied — no duplicate left behind); grepped first to confirm
no code references the old root-level path, only prose in CLAUDE.md/DECISIONS.md/this
file/the review doc — none of which are path-sensitive.

### upsert_rows() performance fix — done 2026-07-16
Not a numbered T-task (tracked only in `docs/UPSERT_PERFORMANCE_FIX_PLAN.md`, found
mid-sweep while auditing the uncommitted backlog before commit) — closing the loop here
so it doesn't fall through the cracks again. `upsert_rows()` (`pipeline/catalog/catalog.py`)
switched from per-row `conn.executemany()` to a vectorised `pd.DataFrame` +
`INSERT ... SELECT` bulk path above `UPSERT_BULK_THRESHOLD = 2_000` rows. Real-world
validation: full 3-source `speaker.cluster` rerun (1,241,586 segments) — **104s total**
vs. the historical ~78min for the podcast source's write alone (45×+ speedup), zero
data drift (identical row/speaker counts). 357/357 tests passing (14 new in
`tests/test_upsert_rows.py`). Full writeup: DECISIONS.md 2026-07-16. Side effect worth
tracking under T14: removes the root cause of the `run-many`
`asr.transcribe`+`speaker.cluster` pairing stall (T15 points 3-5) — worth a retry next
time both have real backlogs queued.

### T6. Re-export default manifest (Issue #2 + N3) — done 2026-07-15
`pipe run manifest.export` re-run after T15's full chain landed (see T15 addendum below)
and T16's auto_gold gate rebuild. `metadata/manifest.jsonl`/`train.jsonl`/`val.jsonl`
regenerated 2026-07-15 21:35 (606,775 entries). `report.build` re-run 2026-07-16 00:03:
1349.3h / 9,023 speakers / 3 sources / 6 domains, 11/12 acceptance criteria PASS (only
`text_verified` fails, expected — see T1).

### T15. Drain the reingest.pending backlog through the full DAG (found 2026-07-12) — done 2026-07-17
- **What**: `docs/IO_OPTIMIZATION_PLAN.md` diagnosed Drive4's file-count skew (2.55M
  files, dentry-cache thrashing was the real `speaker.cluster` I/O bottleneck, not
  clustering compute or GPU availability — see that doc §1-2). Phase 0
  (`vm.vfs_cache_pressure=50`) and Phase 1 (archive-then-delete 1,310,284 dead legacy
  `.transcript.json`/`.pregate.json` sidecars, verified 100% superseded by
  `asr_results`/`orphan_segments` via a 5,500-file sample, tar'd to
  `/mnt/Drive3/canto/archive/` first) are both done — Drive4 file count
  2,547,466 → 1,237,182. Phase 2 was **re-scoped by owner decision (2026-07-12)**: rather
  than delete the 578,889-row `orphan_segments.status='pending_delete'` queue (≈411GB,
  classified by the one-time `recover.orphans` node using LEGACY ASR/pregate evidence),
  re-admit it into `segments` for a fresh pass through the CURRENT 3-model ASR ensemble
  (`canto_ft`/`qwen3_asr`/`sense_voice` — qwen3_asr measured ~0.4% CER vs 17-36% for the
  legacy ASR, so many of these may pass now that failed before). New node
  `recover.reingest_pending` (`pipeline/nodes/recover_orphans.py`,
  `pipe run recover.reingest_pending`) does the catalog admission (no pre-seeded ASR
  text — a clean re-transcription is the point); `orphan_segments.status` flips
  `pending_delete` → `re_admitted`. Smoke-tested `--limit 100`, then launched the full
  578,784-row run in the background (`metadata/logs/reingest_pending_20260712.log`).
  **Admission finished 2026-07-12 15:57** — 578,784 scanned, 578,784 admitted, 0
  unreadable; `orphan_segments` now shows 0 `pending_delete` rows (all flipped to
  `re_admitted`, 578,889 total incl. the earlier 105-row smoke test); `segments` grew
  662,721 → 1,241,610; 578,889 segments now have no `asr_results` row (fresh backlog for
  the drain step below).
  **Correction (2026-07-12, verified by direct catalog query)**: an earlier note here
  claimed `asr.transcribe`/`asr.agreement` "have since been run over the full backlog" —
  this was **wrong**. Verified via direct query: all 2,640,929 `asr_results` rows belong
  only to the pre-existing 662,721-segment population (per-model counts match 662,721
  exactly for the 3 active models); the 578,889 new T15 segments have **zero**
  `asr_results` rows and zero `asr_agreement` rows. No `asr_transcribe_*` log file exists
  after the 15:57 admission either — it was never actually launched. `filters_text`/
  `filters_acoustic`/`filters`/`tiers` are similarly still exactly 662,721 (unchanged) —
  none of the T15 backlog has been filtered/tiered yet. `g2p` is 517,666 (145,055 rows of
  the *old* population still lack g2p, unrelated to T15 — pre-existing gap).
  **Phase 3 embedding migration + T15's `speaker.embed` — done 2026-07-12 20:25**:
  `embed.backfill` (one-time npy→`speaker_embeddings.embedding` column migration, 662,997
  rows total) and T15's own `speaker.embed` (578,589 segments, GPU ECAPA-TDNN) were first
  tried together via `pipe run-many embed.backfill -- speaker.embed` — `embed.backfill`
  ran fine but `speaker.embed`'s discovery query never returned after 25+ minutes (zero
  log output, zero GPU worker spawned, process state `R`/high CPU) — killed and re-run as
  two **separate solo** `pipe run` processes instead (no shared-connection contention).
  `embed.backfill` solo: 457,492 remaining rows, 3,850s, 0 errors. `speaker.embed` solo:
  under the same solo invocation its discover+reuse-check scan of 578,589 rows completed
  in seconds (vs. 25+min stuck under run-many) and GPU workers spawned immediately —
  confirms the hang was `run-many`'s shared-cursor contention on a big query, not a code
  bug. 578,589 GPU-computed, 0 errors, 6,127s (1.7h), run_id=`speaker_embed_58d83b91854a`.
  `speaker_embeddings` now covers 1,241,586/1,241,610 segments (24 pre-existing
  `read_failed` rows, unrelated). **Lesson**: don't pair `embed.backfill`/`speaker.embed`
  (or any node with a large anti-join discovery query) together under `run-many` without
  testing at small scale first — run heavy discovery-query nodes solo.
- **How** (remaining, corrected order per the 2026-07-12 verification above): `asr.transcribe`
  (all 3 active backends over the 578,889 T15 segments) → `asr.agreement` → `filter.text`
  → `filter.acoustic` → `filter.decide` → `g2p` → `tier.assign`, same order as T7's chain.
  `speaker.cluster` (whole-source recompute — will re-cluster all ~1.24M segments, not just
  the new 578,889) is **independent** of ASR/filter/g2p/tier (reads only
  `speaker_embeddings`, already 100% covered).
  **Attempted 2026-07-12 22:51-23:56** (owner asked to run both `asr.transcribe` and
  `speaker.cluster` together with full dual-GPU utilization the whole time) — three tries,
  full postmortem below because the root cause took a while to nail down correctly:
  1. First attempt: `pipe run-many asr.transcribe --models canto_ft,canto_ft,qwen3_asr,qwen3_asr,sense_voice,sense_voice --devices cuda:0,cuda:1,cuda:0,cuda:1,cuda:0,cuda:1 -- speaker.cluster`
     (6-way ASR device split, smoke-tested at `--limit 20`/`--limit 300` first, 0 errors).
     Full run stalled: only 1 of 6 `asr.transcribe` workers ever spawned; `speaker.cluster`'s
     `podcast` source (538,310 embeddings) finished its clustering, then everything went
     silent for 30+ min. Misdiagnosed at the time as "`cluster_embeddings()` is a synchronous
     CPU-bound sklearn call blocking the shared asyncio event loop."
  2. Applied a `run_in_executor()` wrap around `cluster_embeddings()` to offload it to a
     thread, re-tested — **still stalled** the same way under `run-many`. Also broke solo
     performance: sklearn's joblib parallel backend generally needs the main thread to
     fork/spawn its worker pool, so calling from an executor thread pushed it into a
     single-threaded fallback — a solo `speaker.cluster` run went from ~13s/source to 5+ min
     for the same source. **Reverted** the `run_in_executor` change entirely (see the
     `pipeline/nodes/speaker.py` inline note at `run_speaker_cluster`'s clustering call).
  3. **Correct root cause, confirmed by direct observation**: after reverting, ran
     `speaker.cluster` solo (no `run-many`, no sibling to blame) — `podcast` clustered in
     ~12s as expected, but the run then sat at 300-390% CPU for 9+ minutes with **no**
     hang — it was genuinely computing. The `podcast: N segs -> M speakers` log line
     prints *before* `upsert_rows()` writes that source's rows into `speakers`; `upsert_rows()`
     does a plain `conn.executemany("INSERT OR REPLACE INTO speakers ...", tuples)` — one
     parameterised statement per row (`pipeline/catalog/catalog.py`) — which is slow at
     500k+-row scale (per-row constraint/index checking, no bulk/columnar path). This was
     never a new bug: it matches the historical T7 solo `speaker.cluster` run's 6,117s for
     662,697 segments almost exactly (~100 rows/s). The Phase 3 IO_OPTIMIZATION_PLAN
     columnar-`embedding`-column read only fixed the *read* side (which used to dominate via
     ext4 dentry-cache thrashing on per-file `.npy` opens) — the *write* side was always
     this slow, just not exercised by the small `--limit 300`/`--limit 20` smoke tests that
     made earlier attempts look "fast." **This — not asyncio contention — is why pairing
     under `run-many` stalls `asr.transcribe`'s worker spawn**: `upsert_rows()`'s multi-minute
     `executemany` call for a large source is itself a long synchronous call with no await
     points, blocking the shared event loop for its real (long) duration, same failure mode
     as `cluster_embeddings()` was wrongly blamed for.
  4. **Correction (2026-07-13, owner flagged this): the "11 hours, barely progressing" solo
     `speaker.cluster` run that followed was a false alarm — the machine was suspended for
     part of that window, inflating wall-clock `etime` without reflecting real stuck-ness.**
     Killed that run out of caution before this came to light; verified afterward it had NOT
     actually hung — real progress had landed: `podcast` 538,310/538,334 done, `rthk`
     63,057/106,341, `youtube` 361,227/596,935, all cleanly committed (`provenance='speaker_cluster'`,
     no partial/corrupt rows from the `kill -9`). So point 3's "`upsert_rows()` is
     ~100 rows/s at 500k+ scale" throughput estimate is UNRELIABLE (it was computed against
     suspend-inflated elapsed time) — the true per-row upsert cost may be much better than
     that; **do not treat "several hours for one source's upsert" as an established fact**,
     it was never cleanly measured end-to-end without a suspend in the window.
     **Decision, made together with the owner after re-asking**: retry the original
     `run-many` combined approach (owner's read: the original 30-min "stall" in attempt #1
     might also have been a measurement artifact, not proven asyncio contention — worth
     one more clean attempt now that the false lead is identified). Re-launched 2026-07-13
     10:56: `pipe run-many asr.transcribe --models canto_ft,canto_ft,qwen3_asr,qwen3_asr,sense_voice,sense_voice --devices cuda:0,cuda:1,cuda:0,cuda:1,cuda:0,cuda:1 -- speaker.cluster`,
     PID 1518406, log `metadata/logs/t15_asr_transcribe_speaker_cluster_20260713_v2.log`.
     **Lesson for future monitoring**: when a background job looks stalled after a long
     `ScheduleWakeup` gap, check for a machine suspend/resume in the window (e.g. compare
     `uptime`/last-suspend evidence, or just distrust a single huge `etime` jump) before
     concluding it's hung and killing it — verify via actual forward-progress signals
     (`/proc/<pid>/io` write_bytes delta over a short live interval, not cumulative elapsed
     time) first.
  5. **Clean re-confirmation, 2026-07-13 10:56-11:29, no suspend this time** (verified via
     `uptime -s` showing continuous boot since 2026-07-02 and a live `/proc/<pid>/io`
     `write_bytes` delta check per the lesson above): the SAME stall reproduced exactly —
     32+ min elapsed, only `canto_ft@cuda:0` ever spawned, `podcast` clustering finished at
     10:57:06 then silence, GPU0 at 0% the whole time. The `write_bytes` delta check showed
     genuine forward progress (+5MB over 8s, ~630KB/s, process state `D`) — so it is NOT a
     deadlock, but IS a genuinely slow, long-running synchronous `upsert_rows()` call
     (`podcast`'s 538,310-row `executemany`) that blocks the shared event loop for its whole
     (apparently 30+ minute) real duration, confirming point 3's original diagnosis was
     correct after all — the "11h" measurement was suspend-corrupted, but the underlying
     slow-executemany mechanism is real, just possibly faster than the corrupted number
     suggested (order of tens of minutes per large source, not hours, based on this cleaner
     partial observation). Killed (PID 1518406) after this clean confirmation.
     **Conclusion: `run-many` pairing of `asr.transcribe` + `speaker.cluster` is not viable
     until `upsert_rows()`'s large-batch cost is fixed.** Go with sequential solo execution
     (as originally decided) unless/until the batching fix below is implemented and verified.
  6. **Solo `asr.transcribe` (3 active models, 6 workers interleaved on 2 GPUs) — throughput
     far below estimate.** Launched standalone (no `speaker.cluster` pairing), confirmed all
     6 workers up and GPUs at 93-95% via `nvidia-smi`, but the combined rate stabilized at
     only **9.2/s** — projecting a ~51h ETA for the 578,889-segment backlog, vs. the original
     5-7h estimate. Killed to investigate.
  7. **Sequential-exclusive fix attempt, disproven.** Hypothesis: 3 different ASR backends
     sharing each GPU's per-device pool (`target=1` semaphore) causes expensive
     context-switching overhead. Rewrote as `run_t15_asr_sequential.sh` — each model gets
     exclusive access to both GPUs, one after another. Result: `canto_ft`-alone-on-both-GPUs
     still measured only **8.9/s** — nearly identical to the interleaved rate, disproving the
     context-switching hypothesis outright.
  8. **Red herring investigated and disproven via `nvidia-smi pmon`.** While running the
     canto_ft-alone stage, every `registry.snapshot()` debug log line showed exactly one
     device pool `in_use:1`, the other `in_use:0` — suggestive of serialization even across
     separate devices/GPUs. Directly measured with `nvidia-smi pmon -c 5` (no sudo needed):
     both canto_ft worker PIDs (one per GPU) showed 66-88% SM utilization **simultaneously**
     across all 5 samples — true parallel GPU execution confirmed. The log pattern was
     purely a timing artifact of when that debug line fires (right after a batch completes),
     not real serialization. `py-spy dump --pid <pid>` was attempted repeatedly for deeper
     stack-trace diagnosis and always failed with a permission error (`sudo -n` also fails,
     no password configured) — all diagnosis this session used `/proc/<pid>/wchan`,
     `/proc/<pid>/io`, `top -H`, and `nvidia-smi pmon` instead.
  9. **Root cause found via code reading, not experimentation.**
     `TranscribeWorker.forward_batch()` (`pipeline/nodes/asr.py`, faster-whisper/ctranslate2
     backend — used by canto_ft) does explicit sequential per-item decode
     (`return [self._transcribe_one(y16) for y16 in items]`) — no batched-tensor API — matching
     the legacy script's behaviour for golden-parity. This is an inherent architectural
     ceiling (~4.45/s/GPU), not a bug or a tunable parameter. Fully explains points 6-7.
  10. **`BatchedInferencePipeline` researched (owner instruction: use `agy`/`weir` +
      WebSearch) and ruled out.** `agy -p "..." --model "Gemini 3.1 Pro (High)"` plus two
      WebSearch queries converged on the same conclusion: `faster-whisper`'s
      `BatchedInferencePipeline` gives **no speedup** for our segments — they're already
      pre-cut to 3-20s, each fitting in ≤1 internal VAD chunk, so the pipeline's batching
      produces an effective batch size of 1. It also carries a confirmed accuracy/parity
      risk (GitHub issue #1179: "degrades transcription quality heavily" via lost
      cross-chunk context and different VAD-boundary chunking) — incompatible with this
      project's golden-parity discipline (Hard Constraint #7/#8-adjacent). A separate
      experimental method, `transcribe_batch_multiple_audios` (unmerged/recent PRs
      #1302/#1359), might suit batching multiple distinct short files together but is
      unproven — not pursued.
  11. **Decision: `canto_ft` retired (2026-07-13, owner confirmed via AskUserQuestion), same
      mechanism as `whisper_v3`'s 2026-07-10 retirement.** Cross-checked against existing
      calibration-review CER data: canto_ft measured in the same poor 17-36% band as the
      already-retired whisper_v3, vs. qwen3_asr's ~0.4% — same "slow AND inaccurate"
      profile. `ASR_MODELS["canto_ft"]["enabled"] = False`, added to
      `EXCLUDED_FROM_AGREEMENT`, `pipeline/cli.py`'s stale `--models canto_ft,whisper_v3`
      default fixed to `qwen3_asr,sense_voice`. 8 unit tests in `tests/test_asr_node.py`
      updated to reflect canto_ft's exclusion (mirroring the existing whisper_v3-exclusion
      test pattern) — full suite green afterward except 2 pre-existing, unrelated
      live-catalog staleness failures (see below). Full writeup in `DECISIONS.md`
      2026-07-13 entry. The already-running canto_ft stage (PID 1565755, ~2:08 elapsed,
      ~5.8-11万 segments' worth of now-retired-model rows written) was killed immediately
      to stop wasting GPU time once the retirement was confirmed.
  12. **New consequence, NOT yet resolved: `tier.assign`'s `auto_gold` gate has no
      confidence signal left.** `auto_gold` requires `agreement >= 0.95 AND
      canto_ft_confidence > 0.8` — canto_ft was deliberately chosen for that gate because
      it was the only active model with a real logprob-derived confidence (qwen3_asr/
      sense_voice both report a nominal `1.0` placeholder, explicitly rejected as a
      confidence source in the 2026-07-10 decision). With canto_ft retired,
      `canto_ft_confidence` is always `None` for new segments — `assign_tier()` already
      treats that as failing the gate (no code change needed, fails closed automatically),
      so **new segments cap at silver/bronze until a 2-model-agreement-only auto_gold
      threshold is adopted**. Owner's direction: default to qwen3_asr as primary, pick a
      new pure-agreement threshold (something higher than the current 0.95, since it's now
      the ONLY signal) — but **check real agreement-distribution statistics first**, don't
      guess a number. Needs a follow-up backfill/recompute of `asr_agreement` across the
      corpus excluding canto_ft (mirroring the 2026-07-10 whisper_v3 backfill), then a
      FINDINGS_ASR_AGREEMENT_THRESHOLDS.md-style distribution analysis, then an owner
      decision on the new bar. **Not started.**
  13. **A second, independent throughput bug found while restarting T15 with only the 2
      remaining models.** Running `qwen3_asr` + `sense_voice` interleaved on the same 2 GPUs
      (`--models qwen3_asr,qwen3_asr,sense_voice,sense_voice`) reproduced a suspiciously low
      combined rate (~16.9/s). `nvidia-smi pmon -c 5` showed the `sense_voice` workers at
      **0% SM across all 5 samples** — full starvation (not slow sharing) — while
      `qwen3_asr` held each device's `target=1` semaphore continuously. This differs from
      point 7's canto_ft-alone sequential test (which found NO benefit from avoiding
      contention, because canto_ft's own decode speed was already the binding constraint
      regardless): here, neither qwen3_asr nor sense_voice is inherently that slow, so the
      per-device semaphore contention genuinely is the bottleneck. Fixed by reverting
      `run_t15_asr_sequential.sh` to sequential-exclusive execution for the 2 remaining
      models (qwen3_asr, then sense_voice, canto_ft stage removed) — launched 2026-07-13,
      log `metadata/logs/t15_asr_transcribe_sequential_qwen_sense_20260713.log`.
  14. **A third throughput bug found the same day, in direct response to the owner's
      "is it possible to increase the GPU-Util for qwen-asr?" question — CLI `--batch`
      default (8) starved `Qwen3ASRWorker`'s tuned `max_inference_batch_size=64`.** The
      sequential run at the CLI default measured only ~17.4/s combined with 5-6GB/24.5GB
      VRAM per GPU (matching the batch≈8 region of the 2026-07-07 tuning curve in
      `load_model()`'s comment). Fixed by adding `--batch 64` to both invocations in
      `run_t15_asr_sequential.sh`, relaunched (PID 1617642, log
      `..._sequential_qwen_sense_batch64_20260713.log`). **Confirmed steady-state:
      42.6/s combined** (2.4× the batch-8 rate, above the 36.3/s historical benchmark),
      18-21GB VRAM + 50-62% SM per GPU, zero errors/OOM over 100k+ segments — qwen3_asr
      pass ETA ~3.5-4h, sense_voice pass after. Full writeup: DECISIONS.md 2026-07-13
      addendum. Structural fix (per-model `dispatch_batch` or raised CLI default) tracked
      as `docs/PIPELINE_REVIEW_2026-07-13.md` Issue #19 / its Phase B2.
  **2 pre-existing test failures, unrelated to the canto_ft retirement itself** (confirmed
  by direct investigation, not swept under the rug): `test_asr_results_at_least_two_
  architectures_per_segment` (live-catalog test) currently fails because ~107,668 T15
  segments got a canto_ft-only asr_results row from the killed point-11 run before canto_ft
  was excluded — will self-heal once qwen3_asr/sense_voice cover those ids from the restarted
  run. `test_manifest_build_matches_expected_corpus_totals` fails because the corpus has
  grown beyond its hardcoded baseline tolerance from ongoing T15 tier/speaker work across
  several sessions (472,167 vs. baseline 458,843 + 5000 tolerance) — pre-existing drift,
  explicitly not to be "fixed" by guessing a new baseline until T15 fully lands and a
  deliberate re-export + verification pass is done (per the test's own docstring).
  15. **A fourth throughput bug, found while investigating why `sense_voice` ran at
      about the same speed as pre-fix `qwen3_asr` (~36/s) despite its documented ~105×
      RTF**: `SenseVoiceWorker.forward_batch()` passed `batch_size_s=300` to funasr's
      `AutoModel.generate()` — a parameter that is a **complete no-op** on the code path
      we actually take (`generate()` routes to plain `inference()` since no `vad_model`
      is configured; `inference()` only reads a literal `batch_size` kwarg, default 1;
      `batch_size_s` is exclusively consumed inside `inference_with_vad()`, never
      reached). Confirmed via log grep: 100% of ~38k logged steps showed
      `'batch_size': '1'` — every item decoded one at a time regardless of the 64-item
      chunks dispatched. **Fixed** (`pipeline/nodes/asr.py`): pass `batch_size=len(items)`
      instead. 53/53 tests in `tests/test_asr_node.py` pass (fixture updated to assert
      the real batch_size is passed). Full root-cause trace: DECISIONS.md 2026-07-13
      "`sense_voice` throughput bug" entry.
      **Applied 2026-07-13, owner confirmed kill+restart**: killed the pre-fix run (PID
      1753479/1617642, 94,720/578,849 done at kill time — that work is preserved, the
      restart resumed via the normal idempotent discovery anti-join, not from zero).
      Restarted `run_t15_asr_sequential.sh` (new PID 1772752); `qwen3_asr` correctly
      re-discovered 0 remaining rows and no-opped in ~1s (confirms its earlier pass was
      already complete), `sense_voice` picked up the remaining 479,264 segments. Log
      confirms the fix took: every `forward_batch()` call now shows `'batch_size': '64'`
      (was `'1'`), `rtf` dropped to ~0.001 (was ~0.010-0.017). Measured steady throughput
      **~87.8/s combined — 2.4× the pre-fix ~36/s**. Note: `nvidia-smi` showed low GPU
      utilization (0-4%) right after restart despite this throughput — the bottleneck
      may have shifted from GPU compute (now near-instant per 64-batch) to CPU-side
      audio load/feature-extraction between batches; not investigated further this
      session, worth a look if sense_voice throughput needs to go even higher later.
      Phase C auto-runner relaunched pointed at the new PID (background task `bbdphc24e`,
      confirmed waiting on 1772752) — will fire Phase C1+C2 once this pass finishes.
  16. **The "shifted bottleneck" from point 15 root-caused and fixed 2026-07-14 (owner-
      approved full suite, applies to FUTURE runs — the live T15 pass was left to drain
      on the old code)**: the low GPU utilisation was the decode+resample CPU stage
      running strictly serialised with the GPU forward (no overlap at either the
      supervisor or worker level). Three-part fix in `pipeline/nodes/asr.py` + CLI:
      (a) `_load_and_resample()` swapped scipy.resample_poly → `soxr` HQ (3.8× faster
      resample, measured on real corpus files); (b) `worker_main()` restructured into a
      3-stage threaded producer-consumer pipeline (stdin reader → preprocess pool →
      GPU+emit, bounded queues) so decode of batch N+1 overlaps the forward of batch N;
      (c) supervisor `worker_loop()` double-buffers dispatch (`--prefetch`, default 2,
      pool acquired around the send only; results matched by task_id), `--io-workers`
      default 8→16 passed through. 56/56 node tests green incl. 3 new supervisor
      regression tests; full-suite failures identical to the T15-writer-lock baseline.
      Full writeup: DECISIONS.md 2026-07-14. **Live-GPU validated the same day** (the
      point-17 suspend-wedge kill freed the writer lock earlier than planned):
      `--limit 512` → 512/512, 0 errors on both GPUs, then the T15 sense_voice remainder
      relaunched on the new code path (see point 17). Single-pass dual-model shared
      decode deliberately deferred until post-fix throughput data exists (owner decision
      2026-07-14).
  17. **T15's sense_voice pass wedged by a machine suspend/resume (~10:43 2026-07-14)
      and was killed a third time (owner-approved) — remainder relaunched on the new
      pipelined code.** The cuda:0 worker's main thread spun at ~85% of a core with 0%
      GPU after resume (wedged CUDA context); the supervisor burned one 64-segment batch
      per 600 s read-timeout from 10:52:59 (12 cycles / 768 segments errored — no
      asr_results rows written for those, so discovery re-surfaces them; zero loss).
      cuda:1's shard had already finished cleanly. Killed auto-runner first, then the
      T15 tree (wedged worker needed SIGKILL); verified lock free + 380k completed rows
      intact; ran the --limit 512 live validation of the point-16 code; relaunched the
      97,488-segment remainder on the new code (~85/s early rate on the longest-duration
      tail of the queue, log `metadata/logs/t15_sense_voice_remainder_pipelined_20260714.log`);
      re-pointed the Phase C auto-runner at the new supervisor PID. Full narrative:
      DECISIONS.md 2026-07-14 (second entry). **Future work**: supervisor should kill+
      respawn a worker after N consecutive read timeouts instead of feeding a wedged
      subprocess forever; avoid suspending the machine during GPU batch runs until then.
  **Future work worth doing** (not attempted, out of scope for tonight): batch
  `run_speaker_cluster`'s per-source `upsert_rows()` call into chunks (e.g. 5,000 rows,
  matching the `batch_size` convention every other node in this codebase already follows)
  instead of one `executemany` per whole source (538k+ rows in one call for `podcast`) —
  `speaker.cluster` is the one node that doesn't chunk its writes. This would very likely
  both fix the `run-many` starvation (each chunk's blocking window shrinks from tens of
  minutes to a fraction of a second) and make solo runs faster/more resumable. Worth doing
  before the NEXT time `speaker.cluster` needs to run concurrently with anything.
- **Effort**: large, mostly unattended GPU wall-clock (asr.transcribe) + several hours CPU
  wall-clock (speaker.cluster's upsert cost, unattended). **Depends on**: GPU availability;
  DuckDB writer free.
- **Follow-up**: once this backlog is scored by the current pipeline, compare its
  pass-rate/tier distribution against the original `pending_delete` classification to
  quantify how much the old legacy-ASR rejection was actually a false-negative problem —
  worth a `PROGRESS.md`/`DECISIONS.md` note either way.

### T15 — addendum: full downstream chain confirmed drained, 2026-07-16
Verified live (not just claimed) via `pipe catalog verify` (17/17 PASS) and direct catalog
query: `asr_agreement`/`filters`/`tiers` all now sit at the `segments` ceiling
(1,241,610 rows each) — the `asr.agreement → filter.text → filter.acoustic → filter.decide
→ tier.assign` chain (interrupted mid-session 2026-07-14 by the `filter.decide` OOM, see
DECISIONS.md 2026-07-14) fully drained after that fix. `speaker.cluster` also re-ran:
speakers 11,679 → 14,330. `g2p` sits at 780,219/1,241,610 — this LOOKS like a lag against
`segments`, but is not one: g2p's real population is `filters.pass = TRUE` (also exactly
780,219), not all segments (most segments legitimately fail some filter gate and never need
g2p at all — see `pipeline/nodes/g2p.py`'s discovery SQL). **Correction 2026-07-17**: verified
live via `discover()` — g2p backlog is genuinely **0**, fully caught up. The "lag" framing in
the original addendum was a miscalculation (comparing against the wrong denominator), not a
real gap; no g2p work is outstanding. T15 moved to Done in full this session (2026-07-17) —
the DAG-drain work described above was already complete, this was purely a bookkeeping fix
(the Tier 3 list still had prima facie the entire investigation/postmortem sitting there as
if still open).

### T19. `calibrate.serve` — 'rejected' propagation fix + one-click Mandarin
flag button — done 2026-07-15
- **What**: found while implementing the Mandarin flag request — `record_decision('rejected',
  ...)` was recorded in `calibration_review` but never read by `manifest.py`'s eligibility
  join (`segments`/`asr_agreement`/`g2p`/`filters`/`tiers` only). A human "Reject" click had
  zero effect on what shipped in the manifest.
- **Done**:
  1. `record_decision()`: `decision == 'rejected'` now also directly upserts
     `tiers.tier='excluded'` (provenance `calibrate_reject`), mirroring the existing
     `'verified'`→`'gold'` direct write and its rationale (sidesteps `tier.assign`'s
     `provenance='tier_assign'`-scoped anti-join). Applies to every rejection, not just
     Mandarin-flagged ones.
  2. `pipe calibrate serve`: new one-click **Mandarin** button (`M` key) submits
     `decision='rejected', flag_reason='mandarin'` (`MANDARIN_FLAG_REASON` constant) — for
     segments that surface for QA but are actually non-HK-Cantonese. Both excludes (via the
     fix above) and records the reason in one click.
  3. `summary_stats()`'s `top_flag_reasons` leaderboard broadened to include `'rejected'`
     rows with a reason, not just `'flagged'`, so Mandarin rejections surface in triage.
  4. `'flagged'` (generic pipeline-bug report) unchanged — still non-exclusionary, free text.
  5. (2026-07-16 follow-up, same task) **Sample-options controls added to the browser UI**
     itself — a new "Sample:" control group in the topbar (tier / min-agreement / code-switch),
     the web equivalent of `pipe run calibrate.sample --tier/--min-agreement/--code-switch`.
     Scopes both the manual "↻ Refill" button and the auto-refill-on-empty-queue path (so a
     reviewer's focused session, e.g. tier=auto_gold + code-switch=only, doesn't get diluted
     by an unscoped auto top-up once they run dry). New `_parse_sample_options()` helper
     (shared by `/api/refill` POST body and `/api/next`'s query params) validates tier against
     `_VALID_QA_TIERS` and min_agreement as a float, returning a JSON 400 on bad input instead
     of silently sampling something unintended. Distinct from the existing batch/source/order
     controls, which only filter browsing of already-queued items, not what gets queued.
- **Also done same session**: deleted the two T16 backfill safety-net DB backups
  (`corpus.duckdb.pre_agreement_t16_backup` / `.pre_tier_t16_backup`, 8.6GB) — not
  git-tracked, T16 already verified and documented, no further use.
- **Tests**: 3 new in `tests/test_calibrate_node.py` (357/357 total). Sample-options controls
  verified via a live smoke test against a scratch catalog (page renders the new controls,
  scoped refill queues the correct tier/code-switch population, invalid tier/min-agreement
  return a 400 with a JSON error body) — no dedicated pytest suite exists for
  `calibrate_server.py`'s HTTP layer (it's an interactive tool, tested live per CLAUDE.md).
- **Not done**: no retroactive re-tiering of any segment already 'rejected' before this fix
  landed (none exist yet — T1 pilot QA, the main consumer of this UI, hasn't started).

### T18. Code-switching-aware export cut + QA oversampling — done 2026-07-15
- **What**: T16's distribution analysis found code-switched segments (`filters.english_ratio
  > 0`, 220,364 segments / 17.7% of the corpus) clear ASR-agreement thresholds far less
  often than pure-Cantonese segments (e.g. 18.8% vs 48.5% at agreement≥0.90) — a systematic
  AR-vs-CTC English-token transliteration divergence, not necessarily a quality signal.
  Owner decision (AskUserQuestion): keep ONE unified corpus (no physical fork), add
  on-demand tooling for training-side filtering + QA focus, with a **10x** QA oversampling
  multiplier (owner-specified) for code-switch segments.
- **Done**:
  1. `pipeline/nodes/manifest.py`: `--code-switch {only|exclude}` cut flag added to
     `manifest.build`/`manifest.export` (`discover()`/`build_manifest()`/
     `run_manifest_build()`/`run_manifest_export()`/`_export_tag()` all threaded through;
     `CODE_SWITCH_CONDITIONS` dict maps `only`→`english_ratio > 0`, `exclude`→`= 0`).
     Combinable with `--min-tier`/`--min-agreement`. 8 new tests in
     `tests/test_manifest_node.py`. Real exports produced: `manifest_codeswitch_only.jsonl`
     (84,770 entries / 226.6h / 3,692 speakers) and `manifest_codeswitch_exclude.jsonl`
     (522,005 / 1,122.7h / 8,728 speakers) — sums to the full 606,775-entry pool.
  2. `pipeline/nodes/calibrate.py`: `CODE_SWITCH_QA_MULTIPLIER = 10.0`;
     `recommended_sample_n(..., code_switch=True)` scopes population to
     `english_ratio > 0` AND multiplies the tier's base QA rate by 10x (capped at 100%);
     `discover()`/`run_calibrate_sample()` gained a matching `code_switch` param
     (`'only'`/`'exclude'`), wired into `pipe run calibrate.sample --code-switch
     {only|exclude}` and the `run-many` adapter. 8 new tests in `tests/test_calibrate_node.py`.
  3. CLI: both `manifest.build`/`manifest.export`/`calibrate.sample` subcommands expose
     `--code-switch`.
- **Result** (not yet acted on — no QA batch queued this session, left for the owner since
  it consumes real review time): recommended code-switch-scoped QA sample sizes —
  auto_gold 1,250 (15% of 8,332), silver 10,366 (40% of 25,907), **bronze 50,524 (100% —
  the 10x multiplier hits the rate cap exactly at bronze's 10% base rate, i.e.
  "review all of them," not realistically a near-term QA target as a full pass; a smaller
  pilot batch like the 2026-07-11 precedent's 300-per-tier is the practical next step).
- **Tests**: 354/354 passing (16 new: 8 manifest + 8 calibrate).
- **Not done**: no QA batch actually queued (deliberately left to the owner to trigger via
  `pipe run calibrate.sample --tier <t> --code-switch only --n <N>`).

### T16. Rebuild the `auto_gold` gate for the 2-model era — done 2026-07-15
- **What**: `tier.assign`'s `auto_gold` gate required `canto_ft_confidence > 0.8` — always
  `NULL` for new segments since canto_ft's 2026-07-13 retirement, failing closed. All 5
  steps completed same session: (1) normalization fix (done 2026-07-13, see Issue #20),
  (2) full-corpus `asr_agreement` backfill (1,241,610 ids, canto_ft excluded + normalized
  text, `scratchpad/backfill_agreement_t16.py`, 84.8s, `text_verified` preserved for all
  58 pre-existing gold rows), (3) distribution analysis (agreement × dnsmos crosstab +
  code-switch-split breakdown, presented to owner), (4) owner picked the "Balanced"
  bundle via AskUserQuestion: `AUTO_GOLD_AGREE_MIN` 0.95→**0.92**, confidence gate
  replaced with `AUTO_GOLD_DNSMOS_MIN=3.5` (new, `filters.dnsmos`), silver/bronze
  unchanged (0.85/0.70) — `pipeline/nodes/tier.py`'s `assign_tier()` signature changed
  `canto_ft_confidence` param → `dnsmos`, 13/13 tests updated+passing in
  `tests/test_tier_node.py`, (5) `tiers` re-derived corpus-wide
  (`scratchpad/backfill_tier_thresholds_t16.py`, 5.6s, 0 human-gold rows touched).
- **Result** (manifest-eligible pool, filters.pass=TRUE): `auto_gold` 73,252→**279,195**
  segments (151.9h→**640.9h**, +322%), `silver` 158,087 (333.9h), `bronze` 169,435
  (374.4h), `gold` unchanged at 58. Full pool 606,775 entries/1,349.3h/9,023 speakers
  (up from 590,410/1,317.0h — normalization alone recovered ~32h even before the gate
  change). `manifest.export`/`report.build` re-run (default + `--min-tier
  auto_gold/silver/bronze` cuts, each also copied to `metadata/DATASET_REPORT_<tier>.md`).
  `tests/test_catalog.py`'s `test_manifest_build_matches_expected_corpus_totals` baseline
  updated per its own docstring's "update only after an intentional, verified re-run" rule.
- **Provisional**: T1 pilot QA (still 0/~900 reviewed) has NOT yet validated this gate's
  real precision — owner explicitly chose to unblock now rather than wait, revisit the
  0.92/3.5 numbers once T1 ground truth exists.
- **Follow-up spun out as T18** (done same day, see T18 below): code-switching handling —
  segments with `english_ratio > 0` clear agreement thresholds far less often (e.g. 18.8%
  vs 48.5% pure-Cantonese at agreement≥0.90) since the two ASR backends diverge on
  English-token transliteration, not necessarily quality.

### T17. `filter.acoustic` GPU offload via onnxruntime-gpu — done 2026-07-14
- **What**: `filter.acoustic` (SNR+DNSMOS) was CPU-only and had hit a scaling wall —
  `--workers 4→8` doubled CPU usage (~18.8→~35.3 cores) for almost no throughput gain
  (~21→~21.6/s), while both RTX 4090s sat idle. Swapped in `onnxruntime-gpu` (system had
  only CPU `onnxruntime` installed) so DNSMOS ONNX inference runs on GPU.
- **How**: `uv pip uninstall onnxruntime && uv pip install onnxruntime-gpu`; new
  `--gpu 0,1` flag on `pipe run filter.acoustic` (comma-separated CUDA device ids,
  round-robinned across the worker pool; omitted = unchanged CPU-only path, so this is
  backward compatible). `LD_LIBRARY_PATH` for worker subprocesses points at torch's
  pip-bundled CUDA 13 runtime (no separate system CUDA toolkit needed). Code:
  `pipeline/nodes/filter.py` (`_build_capped_dnsmos`, `AcousticWorker`, `worker_main`,
  `run_filter_acoustic`) + `pipeline/cli.py`. Full detail + correctness verification
  (GPU vs CPU sig_mos/ovrl_mos exact match on 5 real segments, |Δ|=0.0) in
  `DECISIONS.md` 2026-07-14.
- **Result**: `--workers 8 --gpu 0,1` → ~115-122/s (~5.5× the CPU baseline). Tried
  `--workers 16 --gpu 0,1` — no further gain (~115/s, same ballpark), GPU util stayed
  low (10-25%) at both worker counts — the bottleneck past 8 workers is the supervisor's
  asyncio dispatch loop / IPC round-trip, not GPU or CPU compute. **8 workers is the
  practical ceiling for the current dispatch pattern.**
- **Follow-up not done**: no regression test yet for the `--gpu` code path (add once a
  default-on-GPU decision is made for future runs — currently opt-in only). If the
  asyncio dispatch loop is ever revisited for other worker-pool nodes, this is a second
  data point (after label_prosody-style nodes) that per-batch IPC overhead caps
  throughput around ~8 concurrent workers regardless of backend speed.

### T7. Drain the 2026-07-11 ingest round's downstream backlog (N3) — done 2026-07-12
`run_t7_chain.sh` (resumed #4) ran the full waterfall — `filter.decide` (44,026 decided,
32,834 passed) → `g2p` (32,834 converted, 100% accepted) → `speaker.embed` (44,026
GPU-computed, 0 legacy-reused, 417s) → `speaker.cluster` (whole-source recompute, 3
sources, 662,697 segments → 11,679 speakers, 6,117s) → `tier.assign` (44,084 tiered:
gold=58, auto_gold=4, silver=640, bronze=16,921, excluded=26,461). Completed
2026-07-12T14:22:39. See `metadata/logs/t7_chain_20260711.log`. **Note**: this run's
`speaker.cluster` pass predates T15's reingest admission (which finished 15:57, after
this chain completed) — it does NOT include any of the 578,889 re-admitted segments;
those still need their own `speaker.embed` pass before the next `speaker.cluster` run.
T6 (manifest re-export) is now unblocked.

### T8. Finish `conn=` injection on remaining nodes (Issue #5) — done 2026-07-07
Found already fully complete while investigating T14 (2026-07-12) — this entry was stale.
Per `docs/ORCHESTRATOR_PLAN.md` line 3: **23/23 call sites done**, `pipe run-many`
validated live at full backlog scale same day (`label.music`+`filter.acoustic`, then
`asr.transcribe`+`filter.acoustic`). All 23 registered in `RUN_MANY_ADAPTERS`
(`pipeline/cli.py`), regression-tested in `tests/test_run_many.py` (29 tests). No
remaining nodes need `conn=` injection.

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
