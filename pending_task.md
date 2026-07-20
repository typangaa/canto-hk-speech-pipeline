# Pending Tasks

> **Maintenance rule**: this file must be updated whenever a task here is completed —
> move it to the "Done" section (with the completion date + commit/ref if applicable)
> instead of deleting it, and update `docs/archive/PIPELINE_REVIEW_2026-07-11.md` §6 disposition
> table if the task closes one of that doc's numbered issues. Keep this file's Tier
> ordering current if priorities shift. See `CLAUDE.md` for the standing instruction to
> keep this file in sync.

Source: round-2 post-execution review of `docs/archive/PIPELINE_REVIEW_2026-07-11.md` §6,
2026-07-11. Re-derive priorities from that doc if this file and it ever disagree.

---

## 🔴 Tier 1 — data-trust-critical, do first

### T1. Pilot QA batch review (Issue #15)
- **Update 2026-07-19 (stale-excluded prune — see T25 in Done)**: found 172 (later 221 at
  a second look, after 72 more offline decisions were flushed) queue rows still `pending`
  whose segment had since been auto-excluded by T20/T22/T23's gate catch-up — these can
  never become `gold`, so reviewing them was wasted effort. Pruned via the new
  `prune_excluded_pending()` node function (`pipe calibrate prune-excluded` CLI, or the
  "🧹 Prune excluded" button now in `pipe calibrate serve`'s UI). Queue is now **299
  total, 243 reviewed, 56 pending** (all genuinely reviewable — 0 excluded-tier pending
  left). Run this anytime a queued batch might have drifted stale against later gate
  changes; it's idempotent and safe to click even when there's nothing to prune.
- **Update 2026-07-18 (queue reset — see "Near-incident" in Done)**: owner asked to empty
  the accumulated pending queue so `calibrate.sample`/Refill can re-sample fresh going
  forward. `calibration_review` is now **171 total rows, 0 pending** (149 `verified` / 21
  `rejected` / 1 `flagged` — all prior review decisions preserved intact, see the
  Near-incident writeup for why a bulk delete of `pending` rows nearly discarded 113 of
  them and how it was recovered). All the tier/code-switch progress numbers below (2,342
  pending, code-switch 1,237, etc.) are now **stale** — the queue starts fresh from here;
  run `pipe calibrate progress` for current numbers before resuming review.
- **Update 2026-07-18 (see T20/T21 in Done)**: found and fixed a real gap while reviewing
  this queue — 44 of the currently-pending segments were already auto-excluded by T20's
  new audio-based Mandarin gate (`filters.fail_reason='mandarin_audio'`) before you get to
  them; reviewing them is no longer load-bearing for `text_verified` but still useful as a
  sanity-check on the `labels_lang` classifier if you want to spot-check a few. New
  batches can now be biased toward the riskiest segments instead of pure-random within a
  tier: `pipe run calibrate.sample --order agreement_asc` (composable with
  `--tier`/`--code-switch`), also exposed as a "Sample:" dropdown in `pipe calibrate
  serve`. The browser UI's separate "Order" dropdown (`agreement_asc`/`agreement_desc`)
  already let you re-sort items *already queued* — that was pre-existing, unrelated to
  this fix.
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

(none currently — T23 done 2026-07-18, see Done section)

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

### T28. Report `tied`-confidence Jyutping tokens upstream to canto-hk-g2p — deferred, waiting on more human-verified data

**What**: canto-hk-g2p v2.0.0's `convert_candidates()` exposes per-token `confidence`
("certain"/"ranked"/"tied") + `source`. `"tied"` = a rime-cantonese arbitrary tie-break with
no real context-aware disambiguation signal — the highest-risk case for a wrong reading.

**Decision (2026-07-19)**: disambiguation logic belongs in canto-hk-g2p itself, not as a
pipeline-local `user_dict` patch (would duplicate/drift from upstream dictionary data). Any
genuine gap gets reported as a GitHub issue on `typangaa/canto-hk-g2p` — problem listing
only (token + example sentences + candidates), no suggested "correct" reading claimed.

**Status**: ran a one-time offline audit (scanned `asr_agreement` where `text_verified=true`,
collected every `tied`-confidence token via `_G2P.convert_candidates()`). Only **189**
human-verified segments exist right now (gold tier via `pipe calibrate serve` — small,
calibration is still early) — found 46 distinct tied tokens, 9 of them seen 2+ times
(`嗰個` 9x, `呢啲` 7x, `使用` 3x, plus `成日`/`同行`/`立場`/`人物`/`教堂`/`入邊` 2x each).
Owner decided to **wait for more human-verified data** before opening the issue — 189
sentences is too small a sample to be confident which findings are real vs. noise.

**To resume**: the audit script + full report + a draft GitHub issue body were written to
the session scratchpad (ephemeral, not in this repo) — the script is short and reruns in
seconds even against the full corpus (read-only via `connect_ro()`, no catalog writes), so
just re-ask for the audit once the gold-tier verified count has grown meaningfully. No code
changes needed in this repo either way — this is a reporting-only task, not an
implementation.

**Not done**: GitHub issue not opened, no `user_dict` override added anywhere (by design).

---

## Suggested execution order

```
Do now (independent):      T1 (human QA, owner-paced, code-switch focus — 2026-07-17)
Schedule into P6 proper:   T9 (depends on T1), T10, T14 levers (3)+(4) live-validation
Pulled by training needs:  T13
```
(updated 2026-07-19 — the T23+T24 combined follow-up (filter.decide re-run + full
manifest re-export) is DONE, see the "T23+T24 follow-up" Done entry below; nothing left
queued behind it. T24 (canto-hk-g2p 1.5.0→1.9.0 upgrade + phonological validation +
corpus-wide reprocess) and T23 (label.suite chain-wiring) both in Done. T20/T21/T22 +
the calibration_review near-incident moved to Done 2026-07-18. T5/T6/T16/
T15-remaining-chain moved to Done in the prior 2026-07-17 update; see Done section for
what actually landed and when.)

---

## Done

> Entries dated 2026-07-16 and earlier rotated out to `DECISIONS.md` (2026-07-19 cleanup pass) to keep this file to the recent working window — same rotation `PROGRESS.md` uses. Full history is in `DECISIONS.md`, chronological, nothing lost.

### T29. Upgrade `canto-hk-g2p` 2.0.0 → 2.1.0 (借音字 phonetic-loan alias layer) — done 2026-07-19
- **What**: upstream v2.1.0 (now on PyPI) adds a hand-curated alias table
  (`data/variant_words.tsv`, ~20 entries) that corrects common sound-borrowing
  miswritings — e.g. `訓覺` used to resolve to `訓`'s own native reading `fan3 gok3`
  instead of the intended `瞓覺` reading `fan3 gaau3`. Purely additive: new
  `source="variant_alias"` tag on the existing 5-tuple shape, no unpack changes.
- **Reinstalled** `uv pip install -e ~/Documents/canto-g2p` to refresh the dist-info
  version metadata (was stale at 2.0.0 even though the compiled extension had already
  moved — same drift pattern as T26).
- **`pipeline/nodes/g2p.py`**: docstring-only change (new dated upgrade note); no code
  change needed — `_convert_for_moss()`'s starred unpack and `candidate_preview()`'s
  generic `source` passthrough already handle any new source-tag value.
- **Verified**: `text_to_jyutping("訓覺")` now returns `"fan3 gaau3"` (was `"fan3 gok3"`
  under 2.0.0); `convert_candidates("訓覺")` confirms `source="variant_alias"`. Full
  suite 476 passed / 1 failed — the 1 failure
  (`test_manifest_build_matches_expected_corpus_totals`, off by 4 rows against its
  live-catalog baseline) is the known baseline-drift pattern, unrelated to this
  upgrade (no g2p rows were reprocessed this round; see "Not done" below).
- **Not done**: corpus-wide `g2p` reprocess to retroactively correct the ~20 words'
  worth of existing rows — same "not automatically revisited by anti-join discovery"
  caveat as T24's v1.7.0 note; still queued, not scheduled this round.

### T27. Surface polyphone-candidate ambiguity in `pipe calibrate serve`'s live preview — done 2026-07-19
- **What**: T24's "Not done" list item 2 ("wire `convert_candidates()` into `pipe
  calibrate serve`'s UI to surface polyphone alternatives during human review") —
  closed out now that T26 confirmed the library's v2.0.0 `confidence`/`source` fields
  work end-to-end. Before this, the live Jyutping preview in the review textarea only
  ever showed the rank-0 reading `g2p_one()` would commit — a reviewer had no way to
  tell "g2p is confident" apart from "g2p silently picked one of 4 candidates via an
  arbitrary tie-break," since both rendered identically (a single string).
- **`pipeline/nodes/g2p.py`**: added `candidate_preview(text)` — calls
  `_G2P.convert_candidates()` (same Pipeline singleton as `g2p_one()`, so this can
  never drift from what the DAG node actually commits) and returns one
  `{token, candidates, confidence, source}` dict per Cantonese token with 2+ known
  readings; unambiguous tokens (single reading, English, punctuation) are omitted.
  The `g2p` node's own write path (`_convert_for_moss()`) is untouched — this is
  purely an additive read-only helper for the calibrate UI.
- **`pipeline/nodes/calibrate.py`**: `jyutping_preview()` now also calls
  `candidate_preview()` and returns the result as a new `ambiguous` key alongside the
  existing `jyutping`/`valid_fraction`/`accept`/`bad_tokens` fields.
- **`pipeline/tools/calibrate_server.py`**: `/api/g2p_preview` needed no change (already
  forwards whatever `jyutping_preview()` returns verbatim). Added `.amb-token` CSS
  (dotted underline; `--warn` amber for `confidence="tied"` — rime-cantonese's raw
  arbitrary tie-break, no real preference signal, most worth a second look; `--muted`
  grey for `confidence="ranked"` — a real context-aware lean, lower priority) and
  updated `refreshJyutpingPreview()`'s JS to render an "— ambiguous: …" segment with a
  hover tooltip showing `confidence via source: candidate / candidate`.
- **Verified**: `curl /api/g2p_preview?text=重` (a known polyphone: 重要"heavy" vs
  重複"repeat") returns `ambiguous: [{"token": "重", "candidates": ["cung5", "cung4",
  "zung6", "cung6"], "confidence": "ranked", "source": "tojyutping_tiebreak"}]` against
  a live `pipe calibrate serve` instance; `text=心臟病中風` (resolves via whole-word
  dictionary entries, no per-character ambiguity) returns `ambiguous: []`. 5 new tests
  (`test_g2p_node.py`: empty/unambiguous/polyphone/English+punct-exclusion cases for
  `candidate_preview()`; `test_calibrate_node.py`: `jyutping_preview()`'s `ambiguous`
  key, including the exact-dict-equality empty-text test). Full suite 477 passed / 0
  failed (was 472 before this change).
- **Not done** (out of scope this round): using `confidence`/`source` to build the
  `user_dict` override candidate list (T24's "Not done" item 1) — that's a separate,
  offline corpus-wide audit script, not a UI change; still queued.

### T26. Upgrade `canto-hk-g2p` 1.9.0 → 2.0.0 (breaking tuple-arity change) — done 2026-07-19
- **What**: upstream `canto-hk-g2p` v2.0.0 (closes
  [#12](https://github.com/typangaa/canto-hk-g2p/issues/12) and
  [#13](https://github.com/typangaa/canto-hk-g2p/issues/13), both filed during T24) added
  two new trailing fields — `confidence`, `source` — to `convert_detailed()`'s per-token
  tuples, `(token, jyutping, lang)` → `(token, jyutping, lang, confidence, source)`.
  `pipeline/nodes/g2p.py`'s `_convert_for_moss()` unpacked that tuple by fixed arity
  (`for _, jp, lang in tokens`), so every `g2p_one()` call started raising
  `ValueError: too many values to unpack (expected 3)` — caught internally and logged,
  degrading silently to `jyutping="" / valid_fraction=0.0` (a hard reject) for every
  segment, not a crash. Surfaced via 4 failing tests
  (`test_g2p_node.py::test_text_to_jyutping_basic` /
  `test_text_to_jyutping_excludes_english` / `test_g2p_one_valid_cantonese_text`,
  `test_calibrate_node.py::test_jyutping_preview_valid_cantonese_text`).
- **Fix (`pipeline/nodes/g2p.py`)**: unpack with a starred catch-all —
  `for _, jp, lang, *_ in tokens` — per the library's own CHANGELOG migration guide.
  `confidence`/`source` are not consumed here; same "calibration-UI feature, not a
  `g2p`-node concern" scoping as `convert_candidates()` in T24 (surfacing polyphone
  confidence to a human reviewer is a `pipe calibrate serve` UI feature, still not
  started — see T24's "Not done" list). Module docstring updated with a dated note.
- **Reinstall**: `uv pip install -e ~/Documents/canto-g2p` — the source repo's compiled
  extension and `pyproject.toml` version had already moved to 2.0.0, but the `.venv`'s
  editable-install dist-info metadata was stale at 1.9.0 until reinstalled;
  `canto_hk_g2p.__version__` now correctly reads `"2.0.0"`.
- **Verified**: full suite 472 passed / 0 failed (was 468 passed / 4 failed before this
  fix). No corpus-wide reprocess needed this round — this was a pure regression fix
  restoring prior behaviour, not a data-quality improvement like T24's tie-break/
  phonology changes, so there's no `provenance` reset to run.

### T25. Prune stale excluded-tier rows from the QA queue + web UI button — done 2026-07-19
- **What**: T23+T24 follow-up's `filter.decide` re-run (5,288 newly-excluded segments)
  left the `calibration_review` queue with rows still `decision='pending'` whose segment
  had since flipped to `tiers.tier='excluded'` — these can never become `gold` (a
  `verified` decision only flips `tiers.tier`, it doesn't undo an existing exclusion
  reason), so reviewing them was wasted human effort. Owner asked for a query to clean
  these up, then asked whether a web UI button could do it going forward.
- **One-off live cleanup**: flushed 72 unflushed offline decisions first (never delete
  `pending` rows without flushing immediately before — the exact lesson from the
  2026-07-18 near-incident, see that Done entry), then deleted 172 stale rows via a direct
  `DELETE ... WHERE decision='pending' AND tiers.tier='excluded'` join query. Queue went
  471 total/300 pending → **299 total/56 pending** (all genuinely reviewable).
- **Made it a permanent tool**: added `prune_excluded_pending(conn)` (the delete query
  itself, returns `{removed, pending_before, pending_after}`) and
  `run_calibrate_prune_excluded(*, conn=None, in_path=None)` (flushes the offline decision
  buffer first via `run_calibrate_flush_pending()`, then prunes — same near-incident
  safeguard as the one-off cleanup, now baked into the function) to
  `pipeline/nodes/calibrate.py`.
- **CLI**: `pipe calibrate prune-excluded` (`pipeline/cli.py::cmd_calibrate_prune_excluded`).
- **Web UI**: "🧹 Prune excluded" button next to "↻ Refill" in `pipe calibrate serve`
  (`pipeline/tools/calibrate_server.py`) — `POST /api/prune-excluded`, confirm dialog
  before running, toast shows rows removed + decisions flushed. Also clears the server's
  in-memory `_local_decisions` overlay for any id that got flushed during the call (the
  flush bypasses this server's own `/api/submit` path, so without this the overlay would
  keep serving stale 'pending' for those ids until the next server restart — the same
  class of bug as the 2026-07-18 near-incident's residual double-counting issue, this time
  fixed at the source instead of needing a manual restart).
- **Tests**: 4 new tests in `tests/test_calibrate_node.py` (`prune_excluded_pending`
  removes only pending+excluded rows and no-ops when nothing's stale;
  `run_calibrate_prune_excluded` flushes a buffered decision before pruning, spares a row
  that was buffered-verified even though its tier is excluded, and no-ops on an empty
  buffer). Full suite 472/472 passing.
- **Live-restarted** `pipe calibrate serve` (old PID 374410 → new PID 652205) to pick up
  the new endpoint/button, smoke-tested `POST /api/prune-excluded` against the live
  catalog (returned `{"removed": 0, ...}` — correctly idempotent, nothing left to prune
  since the one-off cleanup already ran).
- **Not committed to git yet** — implementation only, no commit/push requested this
  session.

### T23+T24 follow-up. `filter.decide` re-run + full manifest re-export — done 2026-07-19
- **What**: T23's label.suite catch-up and T24's g2p reprocess both landed on the same
  day, each leaving its own pending "make it actually reach filters.pass/manifest.jsonl"
  follow-up (see both entries below). Closed both in one combined pass rather than
  re-exporting twice for two catalog changes landing the same week.
- **`lang_label_checked` reset turned out to be unnecessary**: re-reading
  `DECIDE_DISCOVER_SQL` (`pipeline/nodes/filter.py`) showed it already self-heals —
  `(COALESCE(f.lang_label_checked, FALSE) = FALSE AND ll.id IS NOT NULL)` re-triggers a
  decided row the moment a `labels_lang` row lands for it later, no manual `UPDATE`
  needed (unlike T20's original rollout, which genuinely needed one because it was
  deploying new gate *logic* onto rows already marked `lang_label_checked = TRUE` from
  before the gate existed — a different scenario). Verified live: querying
  `DECIDE_DISCOVER_SQL` directly showed 785,680 rows already in scope before touching
  anything. The "needs a manual reset" note in T23's write-up below was an
  overcautious inference, not re-checked against the live query at the time — corrected
  here.
- **`filter.decide` re-run**: `pipe run filter.decide`, 785,680 rows re-decided in 21s
  (37,724/s at completion), 319,427 passed this batch, 0 errors,
  `run_id=filter_decide_7df33c041aff`. Net effect on `filters`: pass 768,663→**763,375**
  (-5,288), driven entirely by the three audio-language gates now seeing labels_lang for
  segments they never had it for before: `mandarin_audio` 10,940→**13,979** (+3,039),
  `other_language_audio` 546→**2,322** (+1,776), `english_audio` 70→**543** (+473). Every
  other fail_reason (snr/mandarin_ratio/text_too_short/dnsmos/english_ratio/
  text_too_long/dnsmos_error) unchanged, as expected — those gates don't read
  `labels_lang`. `catalog verify` 17/17 PASS before and after.
- **Manifest re-export**: all 7 cuts re-exported (default + 6 derived, same set as
  T20/T22's rollout) —
  - default (`manifest.jsonl`/`train.jsonl`/`val.jsonl`): 596,089→**594,010** entries,
    1331.6h→**1328.6h**, 8,682 speakers
  - `--min-tier auto_gold`: **274,848** / 634.0h
  - `--min-tier silver`: **427,757** / 959.6h
  - `--min-tier bronze`: **594,010** / 1328.6h (same population as default)
  - `--code-switch only`: **83,510** / 224.3h
  - `--code-switch exclude`: **510,500** / 1104.3h
  - `--min-tier auto_gold --min-quality-tier B`: **55,360** / 151.6h
  `report.build` re-run: 594,010 entries, 10/11 acceptance criteria pass (unchanged
  pattern, `text_verified` still the only failure). `grep -c "/mnt/Drive1/"` = 0 across
  manifest.jsonl/train.jsonl/val.jsonl. `catalog verify` 17/17 PASS.
- **Test baseline refresh**: `tests/test_catalog.py::test_manifest_build_matches_
  expected_corpus_totals` had already drifted stale from T22's 2026-07-18 re-export
  (596,571 floor vs. that session's actual 596,089 — flagged but not fixed in T23's
  write-up) and this session's further legitimate drop widened the gap. Per the test's
  own documented update policy ("update this baseline only after an intentional,
  verified manifest.export re-run" — exactly what this entry is), refreshed all 6
  baseline constants to this session's real values (count=594010, speakers=8682,
  gold=149, auto_gold=274699, silver=152909, bronze=166253) with a dated comment
  explaining the drop. Full suite: 468 passed, 0 failed (was 467 passed / 1 failed
  before this fix).
- **Git**: all of this session's changes (T24 code + T23/T24 follow-up) committed
  together — see commit referenced at the top of this file's git log.

### T24. Upgrade `canto-hk-g2p` 1.5.0 → 1.9.0 + phonological validation — done 2026-07-19
- **What**: the pipeline's `.venv` had `canto_hk_g2p` pinned at 1.5.0 while the local
  source repo (`~/Documents/canto-g2p`, this pipeline's own upstream dependency) had
  already shipped four releases on top of that (v1.6.0 inventory/segment API, v1.7.0/
  v1.7.1 polyphone tie-break data fixes, v1.8.0 `user_dict` runtime override, v1.9.0
  `convert_candidates()`) — all already tagged/published on PyPI, confirmed via
  `gh release list` + PyPI JSON API. `pipeline/nodes/g2p.py`'s Jyutping validity check
  was also regex-shape-only (`^[a-z]+[1-6]$`), which accepts syllable-shaped garbage
  like `"zzz1"` that isn't a real Cantonese syllable — a real gap against Hard
  Constraint #8's intent, closeable now that v1.6.0 exposes the LSHK phonology inventory.
- **Reinstall**: `uv pip install -e ~/Documents/canto-g2p --force-reinstall --no-deps`
  (editable, source repo already at `df8c552`/v1.9.0) — `.venv` now reports
  `canto_hk_g2p.__version__ == "1.9.0"`.
- **Code change (`pipeline/nodes/g2p.py`)**: added `_is_valid_token()` — a token must
  pass both the existing regex AND `canto_hk_g2p.segment(token) is not None` (v1.6.0).
  `validate_jyutping()` now calls this instead of the bare regex. Module docstring
  updated with a dated note explaining the upgrade, what changed, and what was
  deliberately NOT adopted this round (see below). 6 new tests in
  `tests/test_g2p_node.py` (`_is_valid_token` accept/reject cases including the
  `"zzz1"`-passes-regex-but-not-phonology case, plus a `validate_jyutping`-level
  regression test) — 19/19 passing in that file; full suite 467 passed / 1 failed
  (the same pre-existing `test_manifest_build_matches_expected_corpus_totals` baseline
  staleness flagged in T23's write-up below, unrelated to this change).
- **Corpus-wide reprocess**: sampled 3,000 already-converted rows comparing old
  regex-only vs. new segment()-gated validation — **zero accept/reject flips** (real
  ASR-derived text essentially never produces phonologically-invalid-but-regex-shaped
  garbage), so the stricter check was safe to roll out corpus-wide with no expected
  manifest-eligibility impact. Separately, a 500-row before/after diff showed **287/500
  (57%) rows got a corrected `jyutping` string** from the v1.7.0/v1.7.1 tie-break data
  fix alone (e.g. `一本正經` `zing1`→`zing3`, `沉重` `zung6`→`cung5`), with 0
  valid_fraction regressions — confirmed the reprocess is a pure correctness win, not a
  gate-shifting risk. Reset `provenance = NULL` for all 780,215 `g2p_node`-tagged rows
  (same idempotent-reset pattern as T20/T22's `lang_label_checked`) and re-ran
  `pipe run g2p` in the background (`nohup`, `metadata/logs/g2p_reprocess_20260719.log`)
  — **768,663 converted, 768,634 accepted (99.996%), 29 rejected (valid_fraction <
  0.80), 0 errors, 225s (3,412/s), `run_id=g2p_e1d6091188a0`** (the ~11,552-row drop
  from 780,215 to 768,663 is expected: discovery is scoped to `filters.pass = TRUE`
  ids, same anti-join shape as every other run — a handful of ids lost `filters.pass`
  between the original g2p run and this reprocess, e.g. via T20/T22's later gate
  tightening, and correctly fell out of scope rather than being force-reprocessed
  anyway).
- **Deliberately NOT adopted this round** (scoped out, tracked as follow-ups, not
  spec-first-guessed without real data): v1.8.0's `Pipeline(user_dict=...)` override —
  no curated correction list exists yet; needs sourcing from `calibrate.sample` QA
  reject/flag patterns first (RTHK/YouTube/podcast presenter names, programme titles
  are the likely candidates, per the earlier advisory conversation). v1.9.0's
  `convert_candidates()` — a `pipe calibrate serve` UI feature (surface ambiguous
  polyphone alternatives to the human reviewer instead of blind-trusting rank-0), not a
  `g2p` node concern; needs its own design pass on the calibrate UI.
- **Upstream requests filed** (per the earlier advisory): 3 issues opened against
  `typangaa/canto-hk-g2p` — [#11](https://github.com/typangaa/canto-hk-g2p/issues/11)
  `convert_candidates_batch()` (throughput parity with the rest of the batch-capable
  API), [#12](https://github.com/typangaa/canto-hk-g2p/issues/12) a confidence/frequency
  weight per candidate (to distinguish a strong lean from a genuine tie when deciding
  what's worth a human QA look), [#13](https://github.com/typangaa/canto-hk-g2p/issues/13)
  exposing which dictionary layer resolved a token (word_dict/rime/ToJyutping/oral_hk/
  user_dict — would make building the `user_dict` override list evidence-based instead
  of guesswork).
- **Not done** (explicit follow-ups, not started this session):
  1. Build the actual `user_dict` override list from real QA-reject evidence once T1's
     review queue has enough samples flagging G2P mispronunciations specifically (as
     opposed to ASR-text errors) — needs a way to distinguish the two failure modes in
     `calibrate.sample`'s existing flag taxonomy, which doesn't exist yet either.
  2. ~~Wire `convert_candidates()` into `pipe calibrate serve`'s UI to surface polyphone
     alternatives during human review.~~ — **done 2026-07-19**, see T27 above.
  3. ~~`manifest.jsonl` still holds pre-reprocess jyutping strings~~ — **done
     2026-07-19**, see "T23+T24 follow-up" entry above (bundled with T23's re-export
     rather than exporting twice).

### T23. Wire `label.suite` into `pipeline/tools/chain_runner.py` — done 2026-07-18
- **What**: `label.suite` (writes `labels_lang`/`labels_overlap`/`labels_music` — the
  per-segment audio-based lang-id T20/T22's `filter.decide` gates depend on, plus the
  music/overlap signals T13's `quality_tier.assign` depends on) was last run manually
  2026-07-03 and was entirely absent from all 12 `chain_runner.py` rounds — not a one-off
  oversight, the round list simply never included it. Every other stage kept advancing
  via `pipe chain run` automation while `label.suite` silently fell behind: coverage
  stalled at 36.72% (455,930/1,241,610 segments) for 15 days, growing a 785,480-segment
  backlog. Found while investigating why a recurring NBN-advertisement text (embedded in
  18 SBS podcast episodes) was still passing `filter.decide` — 10/18 occurrences had
  `labels_lang` = NULL, so the T20/T22 audio-language elif branches never fired.
- **Immediate mitigation (2026-07-18)**: ran the 785,480-row backlog in the background
  (`nohup .venv/bin/python -m pipeline.cli run label.suite`, PID 452882, both GPUs,
  `metadata/logs/label_suite_backlog_20260718.log`) — finished cleanly, 785,480
  processed, 0 errors, 16,209s (48.5/s), `run_id=label_suite_2e8e5702fa46`. Coverage now
  100%. This alone was only the one-time catch-up; the structural fix is below.
  **Follow-up done 2026-07-19** (see "T23+T24 follow-up" entry above): `filter.decide`
  re-run + full manifest re-export. Turned out `lang_label_checked` did NOT need a
  manual reset — `DECIDE_DISCOVER_SQL` already self-heals on a newly-landed
  `labels_lang` row; the note originally here assumed it needed the same manual reset
  T20's rollout required, which was checked and found unnecessary this time (different
  scenario — see the follow-up entry for why).
- **Fix (differs slightly from the original plan)**: rather than inserting `label.suite`
  as a brand-new round (which would have renumbered rounds 5-12 to 6-13), it was merged
  directly into the existing round 5 as a run-many pair with `pregate.snr` — same DAG
  position, zero renumbering needed elsewhere. `pregate.snr` is CPU-only (CLI help says
  "CPU, pipeline-cut segments only", no `--devices` arg) and `label.suite` is GPU-only
  (`cuda:0,cuda:1`) — no device contention, unlike pairing two different GPU models on
  one device (2026-07-13 starvation finding). Both read only `segments`; writes are fully
  disjoint (`pregate` vs `labels_lang`/`labels_overlap`/`labels_music`) — same safety
  shape as the existing round 2 (`ingest.probe`+`lang_screen.auto`) and round 11 pairings.
  Round 5 now lands before round 10 (`filter.decide`, consumes `labels_lang`) and round
  12 (`quality_tier.assign`, consumes `labels_music`/`labels_overlap`), so both stay
  automatically current on every future `pipe chain run` pass — no more silent drift.
  ```
  1. ingest.commit
  2. ingest.probe + lang_screen.auto
  3. segment.diarize
  4. segment.vad_cut
  5. pregate.snr + label.suite        <- CHANGED: label.suite added, run-many
  6. asr.transcribe                    (unchanged)
  7. asr.agreement
  8. filter.text
  9. filter.acoustic
  10. filter.decide                    (now sees fresh labels_lang every pass)
  11. g2p + tier.assign + speaker.cluster
  12. quality_tier.assign              (now sees fresh labels_music/overlap every pass)
  ```
- **Edits made**:
  1. `pipeline/tools/chain_runner.py`: `build_rounds()` round 5 →
     `Round(5, "pregate.snr + label.suite", ["pregate.snr", "label.suite"],
     extra_args={"label.suite": device_args} if devices else {})`. Module docstring's
     round table, `--devices` CLI help text, and `--devices` argparse help string all
     updated to reflect `label.suite` now being a GPU round threaded via `--devices`.
     Round count unchanged at 12 (no renumbering needed).
  2. `tests/test_chain_runner.py`: `test_build_rounds_expected_run_many_pairs` and
     `test_build_rounds_solo_rounds_are_single_node` updated for round 5 now being
     run-many; `test_build_rounds_devices_threaded_only_to_gpu_rounds` extended to assert
     `label.suite` gets `--devices` while `pregate.snr` doesn't; 2 new tests
     (`test_run_chain_run_many_round_5_pairs_pregate_and_label_suite`,
     `test_run_chain_run_many_round_5_threads_devices_to_label_suite_only`) mirroring the
     existing round-2 pairing/device-threading test pattern. 20/20 passing in this file.
  3. No catalog/schema changes — `label.suite` was already `RUN_MANY_ADAPTERS`-registered
     (`conn=` injection already done) and already accepted `--devices`/`--gpu-policy`/
     `--batch`/`--mem-fraction`/`--limit` via its existing CLI parser.
- **Verified**: `pipe chain run --dry-run` shows the correct 12-round plan with round 5 as
  `pregate.snr + label.suite [run-many]`; `tests/test_chain_runner.py` 20/20 passing; full
  suite 462/463 passing (1 unrelated pre-existing failure, see below).
- **Unrelated pre-existing test failure found while verifying (not fixed, out of scope)**:
  `tests/test_catalog.py::test_manifest_build_matches_expected_corpus_totals` fails —
  live count is 596,089 but the test's `BASELINE_COUNT` floor is 596,571. This is T22's
  same-session re-export (`default 596,577→596,089`, see T22 above) never having been
  reflected back into this test's baseline constants — a pure test-staleness gap, not
  caused by this T23 change (T23 doesn't touch `filters`/manifest counts, only
  `chain_runner.py` round wiring). Flagging for a future baseline refresh; left as-is
  since it's outside T23's stated scope.
- **Not done**: the `lang_label_checked` reset + `filter.decide` re-run + manifest
  re-export that would make the 785,480-row catch-up's newly-covered `labels_lang` rows
  actually affect `filters.pass` — this is the same downstream step T20/T22 each did
  after their own production rollout, still pending as a separate follow-up.

### T20. Audio-based Mandarin gate wired into `filter.decide` — done 2026-07-18
- **What**: found while the owner was reviewing the T1 QA queue and asked why Mandarin
  segments were showing up despite "having a language filter." Both existing
  segment-level gates were weak: `lang_screen.auto` is raw-FILE-level and deliberately
  lets `mixed` files through; `filter.text`'s `mandarin_ratio()` is a TEXT heuristic over
  the ASR transcript that under-detects genuine spoken Mandarin transcribed into fluent
  standard written Chinese. `labels_lang` (mms-lid-126, computed by `label.suite` from
  AUDIO) was a much stronger signal but was never read by `filter.decide`/`tier.assign` at
  all — confirmed live: 48 segments already in the QA queue were `lang='cmn'` at 92-99%
  confidence yet had already cleared `filter.decide`.
- **Fix**: `filter.decide` now hard-fails `lang='cmn' AND cmn_prob >= 0.8`
  (`MANDARIN_AUDIO_PROB_MIN`) as `fail_reason='mandarin_audio'`, checked last (after
  text/acoustic gates). New `filters.lang_label_checked`/`mandarin_audio_prob` columns;
  `manifest.build`/`manifest.export` need no change (already `filters.pass = TRUE`-gated).
- **Result**: backfill against the live catalog — 455,894 rows re-decided in 18s,
  **10,940 segments flipped from pass to `mandarin_audio` fail** (~1.4% of the
  then-780,219-strong passing pool). `catalog verify` 17/17 PASS after. Gate only fires
  going forward for segments `label.suite` has reached (455,894/1,241,610 at run time). 44
  of the flipped rows were already sitting in the pending QA queue — left in place rather
  than pruned (still useful as a human sanity-check on the audio classifier).
- **Re-export 2026-07-18**: default manifest + all 6 derived cuts re-exported to actually
  drop the 10,940 rows from every on-disk file that reads `filters.pass`:
  - default: 606,775→**596,577** entries, 1349.3h→**1332.1h**
  - `--min-tier auto_gold`: **275,064** / 634.3h (was the canto-tts pretrain input)
  - `--min-tier silver`: **428,714** / 960.9h
  - `--min-tier bronze`: **596,577** / 1332.1h (same population as default — bronze is the
    manifest-eligibility floor)
  - `--code-switch only`: **84,167** / 225.4h
  - `--code-switch exclude`: **512,410** / 1106.8h
  - `--min-tier auto_gold --min-quality-tier B`: **55,365** / 151.6h (the canto-tts clean
    fine-tune input)
  `report.build` also re-run: 596,577 entries, 10/11 acceptance criteria (unchanged —
  `text_verified` still the only failure, expected, see T1).
- **Tests**: 9 new in `tests/test_filter_node.py`. Full detail: DECISIONS.md 2026-07-18.

### T21. Low-agreement-first QA sampling order (`calibrate.sample --order`) — done 2026-07-18
- **What**: companion finding from the same investigation — `calibrate.sample` always
  sampled uniformly at random within its scoped tier/min-agreement/code-switch
  population, so e.g. an `auto_gold`-scoped batch skewed to agreement~0.95-1.0 (where
  most of the tier's mass sits) with no way to deliberately pull the riskiest
  boundary-agreement segments into review. (Distinct from `next_pending()`'s pre-existing
  browsing `order` param, which only re-sorts items already queued.)
- **Fix**: added `order_by` (`'random'` default / `'agreement_asc'`) to `calibrate.sample`
  (`discover()`/`run_calibrate_sample()`, CLI `--order`, and a new "Sample:" panel
  dropdown in `pipe calibrate serve`), composable with tier/min-agreement/code-switch —
  e.g. `--tier bronze --code-switch only --order agreement_asc`.
- **Tests**: 5 new in `tests/test_calibrate_node.py`. Full detail: DECISIONS.md 2026-07-18.

### Near-incident: `calibration_review` pending-queue cleanup — caught + recovered, done 2026-07-18
- **What happened**: owner asked to empty the 3,392-row `pending` backlog (16 accumulated
  `calibrate.sample` batches), OK with those segments being re-sampled later, explicitly
  **not** wanting a hard delete. Explained that `discover()`'s anti-join is on row
  EXISTENCE, not `decision` value, so soft-marking rows `skipped` would block re-sampling
  exactly as permanently as leaving them `pending` — only an actual `DELETE` frees the id
  for re-sampling, and since all 3,392 rows were undecided (`pending`), deleting them loses
  zero review data by construction. Owner approved on that basis.
- **What went wrong**: `calibrate_server.py`'s offline-decision buffer
  (`metadata/calibration_pending_decisions.jsonl`, see DECISIONS.md 2026-07-13) meant 113
  unique ids from the day's active review session had a real verified/rejected/flagged
  decision sitting unflushed in that buffer while their `calibration_review` row still
  read `pending` — the bulk `DELETE FROM calibration_review WHERE decision='pending'`
  removed those 113 rows too. `record_decision()` (what `flush-pending` replays through) is
  a plain `UPDATE ... WHERE id = ?`; against a missing row it silently affects 0 rows, so a
  `flush-pending` run at that point would have discarded all 113 decisions with zero error
  output anywhere.
- **Caught pre-loss**: a fresh DB query (58 verified) didn't match the live server's
  `/api/stats` (149 verified) — the gap was `_stats_with_overlay()`'s in-memory buffer
  overlay. Cross-checked against the still-intact JSONL buffer *before* calling
  `flush-pending`, so nothing was actually lost.
- **Recovery**: reinserted 113 placeholder `pending` rows (id + `sample_batch` from the
  buffer + `original_best_text` reconstructed from `asr_agreement.best_text`, still
  accurate since the real flush had never run), then `pipe calibrate flush-pending`
  (113/113, 0 errors). Verified actual side effects, not just counts: 149 `verified` rows
  all have `text_verified=TRUE`+`tiers.tier='gold'`, 21 `rejected` all have
  `tiers.tier='excluded'` — exact pre-incident state reproduced.
- **Residual bug fixed in passing**: `calibrate_server.py`'s `_local_decisions` module
  global loads once at process start and never learns about an externally-run
  `flush-pending`, so the long-running `pipe calibrate serve` process kept double-counting
  the same 113 decisions in `/api/stats` (showing 240/42/2/284) even after the DB was
  correct. Not a data bug — fixed by restarting the server process so it reloads the
  (now-empty) buffer file. **Not yet fixed at the code level** — if this bites again,
  `_local_decisions` should be diffed/reconciled against the DB periodically, or the
  server should re-`load_pending_decisions()` after every `flush-pending`-capable write
  path, rather than relying on a manual restart. Low priority: only matters when
  `flush-pending` is run from a process other than the server itself while the server stays
  up, which is a maintenance-script pattern, not the normal owner workflow.
- **Final state**: `calibration_review` — 171 rows, 0 pending, 149 verified / 21 rejected /
  1 flagged, all downstream side effects correct. Full writeup: DECISIONS.md 2026-07-18.

### "English only" / "Other language" one-click reject buttons — done 2026-07-18
- **What**: owner reported some queued segments are English-only or another
  non-Cantonese/non-Mandarin language; asked for one-click buttons matching the existing
  Mandarin button (T19, 2026-07-15).
- **Done**: `ENGLISH_ONLY_FLAG_REASON`/`OTHER_LANGUAGE_FLAG_REASON` constants in
  `pipeline/nodes/calibrate.py`, same pattern as `MANDARIN_FLAG_REASON` exactly — both
  submit `decision='rejected'` (excludes via `tiers.tier='excluded'`, same as Mandarin —
  CLAUDE.md Hard Constraint #1 language-purity violations, not text-quality issues). Two
  new buttons in `pipe calibrate serve` ("English only (E)" / "Other language (O)"), same
  red styling, keyboard shortcuts `E`/`O`. No schema change — reused the existing
  `flag_reason` TEXT column and `top_flag_reasons` leaderboard (already scoped to
  `rejected`+`flagged` with a reason). 2 new tests in `tests/test_calibrate_node.py`
  (66/66 in that file). Live `calibrate serve` process restarted to pick up the new
  buttons — confirmed present in the served HTML.
- **Also fixed same session**: `tests/test_catalog.py`'s
  `test_manifest_build_matches_expected_corpus_totals` baseline was stale from before
  T20's `mandarin_audio` gate (the earlier T20 work in this same session dropped the
  manifest-eligible pool below the old floor-only `BASELINE_COUNT`, which the test's own
  docstring anticipates: "Update this baseline only after an intentional, verified
  manifest.export re-run" — exactly what T20's re-export was). Refreshed all 6 baselines
  to today's live-catalog numbers (count 606,775→596,571; speakers 9,023→8,715; gold
  58→149 from the recovered T1 review session; auto_gold/silver/bronze similarly down from
  the mandarin_audio exclusions). Full suite green after (455/455).

### T22. Audio-based English/other-language gate wired into `filter.decide` — done 2026-07-18
- **What**: same-day follow-up to the English-only/Other-language buttons above — checked
  whether an automated filter equivalent to T20's Mandarin gate existed for these two
  cases. It didn't: `english_ratio()` is TEXT-only and has no "other language" concept at
  all; `lang_screen.auto`'s low-Cantonese-ratio reject band incidentally catches
  non-Cantonese raw files as a side effect, but is file-level, not segment-level, and not
  an intentional English/other detector.
- **Impact check before building**: 634 segments have `labels_lang.lang NOT IN
  ('yue','cmn')` at `lang_prob >= 0.8`, 616 of them currently passing `filter.decide`
  (169 vie, 147 tha, 125 kor, 70 eng, 32 jpn, 28 mya, long noisy tail of exotic-language
  misclassifications on short clips). For the 70 audio-English segments, `english_ratio()`
  missed all 70 (avg 0.039) — ASR hallucinated fluent Chinese text over English audio,
  same blind spot as the pre-T20 Mandarin gap. Only 0.75h impact, 0 already in QA queue.
- **Fix**: `decide_row()` gains `english_audio`/`other_language_audio` fail reasons (split,
  not merged — matches the two new buttons, per owner decision), gated on
  `labels_lang.lang_prob >= NON_CANTONESE_AUDIO_PROB_MIN` (0.8, same as
  `MANDARIN_AUDIO_PROB_MIN`). New `filters.audio_lang_prob` audit column.
- **Production backfill subtlety**: T20's `lang_label_checked` versioning column was
  already `TRUE` for all 455,894 rows with a label, so a plain `filter.decide` re-run
  would NOT have retroactively applied the new gate — had to
  `UPDATE filters SET lang_label_checked = FALSE WHERE lang_label_checked = TRUE` first to
  force re-discovery (reused T20's existing mechanism, no new code needed).
- **Result**: 455,930 rows re-decided in 15s. New counts matched the pre-check prediction
  exactly: **70 english_audio, 546 other_language_audio** (616 total). `mandarin_audio`
  unchanged at 10,940. `catalog verify` 17/17 PASS. `manifest.build`/`export` need no code
  change (already `filters.pass`-gated).
- **Tests**: 6 new in `tests/test_filter_node.py`. 461/461 full suite green before the
  production run. Full detail: DECISIONS.md 2026-07-18.
- **Re-export (owner requested same session)**: default + all 6 derived cuts re-exported
  — default 596,577→**596,089** entries, 1332.1h→**1331.6h**; auto_gold 275,030/634.3h;
  silver 428,434/960.6h; bronze 596,089/1331.6h; codeswitch_only 84,044/225.2h;
  codeswitch_exclude 512,045/1106.4h; auto_gold+qualityB 55,360/151.6h. `report.build`
  re-run (10/11 criteria, unchanged pattern). `catalog verify` 17/17 PASS. Full numbers:
  DECISIONS.md 2026-07-18.

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

### T15. Drain the reingest.pending backlog through the full DAG (found 2026-07-12) — done 2026-07-17
- **What**: `docs/archive/IO_OPTIMIZATION_PLAN.md` diagnosed Drive4's file-count skew (2.55M
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
      as `docs/archive/PIPELINE_REVIEW_2026-07-13.md` Issue #19 / its Phase B2.
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

