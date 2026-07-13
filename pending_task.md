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
  predates the whisper_v3 retirement backfill and the tier-tightening backfill. **T7 (its
  blocker) is now done** — this task is unblocked but not yet run.
- **How**: `pipe run manifest.export`.
- **Effort**: one command.
- **Depends on**: T7 (done). Trigger a fresh T4 report afterward. Consider waiting until
  T15's drain also lands so this doesn't go stale again within days.

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

### T10. Content-hash linkage (§5 external best-practice gap)
- **What**: audio↔metadata linkage currently relies on absolute paths + `shard_index()`
  discipline; another drive migration would require a full P5-C-style rebalance again.
- **How**: add a content-hash column on `segments`; compute on new writes, backfill the
  rest slowly (io-bound, can run in background).
- **Effort**: small design, long backlog fill (618k+ files); low priority.

### T15. Drain the reingest.pending backlog through the full DAG (found 2026-07-12)
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

### T16. Rebuild the `auto_gold` gate for the 2-model era (found 2026-07-13, canto_ft retirement consequence)
- **What**: `tier.assign`'s `auto_gold` gate requires `canto_ft_confidence > 0.8` — always
  `NULL` for new segments since canto_ft's retirement, so the gate fails closed and new
  segments cap at silver/bronze. Additionally, `char_agreement()` compares RAW texts with
  no punctuation/number normalization — systematically deflates qwen3_asr (AR, infers
  punctuation) vs sense_voice (CTC, weak punctuation) agreement, which is now the ONLY
  trust signal. See `docs/PIPELINE_REVIEW_2026-07-13.md` Issues #17 + #20 and its §5
  targeted external research (agy-gemini, 2026-07-13).
- **How** (order matters — the normalization fix must precede the distribution analysis,
  or the analysis itself is biased):
  1. Add text normalization (strip punctuation, normalize digits) inside the agreement
     computation — compute overlap on the normalized strings, store original texts
     unchanged. Update tests.
  2. Full-corpus `asr_agreement` backfill excluding canto_ft (mirror the 2026-07-10
     whisper_v3 backfill).
  3. FINDINGS-doc-style distribution analysis (agreement histogram × existing gold/QA
     ground truth from T1's reviews).
  4. Owner decides the new bar. Research-suggested starting point: normalized 2-model
     agreement ≥ 0.90–0.93 **AND** a third non-ASR signal — cheapest is
     `filters_acoustic.dnsmos >= 3.5` (already in the catalog, zero new compute).
     Fallback/second-stage candidates if precision is insufficient: SenseVoice
     emotion/event-tag proxy (already stored in `asr_results.metadata` — can be analyzed
     offline NOW without re-running ASR), CTC posterior entropy, forced-alignment
     tie-breaker (see review doc §5 Q2/Q3).
  5. Re-derive `tiers` for affected rows (pure SQL CASE backfill like 2026-07-11's
     threshold change — remember the T5 gap means tier.assign will NOT re-evaluate
     existing rows by itself).
- **Effort**: medium (normalization + tests ~1h; backfill is one long DB pass; analysis
  ~1h; the tier re-derivation is quick).
- **Depends on**: T15's ASR drain finished (DuckDB writer free + agreement rows complete);
  T1's pilot QA ground truth for step 4's validation.
- **Owner decision required**: the final threshold (step 4) — do NOT guess a number.

### T14. Full CPU+GPU utilization during chained node runs (found 2026-07-12)
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
Do now (independent):      T1 (human QA) + git commit backlog (review doc 07-13 Phase B)
Once T15's ASR drains:     T15 remaining chain (agreement → filter → g2p → tier →
                           speaker.cluster solo) → T6 (re-export + report.build)
Right after that:          T16 (auto_gold gate rebuild — normalization fix FIRST,
                           then backfill, then owner threshold decision)
Before next new ASR model: T5 (re-eval mechanism)
Cleanup (owner approves):  docs/PIPELINE_REVIEW_2026-07-13.md §3 Phase A (~13GB dead
                           model weights) + Phase D items
Schedule into P6 proper:   T9 (depends on T1), T10, T12, T14 levers (3)+(4)
Pulled by training needs:  T13
```

---

## Done

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
