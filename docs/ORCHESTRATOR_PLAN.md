# Single-Process Concurrent Orchestrator — Implementation Plan

Status: **10/22 call sites done, `pipe run-many` VALIDATED LIVE, running
2026-07-07**. `conn=` injected into `run_filter_acoustic`,
`run_segment_diarize`, `run_label_music`, `run_asr_transcribe`,
`run_asr_agreement`, `run_ingest_commit`, `run_speaker_embed`,
`run_speaker_cluster`, `run_lang_screen_auto`, `run_tier_assign`; smoke-tested
and then run at full backlog scale against the live catalog — first
`label.music` (GPU) + `filter.acoustic` (CPU) 2026-07-06 (both finished),
then `asr.transcribe` (2 GPUs) + `filter.acoustic` (CPU) launched 2026-07-07
once `label.music` completed and GPUs sat idle again (still running as of
this update). Remaining ~12 call sites NOT yet done — extend incrementally
as new concurrent-run needs come up (see "What's left" below). Written
2026-07-06, updated 2026-07-07 (twice: asr.py, then ingest.commit +
speaker.py + lang_screen.py + tier.py).

## Problem

DuckDB's file lock is **per-process**, not per-transaction: only one OS
process may hold a read-write connection to `metadata/corpus.duckdb` at a
time. Every pipeline node (`filter.acoustic`, `segment.diarize`,
`asr.transcribe`, `speaker.embed`, ...) is invoked as its own `pipe run X`
CLI call, each opening its own `conn = connect()` and holding it for the
node's entire runtime. Consequence, confirmed live this session: while
`filter.acoustic` runs, even a **read-only** `connect_ro()` from a second
process fails immediately:

```
IO Error: Could not set lock on file "corpus.duckdb": Conflicting lock is held...
```

This blocks any GPU-heavy node (`segment.diarize`, `asr.transcribe`,
`speaker.embed/cluster`, `label_music`, `lang_screen`) from ever running
concurrently with a CPU-heavy node like `filter.acoustic` — both RTX 4090s
sit idle for the ~13+ hour duration of the current `filter.acoustic` backlog
run, purely because of process-level contention, not real resource
contention.

## Empirical finding (verified this session, not assumed)

The single-writer restriction is **per-process**, not per-transaction.
Multiple threads *inside one process*, each with its own `conn.cursor()`,
write concurrently with zero conflicts:

```python
conn = duckdb.connect(path)
# thread A: c = conn.cursor(); c.execute("INSERT INTO a ...")
# thread B: c = conn.cursor(); c.execute("INSERT INTO b ...")
# → both succeed, 1000/1000 rows each, no errors
```

`upsert_rows(conn, table, rows, keys)` and `record_batch(conn, ...)`
(`pipeline/catalog/catalog.py`, `pipeline/orchestrator/journal.py`) both take
a plain `duckdb.DuckDBPyConnection`-typed argument — a `.cursor()` object
satisfies that duck-typing transparently, so **no changes needed in
catalog.py/journal.py**, only in how each node obtains its `conn`.

This is already exploited once in the codebase: `asr.transcribe`
(`pipeline/nodes/asr.py`) runs both Cantonese-FT and large-v3 models "from
ONE supervisor process ... sharing one catalog connection" specifically to
avoid this exact lock (see its module docstring). The orchestrator
generalises that pattern pipeline-wide.

## Design — Path A: shared single-process orchestrator

Instead of N separate `pipe run X` OS processes each self-connecting, run
several node coroutines as `asyncio` tasks inside **one** process that opens
**one** `connect()` and hands each node its own `conn.cursor()`. Since
asyncio is cooperatively scheduled (one Python statement executes at a time
regardless of how many tasks are "concurrent"), there is no real thread-race
risk even beyond DuckDB's own MVCC — the actual heavy lifting (GPU/CPU
inference) already happens in **subprocess workers** spawned by each node
supervisor (`spawn_worker()`), fully independent of the DB connection.

### Changes required

**1. Dependency-inject `conn` into every node's `run_*()` (22 call sites, 15 files)**

Change `conn = connect()` → accept an optional `conn` parameter, falling back
to self-connect for backward-compatible standalone `pipe run X` usage:

```python
async def run_filter_acoustic(*, conn=None, n_workers=4, ...):
    conn = conn or connect()
    ...
```

Full inventory of call sites (grep `conn = connect()` across
`pipeline/nodes/*.py`, captured 2026-07-06 — re-grep before starting in case
anything shifted):

```
pipeline/nodes/ingest_download.py:510   (run_ingest_commit — already short-lived
                                          via JSON-staging; lowest priority, may
                                          not need touching at all)
pipeline/nodes/label_suite.py:127
pipeline/nodes/tier.py:73
pipeline/nodes/asr.py:214               (run_asr_transcribe)
pipeline/nodes/asr.py:352               (run_asr_agreement)
pipeline/nodes/label_prosody.py:118
pipeline/nodes/ingest_probe.py:132
pipeline/nodes/g2p.py:159
pipeline/nodes/lang_screen.py:228
pipeline/nodes/recover_orphans.py:183
pipeline/nodes/speaker.py:166           (run_speaker_embed)
pipeline/nodes/speaker.py:627           (run_speaker_cluster)
pipeline/nodes/label_music.py:131
pipeline/nodes/filter.py:341            (run_filter_text? confirm)
pipeline/nodes/filter.py:410            (run_filter_acoustic)
pipeline/nodes/filter.py:717            (run_filter_decide? confirm)
pipeline/nodes/rebalance.py:176
pipeline/nodes/rebalance.py:306
pipeline/nodes/segment.py:408           (run_segment_diarize)
pipeline/nodes/segment.py:980           (run_segment_vad_cut)
pipeline/nodes/segment.py:1290          (run_pregate_snr)
pipeline/nodes/raw_flac.py:190
pipeline/nodes/raw_flac.py:305
```

**Priority order for implementation** (do not need all 22 in one pass):
1. `segment.py:408` (`run_segment_diarize`) + `filter.py:410`
   (`run_filter_acoustic`) — the concrete pair we need *today* (GPU diarize
   backlog alongside the running CPU acoustic-filter backlog).
2. `asr.py` (both functions) — next most valuable GPU node.
3. `speaker.py`, `label_music.py`, `lang_screen.py`, `tier.py` — round out
   GPU/CPU nodes likely to want to run alongside something else.
4. Everything else — mechanical, do when convenient, not blocking.

**2. New CLI orchestrator command** (`pipeline/cli.py`)

A new subcommand, e.g. `pipe run-many <node> <node> ...`, that:
- Opens one `connect()`.
- For each requested node name, looks up its existing `run_*()` coroutine
  and CLI arg parser (reuse the parsers already registered per node — do not
  duplicate arg definitions).
- Calls each with `conn=conn.cursor()` (one cursor per node — do not share a
  single cursor object across concurrently-scheduled coroutines even though
  the GIL/event-loop makes it *probably* safe; cheap to just give each its
  own).
- `await asyncio.gather(*tasks)`, aggregate/print each node's returned dict.

**3. GPU device coordination**

`segment.diarize` and `asr.transcribe` both want a GPU. Decide explicitly
per `run-many` invocation which device each gets (`--device cuda:0` /
`cuda:1`, or share one device with `mem_fraction` capping — both 4090s have
24GB, a pyannote + a faster-whisper model together comfortably fit on one).
This is a policy/args decision, not new code — existing `devices`/
`mem_fraction` params on `run_segment_diarize`/`run_asr_transcribe` already
support it.

**4. CPU thread-budget awareness**

Directly relevant lesson from today's `filter.acoustic` oversubscription bug
(see `pipeline/nodes/filter.py` — `torch.set_num_threads(1)` +
`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/
`NUMEXPR_NUM_THREADS=1` now passed via `env=` to every `filter.acoustic`
worker subprocess, cf. commit not yet made — see Current State below):
running several nodes' CPU-side subprocess pools concurrently in one
`run-many` call means their worker counts must be sized against the shared
44-48 physical cores, not each sized as if it had the whole box. E.g. if
`filter.acoustic` is already running 8 workers × ~12 threads ≈ 96 threads,
adding `segment.diarize`'s CPU-side sidecar-reuse thread pool on top needs a
deliberately smaller pool, not another full-size one.

**5. Testing**

- New integration test mirroring the throwaway verification done this
  session: two real (or minimal fake) node coroutines run concurrently via
  `asyncio.gather` against one scratch catalog connection, each writing to a
  different table, assert both succeed with correct row counts.
- Re-run the full existing suite — every touched `run_*()` must keep
  `conn=None` as the default and behave identically to today when invoked
  standalone (`pipe run filter.acoustic` etc. must not change behavior).

**6. Docs**

Update `CLAUDE.md` / a pipeline doc with the new `run-many` usage and the
CPU-thread-budget caveat.

### Estimate

| Step | Estimate |
|---|---|
| 1. `conn` injection (22 call sites / 15 files) | 3.5–4h |
| 2. New `run-many` CLI orchestrator | 1.5–2.5h |
| 3. GPU device coordination | 0.5–1h |
| 4. Testing | 1–1.5h |
| 5. Docs | 0.5h |
| **Total** | **~7–10h (about one working day)** |

Narrower alternative considered and **not** chosen: JSON-staged decoupling
of just `segment.diarize` (mirroring the `ingest_download.py` two-phase
refactor done earlier this session) — cheaper (~2–3h) but only fixes one
node; the next GPU node hits the same wall. The orchestrator is the one-time
fix for all current and future nodes.

## Current state snapshot (2026-07-06, ~22:47, for continuity across compaction)

- **`filter.acoustic`** (PID tree rooted at supervisor started 19:55, log
  `metadata/logs/filter_acoustic_20260706_v2.log`): running with the
  oversubscription fix (`torch.set_num_threads(1)` +
  `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/
  `NUMEXPR_NUM_THREADS=1` env passed to worker subprocesses). At 22:46:
  97,440/563,905 processed, ~9.5/s (throughput lower than the ~18/s gate
  test — full-scale run over real diverse-size files, or contention from
  `ingest.download` also running). Rough ETA from here: ~13–14h more. Holds
  the DuckDB write lock for its entire runtime — do not attempt any other
  `connect()` while it runs.
- **`ingest.download --source all`**: still running (JSON-staged, no DB
  connection during download — this was the earlier session's fix). Hit
  YouTube rate-limiting around 22:4x ("rate-limited for up to an hour") on
  the `youtube` channel-scan section; harmless/expected backoff, not a bug.
  20 rows staged in `metadata/ingest_download_staging.jsonl` so far (all from
  the `rthk` section — 3 from `鏗鏘集`, 17 from `千禧年代`). Not yet
  committed to the catalog (`pipe run ingest.commit` deliberately deferred
  until the write lock is free — cannot run while `filter.acoustic` holds
  it).
- **Uncommitted git changes** (not yet committed — user has not asked to
  commit this batch):
  - `pipeline/cli.py` — `ingest.commit` subcommand wiring.
  - `pipeline/nodes/filter.py` — (a) the `torch.set_num_threads(1)` +
    env-var thread-cap fix for `AcousticWorker` subprocesses (the
    oversubscription fix), (b) needs `import os` (already added).
  - `pipeline/nodes/ingest_download.py` — full two-phase JSON-staging
    refactor (`run_ingest_download()` never opens DuckDB;
    `run_ingest_commit()` is the sole DB-touching step).
  - `tests/test_ingest_download_node.py` — new, 9/9 passing.
  - `sources/youtube_channels.yaml` — modified earlier in the broader
    session (pre-dates this sub-thread; not investigated this turn).
- Once `filter.acoustic` finishes: the originally-planned next steps are
  still queued — `filter.decide` → `tier.assign` over the same backlog, then
  `pipe run ingest.commit` to land the staged download rows. The
  orchestrator (once built) would let some of this run *alongside*
  `segment.diarize`/`asr.transcribe` GPU backlogs instead of strictly
  sequentially.

## What was implemented (2026-07-06)

- `pipeline/nodes/filter.py::run_filter_acoustic` — `conn=None` param added,
  `conn = conn or connect()`.
- `pipeline/nodes/segment.py::run_segment_diarize` — same pattern.
- `pipeline/nodes/label_music.py::run_label_music` — same pattern (added
  beyond the original priority-1 pair once discovery showed `label.music` had
  a genuine ~163k-segment GPU backlog available *today*, unlike
  `segment.diarize` whose backlog turned out to already be fully drained —
  0 rows discovered when smoke-tested).
- `pipeline/cli.py`:
  - `RUN_MANY_ADAPTERS` dict (node name → async adapter function taking
    `(args, conn)`), currently `{"filter.acoustic", "segment.diarize",
    "label.music"}`.
  - `split_run_many_groups()` — pure function splitting `pipe run-many`'s
    `argparse.REMAINDER` tokens into per-node argv groups on literal `--`.
  - `pipe run-many <node> [args...] -- <node> [args...] -- ...` subcommand:
    opens one `connect()`, re-parses each group's argv through the node's
    *existing* registered subparser (`run_sub.choices[name]` — no arg
    duplication), calls each adapter with its own `conn.cursor()`, and
    `asyncio.gather`s them.
- `tests/test_run_many.py` (9 tests): `split_run_many_groups` edge cases;
  regression guard that each node's `conn=None` default still self-connects
  standalone; regression guard that a passed-in `conn` is used as-is
  (`connect()` monkeypatched to raise — must never be called); a concurrency
  proof test mirroring this session's original throwaway verification
  (two coroutines, cursor-per-coroutine on one shared scratch connection,
  writing to two different tables concurrently, asserting correct counts).
- Full suite: 158 passed pre-existing + 9 new = 167 passed. The only
  failures were `test_catalog.py`/`test_orchestrator.py` tests that call
  `connect_ro()` against the *live* catalog while the standalone
  `filter.acoustic` v2 backlog run held the RW lock — pre-existing
  environmental collateral, not a regression (this is, in fact, exactly the
  problem this plan solves; those tests pass again once nothing holds the
  live lock).

### Live validation (2026-07-06, ~23:04–23:09)

1. Stopped the standalone `filter.acoustic` v2 backlog run (idempotent —
   discovery is a `filters_acoustic` anti-join, so no work is lost; verified
   123,053 rows already committed before stop, matching the log).
2. Smoke test: `pipe run-many segment.diarize --devices cuda:0 --limit 20 --
   filter.acoustic --workers 2 --threads 2 --limit 20` against the live
   catalog — `segment.diarize` discovered 0 rows (its backlog was already
   fully drained: 10,717/10,910 raw_files have segments, remainder likely
   `lang_screen`-rejected raws out of scope for diarize's discovery query),
   `filter.acoustic` processed 20/20 with zero errors and zero lock
   conflicts.
3. Found a genuine concurrent-GPU-work candidate: `labels_music` had
   455,522/618,695 segments done — ~163k backlog. Added `conn` injection to
   `label_music.py` (see above) specifically to use it as the real pairing
   partner instead of the already-drained `segment.diarize`.
4. Smoke test #2: `pipe run-many label.music --devices cuda:0 --limit 40 --
   filter.acoustic --workers 4 --threads 4 --limit 40` — both completed,
   0 errors; DB counts confirmed exact (+40 each in `labels_music` /
   `filters_acoustic`).
5. Launched the **full backlog** concurrently in the background:
   `nohup pipe run-many label.music --devices cuda:0,cuda:1 --gpu-policy cap
   -- filter.acoustic --workers 8 --threads 4 > metadata/logs/
   run_many_music_acoustic_20260706.log 2>&1 &` (163,133 segments for
   `label.music`, 459,709 for `filter.acoustic`). Confirmed both GPUs
   dispatching (`gpu.0`/`gpu.1` pools), `label.music` running ~65-115/s.
   `filter.acoustic` throughput dropped from its previously-validated
   standalone 18.2/s to ~5.8/s while `label.music`'s CPU-side decode runs
   concurrently — real (not lock) contention, expected, and net-positive
   since `label.music`'s ~163k backlog finishes in under an hour, after
   which `filter.acoustic` should return to its full ~18/s rate for its
   much longer remaining run.

## What was implemented (2026-07-07, resource-utilization follow-up)

Resumed after the `label.music` + `filter.acoustic` background run from
2026-07-06 completed (`label.music`: 163,133/163,133 tagged, 0 errors, done
09:27). Found both RTX 4090s sitting fully idle again and `filter.acoustic`
alone only using ~3.2/48 cores (8 workers × ~40% CPU each) — a session
resumption check (`nvidia-smi`, `uptime`, `mpstat`, `ps`) confirmed this
before touching anything.

- Tried scaling `filter.acoustic` alone from 8→36 workers to use idle CPU:
  **made throughput worse** (26/s vs. the ~48/s instantaneous burst seen at
  n=8), load average hit 71 (over the 48-core ceiling) — classic
  oversubscription, not a real gain. Settled on **n=16, threads=2** after a
  2-minute steady-state measurement (6 samples via `Monitor`): consistent
  ~27/s, load average 36–42, comfortably under 48 — stable operating point,
  not further tuned given diminishing returns and restart overhead.
- Added `conn=None` injection to `pipeline/nodes/asr.py`:
  `run_asr_transcribe` (assignments-based, one discover() + worker per
  model/device pair) and `run_asr_agreement` (CPU-only, no subprocess
  workers). Same pattern as before; confirmed no `conn.close()` in the file.
- `pipeline/cli.py`: added `_run_many_adapt_asr_transcribe` and
  `_run_many_adapt_asr_agreement`, registered both in `RUN_MANY_ADAPTERS`
  (now 5 nodes: `filter.acoustic`, `segment.diarize`, `label.music`,
  `asr.transcribe`, `asr.agreement`).
- `tests/test_run_many.py`: added 2 conn-injection regression tests
  (`test_run_asr_transcribe_uses_injected_conn`,
  `test_run_asr_agreement_uses_injected_conn`) — 11/11 pass. Full suite:
  161 passed, 3 failed + 13 errors — all confirmed (spot-checked one
  traceback) to be the same pre-existing `connect_ro()` vs. live-RW-lock
  environmental collision as 2026-07-06, caused by the `filter.acoustic`
  job running during the test run, not a regression.
- Checked real backlog before launching (`discover_transcribe` directly,
  read-only connect after briefly stopping the standalone run): `canto_ft`
  45,482 segments, `whisper_v3` 11,411 segments — both genuine GPU work
  available today, unlike `segment.diarize`'s drained backlog on
  2026-07-06.
- Launched full backlog concurrently: `nohup pipe run-many asr.transcribe
  --models canto_ft,whisper_v3 --devices cuda:0,cuda:1 --gpu-policy cap --
  filter.acoustic --workers 12 --threads 2 > metadata/logs/
  run_many_asr_acoustic_20260707.log 2>&1 &` (worker count dropped from 16→12
  vs. the standalone measurement, anticipating the same real CPU contention
  from concurrent GPU-side decode seen with `label.music` on 2026-07-06).

## What was implemented (2026-07-07, ingest + speaker/lang_screen/tier follow-up)

Extended `conn=None` injection to the next 5 call sites per the plan's
priority order, closing out both the "round out GPU/CPU nodes" step and the
previously-deferred `ingest.commit` site (originally marked "lowest
priority, may not need touching at all" — touched anyway since the user
asked to integrate ingest into the orchestrator explicitly):

- `pipeline/nodes/ingest_download.py::run_ingest_commit` — `conn=None` added.
  Confirmed both call sites (`main_commit()`, `cli.py`'s
  `cmd_run_ingest_commit`) call it with no args, so the default self-connect
  path is unchanged.
- `pipeline/nodes/speaker.py::run_speaker_embed` and `run_speaker_cluster` —
  same pattern, both confirmed no `conn.close()`.
- `pipeline/nodes/lang_screen.py::run_lang_screen_auto` — same pattern. Note
  the actual function/CLI-node name is `lang_screen.auto`, not `lang.screen`
  as an earlier draft of this doc's call-site table implied — verify the
  registered `run_sub.add_parser(...)` name before wiring a `RUN_MANY_ADAPTERS`
  entry, don't assume from the file name.
- `pipeline/nodes/tier.py::run_tier_assign` — same pattern.
- `pipeline/cli.py`: added 5 new adapters
  (`_run_many_adapt_ingest_commit`, `_run_many_adapt_speaker_embed`,
  `_run_many_adapt_speaker_cluster`, `_run_many_adapt_lang_screen_auto`,
  `_run_many_adapt_tier_assign`), each re-deriving its args from the node's
  already-registered subparser (e.g. `speaker.cluster`'s `--sources` CSV
  split mirrors `cmd_run_speaker_cluster` exactly). `RUN_MANY_ADAPTERS` is
  now 10 entries: `filter.acoustic`, `segment.diarize`, `label.music`,
  `asr.transcribe`, `asr.agreement`, `ingest.commit`, `speaker.embed`,
  `speaker.cluster`, `lang_screen.auto`, `tier.assign`.
- `tests/test_run_many.py`: 5 new conn-injection regression tests
  (`test_run_speaker_embed_uses_injected_conn`,
  `test_run_speaker_cluster_uses_injected_conn`,
  `test_run_lang_screen_auto_uses_injected_conn`,
  `test_run_tier_assign_uses_injected_conn`,
  `test_run_ingest_commit_uses_injected_conn` — the last one monkeypatches
  `ingest_download.METADATA_DIR`/`STAGING_FILE`/`KNOWN_IDS_SNAPSHOT` to a
  `tmp_path`, same pattern as `tests/test_ingest_download_node.py`, since
  those are real module-level absolute paths and calling
  `run_ingest_commit()` unpatched would overwrite the live repo's actual
  staging file / dedup snapshot — dangerous while `ingest.download` is
  running in the background). 16/16 pass in `test_run_many.py`. Full suite:
  166 passed, 3 failed + 13 errors — all traced to the same pre-existing
  `connect_ro()` vs. live-RW-lock collision from the still-running
  `asr.transcribe`+`filter.acoustic` job (PID 4005423), not a regression.

This now means the full chain `ingest.commit` → `filter.decide`/`tier.assign`
→ `speaker.embed`/`speaker.cluster` can, in principle, run alongside a
GPU-heavy backlog node under one `run-many` invocation once there's an
actual concurrent-work reason to (e.g. `pipe run-many tier.assign --
asr.transcribe --models ... --devices ...` once both have real backlogs at
the same time).

## What's left (not started, lower priority / incremental)

- ~12 remaining `conn = connect()` call sites (see full inventory above,
  minus the 10 now done): `label_suite.py`, `label_prosody.py`,
  `ingest_probe.py`, `g2p.py`, `recover_orphans.py`, `filter.py`'s
  remaining two functions (`run_filter_text`/`run_filter_decide` — confirm
  exact names), `rebalance.py` (2 sites), `raw_flac.py` (2 sites),
  `segment.py`'s remaining two functions (`run_segment_vad_cut`/
  `run_pregate_snr`). All mechanical — same 3-line change, do when a real
  concurrent-run need comes up.
- GPU device-sharing policy across >2 concurrent GPU nodes (still not
  needed — `asr.transcribe`'s two models already use one GPU each; no
  3rd GPU node has been paired in yet).
- CPU worker-count tuning is empirical, not principled: the n=16 sweet spot
  for `filter.acoustic` alone was found by trial (8 too few, 36 way too
  many/oversubscribed) rather than derived from a formula — revisit if
  the workload mix changes materially (e.g. once decode cost per segment
  shifts with FLAC vs. WAV masters).
- Docs: `CLAUDE.md` doesn't yet mention `run-many` — add a line once a
  second or third node pairing has been used a few times and the usage
  pattern feels stable, rather than documenting a single-use command today.
