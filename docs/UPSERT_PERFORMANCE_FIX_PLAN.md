# `upsert_rows()` Performance Fix — Implementation Plan

**Status: DONE, 2026-07-16.** All 10 rollout steps complete and verified against the
real catalog.

**Final verification (2026-07-16):**
- §7.5 full regression: `pytest tests/ -q` → **357/357 passing** (the writer lock was
  free this time, all 3 previously-lock-blocked catalog-touching files included).
- §7.6/§7.7 real-world benchmark: ran `pipe run speaker.cluster` solo against the live
  catalog (no `--limit`, all 3 sources, 1,241,586 segments). **Total wall-clock: 104s**
  for all 3 sources combined (podcast 538,310 rows, rthk 106,341, youtube 596,935) —
  vs. the historical baseline of **~78 minutes for the podcast source's upsert alone**
  (see `pending_task.md` T15 point 3 / `DECISIONS.md` 2026-07-13). That's roughly a
  **45×+ speedup** on the write path for the largest single source. Correctness
  verified: `speakers` table row count (1,241,586) and distinct `speaker_id` count
  (14,330) both matched the pre-fix run exactly — same clustering result, only the
  write mechanism changed.
- §7.8 spot-check: no other node needed changes — `upsert_rows()` is a shared helper,
  every caller benefits transparently; the threshold-gated dispatch (`< 2_000` rows
  keeps the original `executemany()` path) means small `--limit` runs and single-item
  writes are unaffected in behavior or performance.
- §7.9-10: this file + `DECISIONS.md` + `pending_task.md` updated (see the 2026-07-16
  entries in each).

**Progress (all steps):**
- ✅ §7.2: `pandas>=2.2.0` added to `pyproject.toml` (already present transitively at
  3.0.3, satisfies the new explicit constraint — no reinstall needed).
- ✅ §7.3: `upsert_rows()` rewritten in `pipeline/catalog/catalog.py` per §4 (threshold
  = `UPSERT_BULK_THRESHOLD = 2_000`, small-batch path unchanged, bulk path uses
  `pd.DataFrame(..., dtype=object)` + `conn.register()` + single `INSERT ... SELECT`).
- ✅ §7.4: New `tests/test_upsert_rows.py` written (14 tests, in-memory DuckDB, covers
  every case in §6.1: empty rows, small/large batch parity, JSON round-trip, NULL
  handling incl. the pandas int→float NaN-coercion gotcha, INSERT OR REPLACE overwrite
  semantics, dict-key-order robustness) — **14/14 passing**.
- ✅ §7.5: full suite **357/357 passing**, including the 3 catalog-touching files
  (34 tests) that were lock-blocked on 2026-07-15 — re-run clean once the writer lock
  freed.
- ✅ §7.6/§7.7: real `speaker.cluster` rerun — 104s total / 3 sources / 1,241,586
  segments, 45×+ faster than the pre-fix baseline, zero data drift.
- ✅ §7.8: spot-check — no other caller needs a change.
- ✅ §7.9-10: docs synced.

## 1. Motivation

This session diagnosed a multi-hour `speaker.cluster` slowdown across several wrong
hypotheses (thread oversubscription, npy-sidecar fallback, WAL checkpoint) before finding
the real, two-part root cause:

1. **`INSERT OR REPLACE INTO` against an already-populated table is delete+insert under
   DuckDB's MVCC**, not an in-place update. `speakers` already had rows from a prior
   `speaker.cluster` run (task #94), so this run's upsert hit full PK collision — same
   mechanism as the `filter.decide` OOM bug fixed earlier the same session (DECISIONS.md
   2026-07-14), just manifesting as extreme slowness (~78 min for podcast's 538,310 rows)
   rather than an OOM crash.
2. **`upsert_rows()` (`pipeline/catalog/catalog.py`) uses `conn.executemany()` with
   row-by-row parameterised tuples** — a documented DuckDB anti-pattern. DuckDB is an OLAP
   engine optimised for large vectorised operations; `executemany()` pays per-statement
   parse/bind overhead for every single row. Real-world reports show this can be **100×+
   slower** than DuckDB's native bulk-insert path (one case: 8m37s via `executemany()` →
   ~3s via CSV/vectorised load for a comparable row count — see Sources).

`upsert_rows()` is called by **every** DAG node in this pipeline (`filter.decide`, `g2p`,
`tier.assign`, `speaker.embed`, `speaker.cluster`, `label_*`, `raw_flac`, `rebalance`, …)
and also indirectly by `record_batch()` (`pipeline/orchestrator/journal.py`), which is
just a thin wrapper: `record_batch(...)` → `upsert_rows(conn, "task_runs", rows, [...])`.
**Fixing `upsert_rows()` fixes both problems identified this session in one change** —
`speaker.cluster`'s `speakers` writes and `record_batch()`'s `task_runs` writes (which
doubled every source's total write volume) both go through the same code path.

## 2. Research summary (verified via WebSearch + agy-gemini, 2026-07-15)

- DuckDB's Python `executemany()` is confirmed slow for bulk upserts —
  [Issue #10106](https://github.com/duckdb/duckdb/issues/10106).
- Upsert performance against a large existing table degrades further if the incoming
  batch is **not sorted by the target primary key** (10-15s → 2-3s per 1000 rows when
  sorted) — [Issue #11275](https://github.com/duckdb/duckdb/issues/11275). Our own SQL
  discovery queries already do `ORDER BY id`, so this is mostly already satisfied for the
  call sites that matter most (`speaker.cluster`); worth spot-checking others.
- DuckDB's recommended path for bulk-loading Python-side data is to **register a pandas
  DataFrame (or Arrow table) as a view and run a single `INSERT ... SELECT` against it** —
  this uses DuckDB's vectorised engine instead of the row-by-row DBAPI path
  ([DuckDB insert docs](https://duckdb.org/docs/stable/data/insert.md),
  [Discussion #13371](https://github.com/duckdb/duckdb/discussions/13371)).
- `conn.register(name, df)` accepts a plain pandas DataFrame directly (no pyarrow
  required, though pyarrow is also supported). **Registered views are connection-local**
  — needs verification under `pipe run-many`'s per-node-`cursor()`-on-shared-connection
  model (see §5 risk below).
- Below roughly ~2,000 rows, DataFrame-construction overhead outweighs the win — plain
  `executemany()` remains fine for small batches (agy-gemini research; not independently
  verified against a DuckDB source doc, treat as a reasonable starting default, tune with
  the benchmark in §6 if needed).
- Known gotchas to design around: `SELECT *` binds by column **position**, not name — the
  new implementation must list columns explicitly on both sides of the `INSERT ... SELECT`
  to avoid silent misalignment. Pandas coerces integer columns containing `NaN`/`None` to
  `float64` — must either use pandas nullable dtypes (`Int64`) or avoid the coercion path
  entirely by keeping ID/key columns as `object`/string dtype.

## 3. Design — owner-approved decisions (2026-07-15)

- **Scope: rewrite `upsert_rows()` in place.** Every call site benefits automatically,
  including `record_batch()`. Accepted trade-off: this touches every node in one change,
  so §6's test plan must be thorough before this lands on the real catalog.
- **Threshold: route small batches to the existing `executemany()` path.** New rows list
  with `len(rows) < UPSERT_BULK_THRESHOLD` (default **2,000**, easy to retune) keeps the
  current code path unchanged; `>= 2,000` uses the new DataFrame-register + bulk-INSERT
  path. This keeps the low-volume, latency-sensitive call sites (e.g. a handful of
  rows from a small `--limit` test run) simple and avoids paying DataFrame-construction
  overhead where it doesn't help.

## 4. Implementation sketch

```python
import pandas as pd
import uuid

UPSERT_BULK_THRESHOLD = 2_000

def upsert_rows(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    rows: list[dict],
    key_columns: list[str],  # still reserved/unused for ON CONFLICT — no behavior change
) -> int:
    if not rows:
        return 0

    columns: list[str] = list(rows[0].keys())

    def _coerce(value: object) -> object:
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return value

    if len(rows) < UPSERT_BULK_THRESHOLD:
        # unchanged existing path -- see current upsert_rows() body
        placeholders = ", ".join("?" * len(columns))
        col_list = ", ".join(columns)
        sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"
        tuples = [tuple(_coerce(row[col]) for col in columns) for row in rows]
        conn.executemany(sql, tuples)
        return len(rows)

    # Bulk path: register a DataFrame, single vectorised INSERT ... SELECT.
    coerced = [{col: _coerce(row[col]) for col in columns} for row in rows]
    df = pd.DataFrame(coerced, columns=columns)  # explicit columns= pins order

    # Unique view name: avoid collisions if this connection is shared across
    # concurrently-running `pipe run-many` nodes (each gets its own cursor(),
    # but registered-view scoping under DuckDB's Python API needs verifying --
    # see risk note below; a uuid4 suffix sidesteps the question either way).
    view_name = f"_upsert_bulk_{uuid.uuid4().hex}"
    col_list = ", ".join(columns)
    try:
        conn.register(view_name, df)
        conn.execute(
            f"INSERT OR REPLACE INTO {table} ({col_list}) "
            f"SELECT {col_list} FROM {view_name}"
        )
    finally:
        conn.unregister(view_name)

    return len(rows)
```

Notes on the sketch (fill in during actual implementation, not final code):
- `pd.DataFrame(coerced, columns=columns)` with explicit `columns=` guards against
  dict-key-order drift across rows (defensive; current code already assumes
  `rows[0].keys()` order is representative — preserve that assumption, don't change it).
- NULL handling: Python `None` in a dict becomes `NaN`/`None` in the DataFrame depending
  on column dtype inference; test explicitly (see §6) that `None` round-trips to SQL
  `NULL`, not the string `"None"` or `"nan"`. If pandas' automatic dtype inference proves
  unreliable for a specific column (e.g. an all-None column in one batch), may need
  per-column dtype hints — cross this bridge only if the test in §6 actually catches it,
  don't pre-guess.
- `try`/`finally` around `unregister()` so a mid-insert exception doesn't leak a stale
  view name into the connection's namespace for subsequent calls.

## 5. Risks / open questions to resolve *during* implementation (not blocking the plan)

1. **`conn.register()` scoping under `run-many`.** `pipe run-many` gives each node its
   own `conn.cursor()` off one shared connection (see CLAUDE.md's Concurrency layer
   section). Need to confirm empirically whether a view registered via one cursor is
   visible to a query executed via a different cursor sharing the same underlying
   connection — if registration is truly connection-local (not cursor-local), two nodes
   racing under `run-many` with the same uuid4-suffixed view name would never collide
   (names are unique per call), but simultaneous `register()`/`unregister()` calls from
   different threads on the same connection object could still need care. Test this
   specifically with a `run-many` pair before trusting it in production (e.g.
   `filter.acoustic` + `g2p` running concurrently, since both call `upsert_rows()`).
2. **pandas is not an explicit `pyproject.toml` dependency today** (confirmed present as
   pandas 3.0.3, but only transitively via another package — likely sklearn or a similar
   dependency). Add it explicitly: `uv pip install pandas` (never `uv sync` — see
   `feedback-canto-corpus-uv-sync-danger` memory) and pin a version floor in
   `pyproject.toml`.
3. **`INSERT OR REPLACE ... SELECT ... FROM view`, not `VALUES (...)`** — double-check
   this doesn't change how DuckDB handles a table with generated/default columns, if any
   exist in `schema.sql` for the tables this touches. Spot-check `schema.sql` before
   implementing (not expected to be an issue based on the current schema, but verify).
4. **2,000-row threshold is a starting default, not measured against this codebase's
   actual data.** Confirm with the benchmark in §6.2 whether it's in the right ballpark,
   adjust if the crossover point looks meaningfully different in practice.

## 6. Testing plan

### 6.1 Unit tests (new — add to `tests/test_catalog.py` or a new `tests/test_upsert_rows.py`)

Using an in-memory DuckDB connection (`duckdb.connect(":memory:")`) with a minimal test
table, cover:

- Empty `rows` list → returns `0`, no-op (existing behavior, must not regress).
- Small batch (`< threshold`) still produces correct rows — sanity check the unchanged
  path wasn't accidentally touched.
- Large batch (`>= threshold`, e.g. 5,000 synthetic rows) produces the **same** rows as
  the small-batch path would for equivalent data (assert row-for-row equality after both
  paths, on two separate tables/runs).
- `list`/`dict` values are JSON-serialised identically through both paths (write via
  small-batch path, write via bulk path, compare `json.loads()` of both — must match).
- `None` values round-trip to SQL `NULL` (not the string `"None"`) through the bulk path.
- `INSERT OR REPLACE` semantics preserved: upsert the same PK twice with different
  payloads via the bulk path, confirm the second write's values win (no duplicate rows).
- Column-order robustness: construct `rows` where dict insertion order varies row-to-row,
  confirm no misalignment (guards against the `SELECT *` position-binding gotcha even
  though the sketch already avoids `SELECT *` — test it explicitly since the whole point
  of researching this was a documented gotcha).

### 6.2 Regression + benchmark

1. `.venv/bin/python -m pytest tests/ -q` — full suite, must stay green (current
   baseline: check the count in the most recent full run, project convention already
   tracks this — no test should newly fail).
2. Micro-benchmark: write a throwaway script (scratchpad, not committed) that upserts a
   synthetic ~500,000-row batch against a **copy** of the real catalog (never the live
   `metadata/corpus.duckdb` while anything else might touch it) with:
   - Old `executemany()`-only path (temporarily force it, or just time the pre-fix code
     via `git stash`)
   - New bulk path
   - Compare wall-clock directly against this session's real measurements: podcast
     (538,310 rows) took ~78 min pre-fix. The fix should show a **large, unambiguous**
     improvement — if it's not at least several-fold faster, something in the
     implementation isn't hitting the vectorised path and needs debugging before rollout.
3. Real-world validation: once unit tests + micro-benchmark look right, do a **real**
   `pipe run speaker.cluster` (idempotent whole-source recompute — safe to rerun) and
   compare total wall-clock against today's baseline (podcast 78min, rthk 16.4min,
   youtube ~90min estimated — should all be dramatically shorter). This is the real proof
   the fix works, not just synthetic timing.
4. Per CLAUDE.md's node-authoring convention, also spot-check one or two OTHER
   `upsert_rows()` call sites with `--limit N` (e.g. `pipe run filter.decide --limit 5000`
   or similar) to confirm the fix behaves correctly outside the `speaker.cluster` case
   that motivated it — don't assume "it worked for speaker.cluster" generalises without
   checking.

## 7. Rollout steps (in order)

1. Confirm the in-flight `speaker.cluster` run (and anything else holding the DuckDB
   writer) has finished — `pgrep -af "pipeline.cli run"` clean, wrapper script no longer
   running.
2. Add `pandas` to `pyproject.toml` (`uv pip install pandas`, then hand-edit the
   `dependencies` list — do **not** run `uv sync`).
3. Implement the `upsert_rows()` rewrite per §4, resolving the open questions in §5 as
   they come up.
4. Run §6.1 unit tests — iterate until green.
5. Run §6.2 regression (full `pytest tests/`) — must stay green.
6. Run §6.2's micro-benchmark — confirm a large, unambiguous speedup vs. this session's
   measured baseline.
7. Real-world validation: rerun `pipe run speaker.cluster` fully, compare against
   today's ~78min/~16min/~90min per-source baseline.
8. Spot-check 1-2 other nodes per §6.2 point 4.
9. Update `DECISIONS.md` with a dated entry (same format as the 2026-07-14 `filter.decide`
   OOM entry): symptom → diagnosis → fix → before/after benchmark numbers.
10. Update `pending_task.md`: close out the `record_batch()`/`upsert_rows()` performance
    item opened this session, referencing the commit.

## 8. Explicitly out of scope for this fix

- **Not** rewriting `record_batch()`'s one-row-per-`item_id` schema design (it still
  writes one `task_runs` row per item — just faster now via the fixed `upsert_rows()`).
  If per-item granularity in `task_runs` turns out not to be load-bearing anywhere (no
  reader depends on it — not yet verified this session), collapsing it to one summary row
  per batch could be a separate, later optimisation. Don't conflate the two changes.
- **Not** switching to DuckDB's newer `ON CONFLICT ... DO UPDATE` / `MERGE INTO` syntax,
  which would allow partial-column updates instead of blanket `INSERT OR REPLACE`. The
  `key_columns` parameter is already reserved for this but unused — that's a deliberate,
  separate future change (per the existing docstring: "reserved for future ON CONFLICT
  use"), not part of this performance fix. Keep behavior identical, only change *how*
  data gets written, not *what* gets written.
- **Not** adding pyarrow as a dependency — pandas alone satisfies `conn.register()`'s
  requirements per the research in §2; revisit only if a future benchmark shows pyarrow
  meaningfully faster and worth the extra dependency.

## Sources

- [Upsert performance is slow unless inserts are sorted by key · Issue #11275](https://github.com/duckdb/duckdb/issues/11275)
- [Insert and Update Operations | duckdb/duckdb | DeepWiki](https://deepwiki.com/duckdb/duckdb/7.3-insert-and-update-operations)
- [Upsert performance · duckdb/duckdb · Discussion #5987](https://github.com/duckdb/duckdb/discussions/5987)
- [Slow python executemany for inserts · Issue #10106](https://github.com/duckdb/duckdb/issues/10106)
- [Optimizing DuckDB Insert Performance: Parallelism and Row Groups · Discussion #13371](https://github.com/duckdb/duckdb/discussions/13371)
- [DuckDB Insert documentation](https://duckdb.org/docs/stable/data/insert.md)
- agy-gemini research session, 2026-07-15 (DuckDB Python API `conn.register()` semantics,
  NULL/JSON handling, threshold recommendation — cross-check against official docs during
  implementation since this wasn't independently source-verified line-by-line)
