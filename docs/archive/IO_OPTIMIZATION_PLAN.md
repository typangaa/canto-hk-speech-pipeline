# Segment-Store I/O Optimization & Drive4 Skew Remediation Plan

**Status: ARCHIVED 2026-07-19 — executed via T15 (`pending_task.md` Done section /
`DECISIONS.md`).** Phase 0 (sysctl) and Phase 1 (dead sidecar cleanup) done 2026-07-12;
Phase 3 (embedding → catalog column migration) done 2026-07-12; Phase 2 (pending_delete
execution) was re-scoped by owner decision into a re-admit-for-re-transcription flow
instead of deletion; Phase 5 (hierarchical dirs) stays deferred. Kept here as the design
rationale behind `pipeline/nodes/speaker.py`'s embedding-column read path and
`pipeline/catalog/schema.sql`'s embedding column comment.

---

## 1. Problem statement

Observed during the 2026-07-12 T7 chain run (`metadata/logs/t7_chain_20260711.log`):

| speaker.cluster source | embeddings | `.npy` load time | actual clustering time |
|---|---|---|---|
| podcast | 238,649 | ~45 s (page-cache warm) | ~9 s |
| rthk | 62,933 | **~33 min** (~31 ms/file) | ~8 s |
| youtube | 361,115 | **~11 min** (~1.9 ms/file) | ~9 s |

The clustering algorithm itself is never the bottleneck (sample capped at
`SPK_CLUSTER_SAMPLE_MAX=12000`, fit completes in seconds regardless of source
size). **~98% of the stage's wall-clock is spent opening hundreds of thousands
of tiny per-segment sidecar files.**

Measured facts ruling out the usual suspects:

- All three shard drives are SATA **SSDs** (`lsblk ROTA=0`) — not seek latency.
- A single *cached* `np.load()` of one sidecar takes **0.066 ms** — not read
  bandwidth.
- `iostat` during the stall shows `%util ≈ 0` on sdb/sdc/sdd — the disks are
  idle; time is going into **directory-entry lookup on enormous flat
  directories** (dentry/inode cache misses, ext4 htree traversal), amplified
  by the same dirs being polluted with millions of dead legacy files (see §2).
- The wide per-file latency variance (0.066 ms cached → 31 ms cold on the
  most polluted drive) is the signature of dentry-cache thrashing, not device
  limits.

Affected beyond `speaker.cluster`: every node doing sidecar-reuse existence
scans over the same directories (`speaker.embed` reuse pass, `segment.diarize`
sidecar checks), plus general `ls`/backup/rsync operations on these dirs.

## 2. Investigation: why Drive4 holds 5–7× more files

Per-directory file counts (2026-07-12):

| dir | Drive2 | Drive3 | Drive4 |
|---|---|---|---|
| segments/youtube | 171,438 | 174,662 | **1,201,410** |
| segments/podcast | 128,939 | 131,534 | **1,156,031** |
| segments/rthk | 33,762 | 37,464 | **190,025** |
| **total** | 334,139 | 343,660 | **2,547,466** |

Extension breakdown of Drive4's 2,547,466 files — **every file accounted for**:

| type | count | what it is | still needed? |
|---|---|---|---|
| `.transcript.json` | **1,310,284** (yt 585,266 + podcast 635,219 + rthk 89,799) | legacy `scripts/04_transcribe.py` ASR sidecars | ❌ superseded by `asr_results`/`asr_agreement` catalog tables (P0 import). Only soft consumers: `recover.orphans` (one-time, already run 2026-07-06) and `pipeline/golden.py` golden-snapshot builder (`_SEGMENTS_SUFFIXES`) |
| `.wav` | 882,655 | mixed: see split below | partially |
| — of which pending_delete orphans | **≈578,843** | queued by `recover.orphans` 2026-07-06 (730,824 scanned → 151,981 recovered, 578,843 pending_delete), never deleted by design | ❌ awaiting owner-approved deletion |
| — of which catalog-tracked | ≈303,800 | active segment audio | ✅ |
| `.embed.npy` | 323,805 | ECAPA embedding sidecars | ✅ today; obsolete after Phase 3 below |
| `.flac` | 20,048 | active segment audio (post-2026-07-05) | ✅ |
| `_segments.jsonl` | 10,674 | legacy diarization sidecars — **still actively read** by `segment.diarize`'s reuse-first pass (`segments_root` points at Drive4) | ✅ keep |

**Root causes of the skew** (verified by arithmetic, not conjecture):

1. **Drive4 was the original single segments home** (`segments_root:
   /mnt/Drive4/canto/segments` — still the config value). P5-C's
   `rebalance.segments` (2026-07-06) moved only *catalog-tracked* segment
   audio 3-way; everything else stayed.
2. **`recover.orphans` recovered 151,981 orphans in place on Drive4** — they
   were inserted into the catalog with their existing Drive4 paths and never
   re-sharded. Proof: Drive4 `.npy` (323,805) − recovered (151,981) =
   **171,824 ≈ Drive3's 171,826** — i.e. minus the recovered orphans, Drive4
   holds exactly a fair ⅓ shard.
3. **The 578,843-file pending_delete queue was never executed** (correct per
   design — needs owner approval). Proof: Drive4 audio (902,703) −
   catalog-tracked (=`npy` count 323,805) = 578,898 ≈ 578,843 queued.
4. **1.31M legacy `.transcript.json` sidecars were never cleaned** after the
   P0 catalog cutover made them redundant.

So Drive4's excess is entirely **dead or misplaced legacy data** — the
`shard_index()` hash itself is fine and needs no change.

## 3. Implementation phases

Ordering rationale: Phase 3 (embeddings → catalog) deliberately comes
**before** Phase 4 (re-shard recovered orphans), because `rebalance.segments`
moves only `segments.audio_path` — it knows nothing about `.embed.npy`
sidecars or `speaker_embeddings.embedding_ref` (verified in
`pipeline/nodes/rebalance.py`). Doing Phase 3 first makes all npy sidecars
obsolete, so the re-shard never has to move them or re-point refs.

### Phase 0 — zero-risk quick win (minutes)

- `sudo sysctl vm.vfs_cache_pressure=50` (persist in
  `/etc/sysctl.d/99-canto-corpus.conf`). Machine has 257 GB RAM with 154 GB
  already in page cache; biasing the kernel toward retaining dentries/inodes
  directly attacks the observed cache-miss pattern.
- Mounts are already `relatime` (acceptable; `noatime` remount is optional
  and not worth a remount window on its own).
- Expected effect: helps repeat scans only; does **not** fix the structural
  problem. Do it, but don't measure success by it.

### Phase 1 — delete 1.31M legacy `.transcript.json` sidecars (biggest dentry win, ~1 session)

1. **Verify supersession** (needs the DuckDB writer free): sample N=1000
   random `.transcript.json` files, confirm each segment id has matching
   `asr_results` rows in the catalog (the P0 import consumed exactly these
   files). Spot-check text equality.
2. **Archive before delete** (cheap reversibility): `tar` (no compression
   needed — they're tiny JSON) each source's `.transcript.json` set to
   `/mnt/Drive3/canto/archive/transcript_sidecars_{source}.tar` (Drive3 has
   1.6 TB free). ~1.31M files ≈ a few GB.
3. **Owner approval checkpoint** — then delete via
   `find <dir> -maxdepth 1 -name '*.transcript.json' -delete` per source dir.
4. **Follow-up code note**: `pipeline/golden.py` `_SEGMENTS_SUFFIXES` still
   lists `.transcript.json` — a future golden-snapshot rebuild will simply
   find no sidecars (non-fatal), but update the comment/constant in the same
   change so it doesn't mislead. `recover.orphans` is one-time and already
   run; its transcript-reading path stays valid against the tar archive if
   ever re-needed.
- **Effect**: Drive4 youtube dir 1,201,410 → ~616k files; podcast 1,156,031 →
  ~521k; rthk 190,025 → ~100k. Roughly **halves** Drive4's dentry load for
  zero information loss.

### Phase 2 — execute the pending_delete queue: 578,843 orphan WAVs (owner-approval REQUIRED, ~1 session)

These are pre-catalog segment WAVs that `recover.orphans` classified as not
worth recovering (failed pregate or unclassifiable). Deleting them is the
single biggest space + dentry reclaim, and it was always the intended fate of
this queue — it just needs the explicit owner sign-off the design reserves.

1. **Verify the queue** (DB free): `SELECT status, COUNT(*) FROM
   orphan_segments GROUP BY status` — confirm ≈578,843 `pending_delete`; spot
   check 50 paths exist on disk and are absent from `segments`.
2. **Decide archive-vs-delete with owner**: at ~48 kHz mono WAV these are
   several hundred GB — a full archive is likely impractical; propose instead
   a **checksum + path manifest** (`metadata/orphans_deleted_manifest.jsonl`,
   sha256 + size + path) for audit, then true deletion. Owner decides.
3. Delete in batches from the `orphan_segments` table's own path list (never
   a bare glob — only rows explicitly marked `pending_delete`), updating each
   row's status to `deleted` + timestamp as it goes (idempotent, resumable).
4. **Effect**: Drive4 −578k files, frees roughly 0.3–0.5 TB. Combined with
   Phase 1, Drive4 drops from 2.55M to ≈660k files.

### Phase 3 — move embeddings out of per-file sidecars into the catalog (~1 day, code change)

**Chosen design: store the 192-dim float32 vector in DuckDB itself**
(`speaker_embeddings.embedding FLOAT[192]` array column), replacing per-file
`.npy` sidecars. Considered alternatives:

- *Packed per-source `.npy` + offset index files*: works, but adds a second
  storage artifact with its own consistency rules — rejected in favour of the
  catalog, which is already the project's single source of truth.
- *FAISS / cuML GPU clustering* (researched 2026-07-12 via agy-gemini): does
  not address this bottleneck at all — the clustering compute is already <10 s;
  cuML's AgglomerativeClustering only supports single linkage (chaining risk),
  NeMo NME-SC OOMs at this scale. GPU clustering only becomes relevant if
  `SPK_CLUSTER_SAMPLE_MAX` is ever raised 10×+; out of scope here.

Size check: ~667k embeddings × 192 × 4 B ≈ **512 MB** in-table — comfortably
fine for DuckDB, and loading becomes one columnar SQL scan instead of ~10⁵–10⁶
file opens (the entire §1 stall class disappears).

Steps:
1. `ALTER TABLE speaker_embeddings ADD COLUMN embedding FLOAT[192]` (via a
   small catalog migration in `pipeline/catalog/`).
2. `speaker.embed` worker: return the vector through the existing
   JSONL-over-stdio protocol (or keep writing the npy AND upserting the
   column during a transition window); supervisor upserts the array with the
   row. Keep `embedding_ref` populated during transition for rollback.
3. `speaker.cluster`: read `SELECT id, embedding FROM speaker_embeddings
   WHERE source = ? AND embedding IS NOT NULL` → `np.array`; fall back to
   `_load_npy()` only for rows where the column is NULL.
4. One-time backfill node (`speaker.embed --backfill-catalog` or a small
   `embed.backfill` node): stream existing sidecars into the column in
   batches (this is the *last* time the slow per-file read happens), with the
   standard provenance-tagged anti-join so it's resumable.
5. After backfill verification (`COUNT(*) WHERE embedding IS NULL` = 0 for
   rows with a ref; sample-compare 1000 vectors bit-exact against their npy),
   **owner approval checkpoint** → delete all `.embed.npy` sidecars
   (~663k files across the three drives) and null/retire `embedding_ref`.
6. Follow standard node conventions (conn= injection, RUN_MANY_ADAPTERS,
   `--limit` smoke test against the real catalog first).

- **Effect**: `speaker.cluster` load phase 45 min → seconds; removes ~663k
  files across all drives; `speaker.embed`'s reuse pass becomes a pure SQL
  anti-join (no more 44k-file existence scan).

### Phase 4 — re-shard the 151,981 recovered orphans off Drive4 (~half day)

After Phase 3 there are no npy sidecars to worry about, so this is a pure
audio move:

1. Re-run `pipe run rebalance.segments` — it is already idempotent
   (anti-joins on `segment_shard_migrations`) and already computes targets
   via `shard_index()`; the 151,981 recovered rows are exactly the segments
   it hasn't migrated yet. Verify with `--limit 100` first.
2. Confirm the copy→verify→delete flow updates `segments.audio_path` and the
   migration ledger as it did in P5-C.
3. **Effect**: Drive4 finally holds a fair ⅓ shard. Expected end state per
   drive ≈ 220–230k files total (audio + the small active `_segments.jsonl`
   set), largest single dir ≈ 120k — well inside comfortable ext4 territory.

### Phase 5 — hierarchical segment directories (DEFERRED, likely unnecessary)

Splitting `segments/{source}/` into hashed/dated subdirs would touch
`config/storage_layout.py`, every path-writing node, and 1.5M+ existing
files. **Decision rule**: only revisit if, *after* Phases 1–4, a measured
directory-scan stage still shows >5 ms/file cold-lookup latency. Expected to
be unnecessary once the biggest dir is ~120k entries. Park in
`pending_task.md` as a conditional P6 item; do not build now.

## 4. Expected outcome summary

| metric | before | after Phases 0–4 |
|---|---|---|
| Drive4 file count | 2,547,466 | ≈ 230k |
| Largest single directory | 1,201,410 entries | ≈ 120k entries |
| Total sidecar files (all drives) | ≈ 2.0M (`transcript.json` + `npy`) | ≈ 10.7k (`_segments.jsonl` only) |
| `speaker.cluster` embedding load | 45 min (this run, 3 sources) | seconds (one SQL scan) |
| Disk reclaimed on Drive4 | — | ≈ 0.3–0.5 TB |
| Shard balance (catalog-tracked) | ⅓ / ⅓ / ⅔-ish | ⅓ / ⅓ / ⅓ |

## 5. Risks & rollback

| risk | mitigation |
|---|---|
| `.transcript.json` needed later (golden rebuild, audit) | tar archive on Drive3 before delete (Phase 1.2) |
| pending_delete removes something recoverable | delete strictly from `orphan_segments` rows (never glob); checksum manifest for audit; owner reviews the bucket breakdown before approving |
| embedding column ≠ npy content | transition window keeps both; bit-exact sample verification before sidecar delete |
| rebalance re-run misbehaves | idempotent ledger + copy-verify-then-delete flow already proven in P5-C; `--limit 100` smoke test first |
| DuckDB file grows ~512 MB | acceptable (catalog lives on the NVMe root, 895 GB free); `VACUUM`/`CHECKPOINT` after backfill |
| concurrent writer-lock conflicts | run each phase's DB work standalone or via `run-many`; never while the T7 chain holds the writer |

## 6. Execution order & effort

```
Phase 0  sysctl                          minutes   zero risk       no approval needed
Phase 1  transcript.json cleanup         ~1 session tar+verify+rm   OWNER APPROVAL (delete)
Phase 3  embeddings → catalog column     ~1 day    code + backfill OWNER APPROVAL (npy delete only)
Phase 2  pending_delete execution        ~1 session verify+rm       OWNER APPROVAL (delete)
Phase 4  re-shard recovered orphans      ~half day rebalance rerun  no delete beyond verified moves
Phase 5  hierarchical dirs               DEFERRED  conditional     —
```

(Phases 1/2/3 are independent of each other and can be reordered; only
Phase 4 should wait for Phase 3 as explained in §3. Phase 2 can run any time
the owner approves.)

Add to `pending_task.md` as a new tiered task (suggest 🟡 T15) referencing
this document once the plan is accepted.
