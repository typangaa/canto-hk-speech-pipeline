# Single-Process Concurrent Orchestrator — Status

> **2026-07-19**: 文件瘦身 —— Problem statement、empirical finding、design detail(22
> call site inventory、estimate table)、逐日 "What was implemented" 執行記錄全部搬去
> `docs/archive/ORCHESTRATOR_PLAN_DESIGN_DETAIL.md`,本檔淨留 status + 仲有效嘅
> "What's left"。日常用法/caveat 見 `CLAUDE.md` "Concurrency — `pipe run-many`" 一節
> (權威來源)。

Status: **23/23 call sites done — every node function that opens its own
`connect()` now accepts `conn=`, `pipe run-many` VALIDATED LIVE, running since
2026-07-07**. All 23 registered in `RUN_MANY_ADAPTERS`; regression tests for
every site in `tests/test_run_many.py`. Live-validated at full backlog scale
against the real catalog across multiple node pairings (`label.music` +
`filter.acoustic`, `asr.transcribe` + `filter.acoustic`, `ingest.commit` +
`speaker.embed`/`speaker.cluster`/`lang_screen.auto`/`tier.assign`, etc.).

Full design rationale, the 22-call-site inventory, the effort estimate, and
the day-by-day implementation log (2026-07-06, 2026-07-07 ×2) are archived at
`docs/archive/ORCHESTRATOR_PLAN_DESIGN_DETAIL.md`.

## What's left (not started, lower priority / incremental)

- GPU device-sharing policy across >2 concurrent GPU nodes (still not
  needed — `asr.transcribe`'s two models already use one GPU each; no
  3rd GPU node has been paired in yet).
- CPU worker-count tuning is empirical, not principled: the n=16 sweet spot
  for `filter.acoustic` alone was found by trial (8 too few, 36 way too
  many/oversubscribed) rather than derived from a formula — revisit if
  the workload mix changes materially (e.g. once decode cost per segment
  shifts with FLAC vs. WAV masters).
- ~~Docs: `CLAUDE.md` doesn't yet mention `run-many`~~ — done; `CLAUDE.md`'s
  "Concurrency — `pipe run-many`" section now documents usage + both known
  concurrency caveats (large-discovery-query starvation, same-device
  cross-model starvation).
