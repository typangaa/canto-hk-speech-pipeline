#!/usr/bin/env python3
"""
pipeline/cli.py
P0 CLI entry point: `pipe catalog build|verify|rebuild`, `pipe golden build`.
Usage: python -m pipeline.cli catalog build
       python -m pipeline.cli catalog verify
       python -m pipeline.cli golden build
"""

import argparse
import sys


def cmd_catalog_build(args: argparse.Namespace) -> int:
    from pipeline.catalog.ingest import main as ingest_main
    sys.argv = ["ingest.py"] + (["--dry-run"] if args.dry_run else [])
    return ingest_main()


def cmd_catalog_verify(args: argparse.Namespace) -> int:
    from pipeline.catalog.verify import main as verify_main
    return verify_main()


def cmd_catalog_rebuild(args: argparse.Namespace) -> int:
    # P0: rebuild == build (import_* functions already TRUNCATE + re-INSERT,
    # so there's no separate "incremental" state to reset). A real
    # journal-replay rebuild lands once P1's orchestrator writes journals.
    print("pipe catalog rebuild == pipe catalog build in P0 "
          "(no journals exist yet; see docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md §3.2)")
    return cmd_catalog_build(args)


def cmd_golden_build(args: argparse.Namespace) -> int:
    from pipeline.golden import main as golden_main
    return golden_main()


def cmd_logs_prune(args: argparse.Namespace) -> int:
    from pipeline.tools.prune_logs import prune_logs

    result = prune_logs(
        gzip_after_days=args.gzip_after_days,
        delete_after_days=args.delete_after_days,
        dry_run=args.dry_run,
    )
    verb = "Would gzip" if args.dry_run else "Gzipped"
    print(f"{verb} {len(result['gzipped'])} file(s)")
    verb = "Would delete" if args.dry_run else "Deleted"
    print(f"{verb} {len(result['deleted'])} archive(s)")
    if not args.dry_run:
        print(f"Bytes reclaimed: {result['bytes_reclaimed']:,}")
    return 0


def cmd_chain_run(args: argparse.Namespace) -> int:
    from pipeline.config import LOGS_DIR
    from pipeline.tools.chain_runner import run_chain, _parse_round_set
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOGS_DIR / f"chain_runner_{stamp}.log"

    result = run_chain(
        only=_parse_round_set(args.only),
        skip=_parse_round_set(args.skip),
        devices=args.devices,
        dry_run=args.dry_run,
        log_path=log_path,
    )
    failed = [r for r in result["rounds"] if r.get("returncode", 0) != 0]
    print(f"\nChain {'(dry-run) ' if args.dry_run else ''}done: "
          f"{len(result['rounds'])} round(s), {len(failed)} failed.")
    if not args.dry_run:
        print(f"Log: {log_path}")
    return 1 if failed else 0


def cmd_chain_stream(args: argparse.Namespace) -> int:
    import shlex
    from datetime import datetime, timezone

    from pipeline.config import LOGS_DIR
    from pipeline.tools.stream_drain import run_stream

    downstream_args = {}
    for pair in args.downstream_args:
        node, _, raw = pair.partition("=")
        downstream_args[node] = shlex.split(raw)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOGS_DIR / f"stream_drain_{stamp}.log"

    result = run_stream(
        upstream=args.upstream,
        upstream_args=shlex.split(args.upstream_args) if args.upstream_args else [],
        downstream=args.downstream,
        downstream_args=downstream_args,
        poll_interval_s=args.poll_interval,
        log_path=log_path,
    )
    print(f"\nStream done: upstream rc={result['upstream_returncode']}, "
          f"{len(result['polls'])} poll(s), final drain rc={result['final_drain']['returncode']}")
    print(f"Log: {log_path}")
    return 0 if result["upstream_returncode"] == 0 else 1


def cmd_run_ingest_download(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.ingest_download import run_ingest_download

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_ingest_download(
        source=args.source, dry_run=args.dry_run, limit=args.limit,
        cookies_from_browser=args.cookies_from_browser,
    ))
    print(f"\nDone: {result}")
    return 0


def cmd_run_ingest_commit(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.ingest_download import run_ingest_commit

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_ingest_commit())
    print(f"\nDone: {result}")
    return 0


def cmd_run_ingest_probe(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.ingest_probe import run_ingest_probe

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_ingest_probe(
        workers=args.workers, batch_size=args.batch, limit=args.limit,
    ))
    print(f"\nDone: {result}")
    return 0


def cmd_run_label_prosody(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.label_prosody import run_label_prosody

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_label_prosody(
        n_workers=args.workers,
        threads_per_worker=args.threads,
        batch_size=args.batch,
        limit=args.limit,
    ))
    print(f"\nDone: {result}")
    return 0


def cmd_run_label_suite(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.label_suite import run_label_suite

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    devices = [d.strip() for d in args.devices.split(",")]
    result = asyncio.run(run_label_suite(
        devices,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
        mem_fraction=args.mem_fraction,
        limit=args.limit,
    ))
    print(f"\nDone: {result}")
    return 0


def cmd_run_align_chars(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.align import run_align_chars

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    devices = [d.strip() for d in args.devices.split(",")]
    result = asyncio.run(run_align_chars(
        devices,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
        mem_fraction=args.mem_fraction,
        limit=args.limit,
        prefetch=args.prefetch,
        io_workers=args.io_workers,
    ))
    print(f"\nDone: {result}")
    return 0


def _check_asr_models_enabled(model_keys: list[str]) -> None:
    """Guard rail (added 2026-07-10, DECISIONS.md): refuse to dispatch any ASR model
    ASR_MODELS marks disabled (currently whisper_v3 and canto_ft, both retired for
    measured inaccuracy/throughput -- see pipeline/nodes/asr.py's module docstring).
    Without this, someone
    could still pass --models with the old 4-model list from habit/an old script and
    silently burn GPU time on a model whose output is never read by asr.agreement/
    manifest.build anymore."""
    from pipeline.nodes.asr import ASR_MODELS, is_model_enabled

    for key in model_keys:
        if key in ASR_MODELS and not is_model_enabled(key):
            raise SystemExit(
                f"--models includes {key!r}, which is retired (ASR_MODELS[{key!r}]['enabled'] "
                f"= False) -- see pipeline/nodes/asr.py's module docstring and DECISIONS.md. "
                f"Remove it from --models."
            )


def cmd_run_asr_transcribe(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.asr import run_asr_transcribe

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model_keys = [m.strip() for m in args.models.split(",")]
    devices = [d.strip() for d in args.devices.split(",")]
    if len(model_keys) != len(devices):
        raise SystemExit(f"--models ({len(model_keys)}) and --devices ({len(devices)}) must have the same count")
    _check_asr_models_enabled(model_keys)
    assignments = list(zip(model_keys, devices))
    result = asyncio.run(run_asr_transcribe(
        assignments,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
        mem_fraction=args.mem_fraction,
        limit=args.limit,
        prefetch=args.prefetch,
        io_workers=args.io_workers,
    ))
    print(f"\nDone: {result}")
    return 0


def cmd_run_asr_agreement(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.asr import run_asr_agreement

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_asr_agreement(batch_size=args.batch, limit=args.limit))
    print(f"\nDone: {result}")
    return 0


def cmd_run_filter_text(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.filter import run_filter_text

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_filter_text(batch_size=args.batch, limit=args.limit))
    print(f"\nDone: {result}")
    return 0


def cmd_run_filter_acoustic(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.filter import run_filter_acoustic

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gpu_ids = [int(x) for x in args.gpu.split(",")] if args.gpu else None
    result = asyncio.run(run_filter_acoustic(
        n_workers=args.workers,
        threads_per_worker=args.threads,
        batch_size=args.batch,
        limit=args.limit,
        gpu_ids=gpu_ids,
    ))
    print(f"\nDone: {result}")
    return 0


def cmd_run_filter_decide(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.filter import run_filter_decide

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_filter_decide(batch_size=args.batch, limit=args.limit))
    print(f"\nDone: {result}")
    return 0


def cmd_run_g2p(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.g2p import run_g2p

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_g2p(batch_size=args.batch, limit=args.limit))
    print(f"\nDone: {result}")
    return 0


def cmd_run_pause_plan(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.pause_plan import run_pause_plan

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_pause_plan(batch_size=args.batch, limit=args.limit))
    print(f"\nDone: {result}")
    return 0


def cmd_run_lang_screen_auto(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.lang_screen import run_lang_screen_auto

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    devices = [d.strip() for d in args.devices.split(",")]
    result = asyncio.run(run_lang_screen_auto(
        devices,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
        mem_fraction=args.mem_fraction,
        limit=args.limit,
    ))
    print(f"\nDone: {result}")
    return 0


def cmd_run_label_music(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.label_music import run_label_music

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    devices = [d.strip() for d in args.devices.split(",")]
    result = asyncio.run(run_label_music(
        devices,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
        mem_fraction=args.mem_fraction,
        limit=args.limit,
    ))
    print(f"\nDone: {result}")
    return 0


def cmd_run_speaker_embed(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.speaker import run_speaker_embed

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    devices = [d.strip() for d in args.devices.split(",")]
    result = asyncio.run(run_speaker_embed(
        devices,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
        mem_fraction=args.mem_fraction,
        limit=args.limit,
        verify_existing=args.verify_existing,
    ))
    print(f"\nDone: {result}")
    return 0


def cmd_run_segment_diarize(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.segment import run_segment_diarize

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    devices = [d.strip() for d in args.devices.split(",")]
    result = asyncio.run(run_segment_diarize(
        devices,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
        mem_fraction=args.mem_fraction,
        limit=args.limit,
    ))
    print(f"\nDone: {result}")
    return 0


async def _run_many_adapt_filter_acoustic(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.filter import run_filter_acoustic
    return await run_filter_acoustic(
        conn=conn,
        n_workers=args.workers,
        threads_per_worker=args.threads,
        batch_size=args.batch,
        limit=args.limit,
    )


async def _run_many_adapt_segment_diarize(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.segment import run_segment_diarize
    devices = [d.strip() for d in args.devices.split(",")]
    return await run_segment_diarize(
        devices,
        conn=conn,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
        mem_fraction=args.mem_fraction,
        limit=args.limit,
    )


async def _run_many_adapt_label_music(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.label_music import run_label_music
    devices = [d.strip() for d in args.devices.split(",")]
    return await run_label_music(
        devices,
        conn=conn,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
        mem_fraction=args.mem_fraction,
        limit=args.limit,
    )


async def _run_many_adapt_asr_transcribe(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.asr import run_asr_transcribe
    model_keys = [m.strip() for m in args.models.split(",")]
    devices = [d.strip() for d in args.devices.split(",")]
    if len(model_keys) != len(devices):
        raise SystemExit(f"--models ({len(model_keys)}) and --devices ({len(devices)}) must have the same count")
    _check_asr_models_enabled(model_keys)
    assignments = list(zip(model_keys, devices))
    return await run_asr_transcribe(
        assignments,
        conn=conn,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
        prefetch=args.prefetch,
        io_workers=args.io_workers,
        mem_fraction=args.mem_fraction,
        limit=args.limit,
    )


async def _run_many_adapt_asr_agreement(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.asr import run_asr_agreement
    return await run_asr_agreement(conn=conn, batch_size=args.batch, limit=args.limit)


async def _run_many_adapt_ingest_commit(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.ingest_download import run_ingest_commit
    return await run_ingest_commit(conn=conn)


async def _run_many_adapt_speaker_embed(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.speaker import run_speaker_embed
    devices = [d.strip() for d in args.devices.split(",")]
    return await run_speaker_embed(
        devices,
        conn=conn,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
        mem_fraction=args.mem_fraction,
        limit=args.limit,
        verify_existing=args.verify_existing,
    )


async def _run_many_adapt_speaker_cluster(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.speaker import run_speaker_cluster
    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None
    return await run_speaker_cluster(
        conn=conn, threshold=args.threshold, sources=sources, limit=args.limit,
    )


async def _run_many_adapt_lang_screen_auto(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.lang_screen import run_lang_screen_auto
    devices = [d.strip() for d in args.devices.split(",")]
    return await run_lang_screen_auto(
        devices,
        conn=conn,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
        mem_fraction=args.mem_fraction,
        limit=args.limit,
    )


async def _run_many_adapt_tier_assign(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.tier import run_tier_assign
    return await run_tier_assign(conn=conn, batch_size=args.batch, limit=args.limit)


async def _run_many_adapt_quality_tier_assign(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.quality_tier import run_quality_tier_assign
    return await run_quality_tier_assign(conn=conn, batch_size=args.batch, limit=args.limit)


async def _run_many_adapt_calibrate_sample(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.calibrate import run_calibrate_sample
    return await run_calibrate_sample(
        conn=conn, n=args.n, tier=args.tier, min_agreement=args.min_agreement,
        code_switch=args.code_switch, order_by=args.order,
    )


async def _run_many_adapt_filter_text(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.filter import run_filter_text
    return await run_filter_text(conn=conn, batch_size=args.batch, limit=args.limit)


async def _run_many_adapt_filter_decide(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.filter import run_filter_decide
    return await run_filter_decide(conn=conn, batch_size=args.batch, limit=args.limit)


async def _run_many_adapt_segment_vad_cut(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.segment import run_segment_vad_cut
    return await run_segment_vad_cut(conn=conn, n_threads=args.threads, limit=args.limit)


async def _run_many_adapt_pregate_snr(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.segment import run_pregate_snr
    return await run_pregate_snr(
        conn=conn, min_snr=args.min_snr, min_dnsmos=args.min_dnsmos,
        n_threads=args.threads, batch_size=args.batch, limit=args.limit,
    )


async def _run_many_adapt_g2p(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.g2p import run_g2p
    return await run_g2p(conn=conn, batch_size=args.batch, limit=args.limit)


async def _run_many_adapt_pause_plan(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.pause_plan import run_pause_plan
    return await run_pause_plan(conn=conn, batch_size=args.batch, limit=args.limit)


async def _run_many_adapt_ingest_probe(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.ingest_probe import run_ingest_probe
    return await run_ingest_probe(
        conn=conn, workers=args.workers, batch_size=args.batch, limit=args.limit,
    )


async def _run_many_adapt_label_suite(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.label_suite import run_label_suite
    devices = [d.strip() for d in args.devices.split(",")]
    return await run_label_suite(
        devices,
        conn=conn,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
        mem_fraction=args.mem_fraction,
        limit=args.limit,
    )


async def _run_many_adapt_label_prosody(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.label_prosody import run_label_prosody
    return await run_label_prosody(
        conn=conn, n_workers=args.workers, threads_per_worker=args.threads,
        batch_size=args.batch, limit=args.limit,
    )


async def _run_many_adapt_align_chars(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.align import run_align_chars
    devices = [d.strip() for d in args.devices.split(",")]
    return await run_align_chars(
        devices,
        conn=conn,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
        mem_fraction=args.mem_fraction,
        limit=args.limit,
        prefetch=args.prefetch,
        io_workers=args.io_workers,
    )


async def _run_many_adapt_recover_orphans(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.recover_orphans import run_recover_orphans
    return await run_recover_orphans(conn=conn, limit=args.limit)


async def _run_many_adapt_reingest_pending(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.recover_orphans import run_reingest_pending
    return await run_reingest_pending(conn=conn, limit=args.limit)


async def _run_many_adapt_embed_backfill(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.speaker import run_embed_backfill
    return await run_embed_backfill(conn=conn, limit=args.limit)


async def _run_many_adapt_rebalance_segments(args: argparse.Namespace, conn) -> dict:
    if args.delete_verified:
        from pipeline.nodes.rebalance import run_rebalance_delete_verified
        return await run_rebalance_delete_verified(conn=conn, limit=args.limit)
    from pipeline.nodes.rebalance import run_rebalance_copy
    return await run_rebalance_copy(
        conn=conn, workers=args.workers, batch_size=args.batch,
        batch_gb=args.batch_gb, limit=args.limit,
    )


async def _run_many_adapt_raw_flac(args: argparse.Namespace, conn) -> dict:
    if args.delete_verified:
        from pipeline.nodes.raw_flac import run_raw_flac_delete_verified
        return await run_raw_flac_delete_verified(conn=conn, limit=args.limit)
    from pipeline.nodes.raw_flac import run_raw_flac_transcode
    return await run_raw_flac_transcode(
        conn=conn, workers=args.workers, batch_size=args.batch,
        batch_gb=args.batch_gb, limit=args.limit,
    )


# Nodes wired up for `pipe run-many`. A node must accept a `conn=` kwarg
# (dependency-injected DuckDB connection/cursor) before it can be added here —
# see docs/archive/ORCHESTRATOR_PLAN_DESIGN_DETAIL.md for the full call-site
# inventory and the priority order for extending this incrementally.
RUN_MANY_ADAPTERS = {
    "filter.acoustic": _run_many_adapt_filter_acoustic,
    "segment.diarize": _run_many_adapt_segment_diarize,
    "label.music": _run_many_adapt_label_music,
    "asr.transcribe": _run_many_adapt_asr_transcribe,
    "asr.agreement": _run_many_adapt_asr_agreement,
    "ingest.commit": _run_many_adapt_ingest_commit,
    "speaker.embed": _run_many_adapt_speaker_embed,
    "speaker.cluster": _run_many_adapt_speaker_cluster,
    "lang_screen.auto": _run_many_adapt_lang_screen_auto,
    "tier.assign": _run_many_adapt_tier_assign,
    "quality_tier.assign": _run_many_adapt_quality_tier_assign,
    "calibrate.sample": _run_many_adapt_calibrate_sample,
    "filter.text": _run_many_adapt_filter_text,
    "filter.decide": _run_many_adapt_filter_decide,
    "segment.vad_cut": _run_many_adapt_segment_vad_cut,
    "pregate.snr": _run_many_adapt_pregate_snr,
    "g2p": _run_many_adapt_g2p,
    "pause.plan": _run_many_adapt_pause_plan,
    "ingest.probe": _run_many_adapt_ingest_probe,
    "label.suite": _run_many_adapt_label_suite,
    "label.prosody": _run_many_adapt_label_prosody,
    "align.chars": _run_many_adapt_align_chars,
    "recover.orphans": _run_many_adapt_recover_orphans,
    "recover.reingest_pending": _run_many_adapt_reingest_pending,
    "embed.backfill": _run_many_adapt_embed_backfill,
    "rebalance.segments": _run_many_adapt_rebalance_segments,
    "raw.flac": _run_many_adapt_raw_flac,
}


def split_run_many_groups(tokens: list[str]) -> list[list[str]]:
    """Split `pipe run-many` remainder tokens into per-node argv groups on
    literal '--' separators, e.g.
    ["segment.diarize", "--devices", "cuda:0", "--", "filter.acoustic", "--workers", "8"]
    -> [["segment.diarize", "--devices", "cuda:0"], ["filter.acoustic", "--workers", "8"]]
    """
    groups: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if tok == "--":
            if current:
                groups.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        groups.append(current)
    return groups


def cmd_run_segment_vad_cut(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.segment import run_segment_vad_cut

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_segment_vad_cut(
        n_threads=args.threads,
        limit=args.limit,
    ))
    print(f"\nDone: {result}")
    return 0


def cmd_run_pregate_snr(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.segment import run_pregate_snr

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_pregate_snr(
        min_snr=args.min_snr,
        min_dnsmos=args.min_dnsmos,
        n_threads=args.threads,
        batch_size=args.batch,
        limit=args.limit,
    ))
    print(f"\nDone: {result}")
    return 0


def cmd_run_raw_flac(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.delete_verified:
        from pipeline.nodes.raw_flac import run_raw_flac_delete_verified
        result = asyncio.run(run_raw_flac_delete_verified(limit=args.limit))
    else:
        from pipeline.nodes.raw_flac import run_raw_flac_transcode
        result = asyncio.run(run_raw_flac_transcode(
            workers=args.workers,
            batch_size=args.batch,
            batch_gb=args.batch_gb,
            limit=args.limit,
        ))
    print(f"\nDone: {result}")
    return 0


def cmd_run_recover_orphans(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.recover_orphans import run_recover_orphans

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_recover_orphans(limit=args.limit))
    print(f"\nDone: {result}")
    return 0


def cmd_run_reingest_pending(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.recover_orphans import run_reingest_pending

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_reingest_pending(limit=args.limit))
    print(f"\nDone: {result}")
    return 0


def cmd_run_rebalance_segments(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.delete_verified:
        from pipeline.nodes.rebalance import run_rebalance_delete_verified
        result = asyncio.run(run_rebalance_delete_verified(limit=args.limit))
    else:
        from pipeline.nodes.rebalance import run_rebalance_copy
        result = asyncio.run(run_rebalance_copy(
            workers=args.workers,
            batch_size=args.batch,
            batch_gb=args.batch_gb,
            limit=args.limit,
        ))
    print(f"\nDone: {result}")
    return 0


def cmd_run_tier_assign(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.tier import run_tier_assign

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_tier_assign(batch_size=args.batch, limit=args.limit))
    print(f"\nDone: {result}")
    return 0


def cmd_run_quality_tier_assign(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.quality_tier import run_quality_tier_assign

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_quality_tier_assign(batch_size=args.batch, limit=args.limit))
    print(f"\nDone: {result}")
    return 0


def cmd_run_calibrate_sample(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.calibrate import run_calibrate_sample

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(
        run_calibrate_sample(
            n=args.n, tier=args.tier, min_agreement=args.min_agreement, code_switch=args.code_switch,
            order_by=args.order,
        )
    )
    print(f"\nDone: {result}")
    return 0


def cmd_calibrate_serve(args: argparse.Namespace) -> int:
    import logging
    from http.server import ThreadingHTTPServer

    import duckdb

    from pipeline.catalog.catalog import CATALOG_PATH, connect_ro
    from pipeline.tools.calibrate_server import _build_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("pipeline.tools.calibrate_server")

    # Soft-check only (2026-07-13, was a hard crash before): a long batch node
    # (e.g. asr.transcribe) can hold the writer lock for hours, which also
    # blocks connect_ro() -- that's an expected operating condition now, not
    # a reason to refuse to start. The server falls back to the offline JSON
    # snapshot + local decision buffer (pipeline/nodes/calibrate.py) for
    # reads/writes when the catalog is unreachable; see calibrate_server.py's
    # module docstring. Only a genuinely missing catalog file is worth
    # surfacing loudly here, and even that just delays to first-request time.
    try:
        connect_ro(CATALOG_PATH).close()
    except duckdb.IOException as exc:
        log.warning(
            f"catalog not reachable at startup ({exc}) -- serving from the offline "
            f"snapshot/local buffer if available; live data resumes once the writer is free"
        )

    handler = _build_app(args.batch)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    log.info(f"calibrate serve: listening on http://127.0.0.1:{args.port}/ (batch={args.batch or 'all'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_calibrate_export_snapshot(args: argparse.Namespace) -> int:
    import asyncio
    import logging
    from pathlib import Path

    from pipeline.nodes.calibrate import run_calibrate_export_snapshot

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_calibrate_export_snapshot(
        out_path=Path(args.out) if args.out else None,
        limit=args.limit, sample_batch=args.batch, source=args.source,
    ))
    print(f"\nDone: {result}")
    return 0


def cmd_calibrate_flush_pending(args: argparse.Namespace) -> int:
    import asyncio
    import logging
    from pathlib import Path

    from pipeline.nodes.calibrate import run_calibrate_flush_pending

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_calibrate_flush_pending(in_path=Path(args.input) if args.input else None))
    print(f"\nDone: {result}")
    return 0


def cmd_calibrate_prune_excluded(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.calibrate import run_calibrate_prune_excluded

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_calibrate_prune_excluded())
    print(f"\nDone: {result}")
    return 0


def cmd_calibrate_progress(args: argparse.Namespace) -> int:
    from pipeline.nodes.calibrate import run_calibrate_progress

    report = run_calibrate_progress()
    totals = report["totals"]
    by_kind = report["by_code_switch"]

    print(f"\nQA queue: {totals['total']} total, {totals['reviewed']} reviewed, "
          f"{totals['pending']} pending")
    print(f"  pure Cantonese:  {by_kind['pure']['total']:>6} total, "
          f"{by_kind['pure']['pending']:>6} pending")
    print(f"  code-switch:     {by_kind['code_switch']['total']:>6} total, "
          f"{by_kind['code_switch']['pending']:>6} pending")

    print(f"\n{'tier':<12}{'kind':<14}{'pending':>9}{'verified':>10}{'rejected':>10}"
          f"{'skipped':>9}{'flagged':>9}")
    for tier in sorted(report["breakdown"]):
        for kind in sorted(report["breakdown"][tier]):
            d = report["breakdown"][tier][kind]
            print(f"{tier:<12}{kind:<14}{d.get('pending', 0):>9}{d.get('verified', 0):>10}"
                  f"{d.get('rejected', 0):>10}{d.get('skipped', 0):>9}{d.get('flagged', 0):>9}")
    return 0


def cmd_calibrate_pause_qc_report(args: argparse.Namespace) -> int:
    from pipeline.nodes.calibrate import run_pause_qc_report

    report = run_pause_qc_report()
    print(f"\nPause QC (P4): {report['n_segments_reviewed']} segment(s) reviewed "
          f"({report['n_segments_skipped']} skipped), {report['n_events_reviewed']} event(s) judged")
    if not report["by_plan_verdict"]:
        print("  no events judged yet")
        return 0
    print(f"\n{'plan_verdict':<14}{'n':>6}{'match_rate':>13}{'position_ok_rate':>18}")
    for verdict in ("no_pause", "short", "long"):
        s = report["by_plan_verdict"].get(verdict)
        if not s:
            continue
        print(f"{verdict:<14}{s['n']:>6}{s['match_rate']:>13}{s['position_ok_rate']:>18}")
    return 0


def cmd_run_manifest_build(args: argparse.Namespace) -> int:
    import logging

    from pipeline.nodes.manifest import run_manifest_build

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_manifest_build(
        limit=args.limit, min_agreement=args.min_agreement, min_tier=args.min_tier,
        code_switch=args.code_switch, min_quality_tier=args.min_quality_tier,
    )
    summary = {k: v for k, v in result.items() if k != "entries"}
    print(f"\nDone: {summary}")
    return 0


def cmd_run_manifest_export(args: argparse.Namespace) -> int:
    import logging

    from pipeline.nodes.manifest import run_manifest_export

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_manifest_export(
        limit=args.limit, dry_run=args.dry_run, min_agreement=args.min_agreement, min_tier=args.min_tier,
        code_switch=args.code_switch, min_quality_tier=args.min_quality_tier,
    )
    summary = {k: v for k, v in result.items() if k != "entries"}
    print(f"\nDone: {summary}")
    return 0


def cmd_run_report_build(args: argparse.Namespace) -> int:
    import logging

    from pipeline.nodes.report import run_report_build

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_report_build(min_tier=args.min_tier)
    summary = {k: v for k, v in result.items() if k != "criteria"}
    print(f"\nDone: {summary}")
    return 0


def cmd_run_label_calibrate(args: argparse.Namespace) -> int:
    import logging

    from pipeline.nodes.label_calibrate import run_label_calibrate

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_label_calibrate()
    print(f"\nDone: {result}")
    return 0


def cmd_run_label_store(args: argparse.Namespace) -> int:
    import logging

    from pipeline.nodes.label_store import run_label_store

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_label_store()
    print(f"\nDone: {result}")
    return 0


def cmd_run_speaker_cluster(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.speaker import run_speaker_cluster

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None
    result = asyncio.run(run_speaker_cluster(
        threshold=args.threshold,
        sources=sources,
        limit=args.limit,
    ))
    print(f"\nDone: {result}")
    return 0


def cmd_run_embed_backfill(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.speaker import run_embed_backfill

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_embed_backfill(limit=args.limit))
    print(f"\nDone: {result}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="pipe")
    sub = parser.add_subparsers(dest="command", required=True)

    p_catalog = sub.add_parser("catalog", help="Catalog build/verify/rebuild")
    catalog_sub = p_catalog.add_subparsers(dest="catalog_command", required=True)

    p_build = catalog_sub.add_parser("build", help="Import legacy jsonl into the DuckDB catalog")
    p_build.add_argument("--dry-run", action="store_true")
    p_build.set_defaults(func=cmd_catalog_build)

    p_verify = catalog_sub.add_parser("verify", help="Run P0 gate checks against the catalog")
    p_verify.set_defaults(func=cmd_catalog_verify)

    p_rebuild = catalog_sub.add_parser("rebuild", help="Full catalog rebuild")
    p_rebuild.add_argument("--dry-run", action="store_true")
    p_rebuild.set_defaults(func=cmd_catalog_rebuild)

    p_golden = sub.add_parser("golden", help="Golden test-set build")
    golden_sub = p_golden.add_subparsers(dest="golden_command", required=True)
    p_golden_build = golden_sub.add_parser("build", help="Build stratified golden set + legacy snapshot")
    p_golden_build.set_defaults(func=cmd_golden_build)

    p_logs = sub.add_parser("logs", help="metadata/logs/ maintenance")
    logs_sub = p_logs.add_subparsers(dest="logs_command", required=True)
    p_logs_prune = logs_sub.add_parser("prune", help="T12: gzip old *.log, delete old *.log.gz (safe to run repeatedly / on a schedule)")
    p_logs_prune.add_argument("--gzip-after-days", type=float, default=7)
    p_logs_prune.add_argument("--delete-after-days", type=float, default=60)
    p_logs_prune.add_argument("--dry-run", action="store_true")
    p_logs_prune.set_defaults(func=cmd_logs_prune)

    p_chain = sub.add_parser("chain", help="T14: run the full ingest->tier DAG as ordered rounds, pairing independent nodes via run-many")
    chain_sub = p_chain.add_subparsers(dest="chain_command", required=True)
    p_chain_run = chain_sub.add_parser("run", help="run the chain (see pipeline/tools/chain_runner.py module docstring for the round design)")
    p_chain_run.add_argument("--only", default=None, help="comma-separated round numbers to run, e.g. 2,11")
    p_chain_run.add_argument("--skip", default=None, help="comma-separated round numbers to skip")
    p_chain_run.add_argument("--devices", default=None, help="forwarded as --devices to GPU rounds (diarize/asr.transcribe/lang_screen.auto)")
    p_chain_run.add_argument("--dry-run", action="store_true")
    p_chain_run.set_defaults(func=cmd_chain_run)
    p_chain_stream = chain_sub.add_parser("stream", help="T14 lever 4: drain downstream CPU nodes on a poll interval while an upstream GPU node is still running (see pipeline/tools/stream_drain.py)")
    p_chain_stream.add_argument("--upstream", required=True, help="the long-running GPU node, e.g. asr.transcribe")
    p_chain_stream.add_argument("--upstream-args", default=None, help="quoted extra argv for the upstream node, e.g. '--batch 64 --devices cuda:0,cuda:1'")
    p_chain_stream.add_argument("--downstream", action="append", required=True, help="downstream node to drain on each poll; repeat for multiple (run together via run-many)")
    p_chain_stream.add_argument("--downstream-args", action="append", default=[], help="'node=quoted argv' pairs, repeatable")
    p_chain_stream.add_argument("--poll-interval", type=float, default=300)
    p_chain_stream.set_defaults(func=cmd_chain_stream)

    p_calibrate = sub.add_parser("calibrate", help="Human text-verification calibration (see calibrate.sample DAG node for queuing)")
    calibrate_sub = p_calibrate.add_subparsers(dest="calibrate_command", required=True)
    p_calibrate_serve = calibrate_sub.add_parser("serve", help="Start the local browser review UI (blocks -- Ctrl-C to stop)")
    p_calibrate_serve.add_argument("--port", type=int, default=8420)
    p_calibrate_serve.add_argument("--batch", default=None, help="restrict review to one calibrate.sample run_id")
    p_calibrate_serve.set_defaults(func=cmd_calibrate_serve)
    p_calibrate_export = calibrate_sub.add_parser(
        "export-snapshot",
        help="dump the pending review queue to a JSON file (2026-07-13) so 'serve' can fall back "
             "to it when the catalog is unreachable (e.g. a long batch node holds the writer lock)",
    )
    p_calibrate_export.add_argument("--out", default=None, help="default: metadata/calibration_offline_queue.json")
    p_calibrate_export.add_argument("--limit", type=int, default=None)
    p_calibrate_export.add_argument("--batch", default=None, help="scope to one calibrate.sample run_id")
    p_calibrate_export.add_argument("--source", default=None)
    p_calibrate_export.set_defaults(func=cmd_calibrate_export_snapshot)
    p_calibrate_flush = calibrate_sub.add_parser(
        "flush-pending",
        help="replay review decisions buffered locally while the catalog was busy (2026-07-13) "
             "into record_decision() -- run anytime the writer lock is free",
    )
    p_calibrate_flush.add_argument("--input", default=None, help="default: metadata/calibration_pending_decisions.jsonl")
    p_calibrate_flush.set_defaults(func=cmd_calibrate_flush_pending)
    p_calibrate_progress = calibrate_sub.add_parser(
        "progress",
        help="T1 QA-backlog tracker (2026-07-17): review queue broken down by tier x "
             "code-switch status x decision, read-only",
    )
    p_calibrate_progress.set_defaults(func=cmd_calibrate_progress)
    p_calibrate_prune = calibrate_sub.add_parser(
        "prune-excluded",
        help="delete 'pending' calibration_review rows whose segment has since been "
             "auto-excluded by filter.decide (tiers.tier='excluded') -- flushes the "
             "offline decision buffer first (2026-07-19, T25)",
    )
    p_calibrate_prune.set_defaults(func=cmd_calibrate_prune_excluded)
    p_calibrate_pause_qc_report = calibrate_sub.add_parser(
        "pause-qc-report",
        help="P4 (PAUSE_TOKEN_PUNCTUATION_PLAN.md): aggregate perceived-vs-plan match rate "
             "+ position_ok rate from the 'Pause QC' mode in 'pipe calibrate serve' -- the "
             "QC gate's pass/fail signal, read-only",
    )
    p_calibrate_pause_qc_report.set_defaults(func=cmd_calibrate_pause_qc_report)

    p_run = sub.add_parser("run", help="Run a DAG node via the orchestrator")
    run_sub = p_run.add_subparsers(dest="run_command", required=True)
    from pipeline.nodes.ingest_download import SOURCE_FILES as _INGEST_SOURCE_FILES
    p_run_download = run_sub.add_parser("ingest.download", help="download audio for all registered sources (rthk/youtube/podcast/hktv/radio/audiobook/gov/drama/edu), native container, zero transcode (2026-07-04 policy)")
    p_run_download.add_argument("--source", default="all", choices=list(_INGEST_SOURCE_FILES) + ["all"])
    p_run_download.add_argument("--dry-run", action="store_true")
    p_run_download.add_argument("--limit", type=int, default=None)
    p_run_download.add_argument("--cookies-from-browser", default=None,
                                 help="yt-dlp --cookies-from-browser value, e.g. 'chrome' or 'chrome:Profile 1' (needed since YouTube's 2026-07-16 bot-check)")
    p_run_download.set_defaults(func=cmd_run_ingest_download)
    p_run_commit = run_sub.add_parser("ingest.commit", help="land ingest.download's JSON-staged rows into raw_files (only ingest.* step that opens DuckDB); run whenever the writer lock is free")
    p_run_commit.set_defaults(func=cmd_run_ingest_commit)
    p_run_probe = run_sub.add_parser("ingest.probe", help="P2: ffprobe metadata + L/R correlation per raw file")
    p_run_probe.add_argument("--workers", type=int, default=8)
    p_run_probe.add_argument("--batch", type=int, default=200)
    p_run_probe.add_argument("--limit", type=int, default=None)
    p_run_probe.set_defaults(func=cmd_run_ingest_probe)
    p_run_prosody = run_sub.add_parser("label.prosody", help="P2: rate/pitch/pause raw detector (CPU)")
    p_run_prosody.add_argument("--workers", type=int, default=4, help="number of CPU worker processes")
    p_run_prosody.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    p_run_prosody.add_argument("--batch", type=int, default=8)
    p_run_prosody.add_argument("--limit", type=int, default=None)
    p_run_prosody.set_defaults(func=cmd_run_label_prosody)
    p_run_lscreen = run_sub.add_parser("lang_screen.auto", help="raw-level Mandarin-vs-Cantonese pre-filter, runs BEFORE segment.diarize")
    p_run_lscreen.add_argument("--devices", default="cuda:0,cuda:1",
                                help="comma-separated device list, one worker per device")
    p_run_lscreen.add_argument("--gpu-policy", default="cap", choices=["yield", "cap", "exempt"])
    p_run_lscreen.add_argument("--batch", type=int, default=16)
    p_run_lscreen.add_argument("--mem-fraction", type=float, default=0.25,
                                help="mms-lid-126 alone needs more headroom than label.music's "
                                     "single-model 0.15 default (measured OOM at 0.15, 2026-07-04)")
    p_run_lscreen.add_argument("--limit", type=int, default=None)
    p_run_lscreen.set_defaults(func=cmd_run_lang_screen_auto)
    p_run_suite = run_sub.add_parser("label.suite", help="P2: decode-once lang+overlap+music fan-out")
    p_run_suite.add_argument("--devices", default="cuda:0,cuda:1",
                              help="comma-separated device list, one worker per device")
    p_run_suite.add_argument("--gpu-policy", default="cap", choices=["yield", "cap", "exempt"])
    p_run_suite.add_argument("--batch", type=int, default=16)
    p_run_suite.add_argument("--mem-fraction", type=float, default=0.35,
                              help="hosts 3 models (mms-lid+pyannote+PANNs) in one process — "
                                   "needs more headroom than label.music's single-model 0.15")
    p_run_suite.add_argument("--limit", type=int, default=None)
    p_run_suite.set_defaults(func=cmd_run_label_suite)
    p_run_align = run_sub.add_parser("align.chars", help="P0 (docs/PAUSE_TOKEN_PUNCTUATION_PLAN.md): char-level forced alignment via Qwen3-ForcedAligner, gold+auto_gold scope only")
    p_run_align.add_argument("--devices", default="cuda:0,cuda:1",
                              help="comma-separated device list, one worker per device")
    p_run_align.add_argument("--gpu-policy", default="cap", choices=["yield", "cap", "exempt"])
    p_run_align.add_argument("--batch", type=int, default=64,
                              help="real batched forward pass (2026-07-21) — see "
                                   "run_align_chars docstring for the VRAM measurement "
                                   "behind this default")
    p_run_align.add_argument("--mem-fraction", type=float, default=None)
    p_run_align.add_argument("--limit", type=int, default=None,
                              help="use --limit 200 for the pilot run + manual spot-check "
                                   "before a full gold+auto_gold pass (PAUSE_TOKEN_PUNCTUATION_PLAN.md P0)")
    p_run_align.add_argument("--prefetch", type=int, default=2,
                              help="tasks kept in flight per worker so CPU decode of batch N+1 "
                                   "overlaps GPU forward of batch N (1 = old sequential behaviour)")
    p_run_align.add_argument("--io-workers", type=int, default=16,
                              help="decode+resample thread-pool size inside each worker subprocess")
    p_run_align.set_defaults(func=cmd_run_align_chars)
    p_run_diarize = run_sub.add_parser("segment.diarize", help="P3: pyannote speaker diarization (reuse-first, GPU fallback)")
    p_run_diarize.add_argument("--devices", default="cuda:0,cuda:1",
                                help="comma-separated device list, one worker per device (only spawned for cache misses)")
    p_run_diarize.add_argument("--gpu-policy", default="cap", choices=["yield", "cap", "exempt"])
    p_run_diarize.add_argument("--batch", type=int, default=32)
    p_run_diarize.add_argument("--mem-fraction", type=float, default=0.5)
    p_run_diarize.add_argument("--limit", type=int, default=None,
                                help="process only the first N discovered raw files (testing)")
    p_run_diarize.set_defaults(func=cmd_run_segment_diarize)
    p_run_vadcut = run_sub.add_parser("segment.vad_cut", help="P3: Silero VAD within turns -> 48kHz FLAC segments (CPU+IO, in-supervisor; FLAC since 2026-07-05 P5-A)")
    p_run_vadcut.add_argument("--threads", type=int, default=None, help="thread-pool size (default: min(16, 2*ncpu))")
    p_run_vadcut.add_argument("--limit", type=int, default=None,
                               help="process only the first N discovered raw files (testing)")
    p_run_vadcut.set_defaults(func=cmd_run_segment_vad_cut)
    p_run_pregate = run_sub.add_parser("pregate.snr", help="P3: fast SNR+DNSMOS pre-gate before ASR (CPU, pipeline-cut segments only)")
    p_run_pregate.add_argument("--min-snr", type=float, default=25.0)
    p_run_pregate.add_argument("--min-dnsmos", type=float, default=3.0, help="set 0 to skip DNSMOS")
    p_run_pregate.add_argument("--threads", type=int, default=None, help="thread-pool size (default: min(16, 2*ncpu))")
    p_run_pregate.add_argument("--batch", type=int, default=500)
    p_run_pregate.add_argument("--limit", type=int, default=None)
    p_run_pregate.set_defaults(func=cmd_run_pregate_snr)
    p_run_rawflac = run_sub.add_parser("raw.flac", help="P5-B: transcode raw WAV backlog to lossless FLAC (CPU+IO); --delete-verified reclaims space for already-verified transcodes")
    p_run_rawflac.add_argument("--workers", type=int, default=8)
    p_run_rawflac.add_argument("--batch", type=int, default=50, help="items per catalog-commit batch")
    p_run_rawflac.add_argument("--batch-gb", type=float, default=None,
                                help="stop after ~this many GiB of source .wav this invocation")
    p_run_rawflac.add_argument("--limit", type=int, default=None)
    p_run_rawflac.add_argument("--delete-verified", action="store_true",
                                help="delete original .wav for already-verified transcodes instead of transcoding")
    p_run_rawflac.set_defaults(func=cmd_run_raw_flac)
    p_run_recover = run_sub.add_parser("recover.orphans", help="one-time: classify legacy VAD-cut WAVs missing from the catalog, backfill promising ones, queue the rest as pending_delete")
    p_run_recover.add_argument("--limit", type=int, default=None)
    p_run_recover.set_defaults(func=cmd_run_recover_orphans)
    p_run_reingest = run_sub.add_parser("recover.reingest_pending", help="one-time (2026-07-12): re-admit orphan_segments pending_delete rows into segments for a fresh pass through the current 3-model ASR pipeline, instead of deleting")
    p_run_reingest.add_argument("--limit", type=int, default=None)
    p_run_reingest.set_defaults(func=cmd_run_reingest_pending)
    p_run_rebalance = run_sub.add_parser("rebalance.segments", help="P5-C: spread segments across the 3-way Drive2/3/4 shard (CPU+IO); --delete-verified reclaims space for already-verified migrations")
    p_run_rebalance.add_argument("--workers", type=int, default=8)
    p_run_rebalance.add_argument("--batch", type=int, default=200, help="items per catalog-commit batch")
    p_run_rebalance.add_argument("--batch-gb", type=float, default=None,
                                  help="stop after ~this many GiB of source file this invocation")
    p_run_rebalance.add_argument("--limit", type=int, default=None)
    p_run_rebalance.add_argument("--delete-verified", action="store_true",
                                  help="delete original file for already-verified migrations instead of copying")
    p_run_rebalance.set_defaults(func=cmd_run_rebalance_segments)
    p_run_asr = run_sub.add_parser(
        "asr.transcribe",
        help="P3: multi-model ASR across GPUs (active: qwen3_asr, sense_voice; canto_ft/whisper_v3 retired)",
    )
    p_run_asr.add_argument(
        "--models", default="qwen3_asr,sense_voice",
        help=(
            "comma-separated model keys, paired positionally with --devices. "
            "Active keys: qwen3_asr, sense_voice. canto_ft and whisper_v3 are retired "
            "(ASR_MODELS[...]['enabled'] = False, see pipeline/nodes/asr.py) and refused "
            "by the guard rail below. "
            "Example: --models sense_voice,sense_voice --devices cuda:0,cuda:1 "
            "(splits sense_voice across both GPUs round-robin)."
        ),
    )
    p_run_asr.add_argument("--devices", default="cuda:0,cuda:1",
                            help="comma-separated device list, one worker per (model,device) pair")
    p_run_asr.add_argument("--gpu-policy", default="cap", choices=["yield", "cap", "exempt"])
    p_run_asr.add_argument("--batch", type=int, default=8)
    p_run_asr.add_argument("--mem-fraction", type=float, default=None)
    p_run_asr.add_argument("--limit", type=int, default=None)
    p_run_asr.add_argument("--prefetch", type=int, default=2,
                            help="tasks kept in flight per worker so CPU decode of batch N+1 "
                                 "overlaps GPU forward of batch N (1 = old sequential behaviour)")
    p_run_asr.add_argument("--io-workers", type=int, default=16,
                            help="decode+resample thread-pool size inside each worker subprocess")
    p_run_asr.set_defaults(func=cmd_run_asr_transcribe)
    p_run_agree = run_sub.add_parser("asr.agreement", help="P3: cross-model char-overlap agreement (CPU)")
    p_run_agree.add_argument("--batch", type=int, default=2000)
    p_run_agree.add_argument("--limit", type=int, default=None)
    p_run_agree.set_defaults(func=cmd_run_asr_agreement)
    p_run_ftext = run_sub.add_parser("filter.text", help="P3: sample_rate/duration hard gates + CJK-length/eng/mandarin text gates (CPU, no audio)")
    p_run_ftext.add_argument("--batch", type=int, default=5000)
    p_run_ftext.add_argument("--limit", type=int, default=None)
    p_run_ftext.set_defaults(func=cmd_run_filter_text)
    p_run_facoustic = run_sub.add_parser("filter.acoustic", help="P3: SNR + DNSMOS (CPU worker pool, requires filter.text pass)")
    p_run_facoustic.add_argument("--workers", type=int, default=4, help="number of CPU worker processes")
    p_run_facoustic.add_argument("--threads", type=int, default=4, help="onnxruntime intra_op_num_threads per worker")
    p_run_facoustic.add_argument("--batch", type=int, default=8)
    p_run_facoustic.add_argument("--limit", type=int, default=None)
    p_run_facoustic.add_argument("--gpu", type=str, default=None,
                                  help="comma-separated CUDA device ids to round-robin DNSMOS onto (default: CPU-only)")
    p_run_facoustic.set_defaults(func=cmd_run_filter_acoustic)
    p_run_fdecide = run_sub.add_parser("filter.decide", help="P3: merge filters_text + filters_acoustic into filters.pass")
    p_run_fdecide.add_argument("--batch", type=int, default=5000)
    p_run_fdecide.add_argument("--limit", type=int, default=None)
    p_run_fdecide.set_defaults(func=cmd_run_filter_decide)
    p_run_g2p = run_sub.add_parser("g2p", help="P3: canto-hk-g2p Cantonese text -> Jyutping (CPU, in-supervisor)")
    p_run_g2p.add_argument("--batch", type=int, default=2000)
    p_run_g2p.add_argument("--limit", type=int, default=None)
    p_run_g2p.set_defaults(func=cmd_run_g2p)
    p_run_pause_plan = run_sub.add_parser("pause.plan", help="P2 (PAUSE_TOKEN_PUNCTUATION_PLAN.md): punctuation-anchored pause plan from alignments.chars (CPU, in-supervisor)")
    p_run_pause_plan.add_argument("--batch", type=int, default=5000)
    p_run_pause_plan.add_argument("--limit", type=int, default=None)
    p_run_pause_plan.set_defaults(func=cmd_run_pause_plan)
    p_run_music = run_sub.add_parser("label.music", help="P1 pilot: PANNs music-family tagging")
    p_run_music.add_argument("--devices", default="cuda:0,cuda:1",
                              help="comma-separated device list, one worker per device")
    p_run_music.add_argument("--gpu-policy", default="cap", choices=["yield", "cap", "exempt"])
    p_run_music.add_argument("--batch", type=int, default=16)
    p_run_music.add_argument("--mem-fraction", type=float, default=0.15)
    p_run_music.add_argument("--limit", type=int, default=None,
                              help="process only the first N discovered segments (testing)")
    p_run_music.set_defaults(func=cmd_run_label_music)
    p_run_spk_embed = run_sub.add_parser("speaker.embed", help="P3: ECAPA-TDNN d-vector embedding (reuse-first, GPU fallback)")
    p_run_spk_embed.add_argument("--devices", default="cuda:0,cuda:1",
                                  help="comma-separated device list, one worker per device (only spawned for cache misses)")
    p_run_spk_embed.add_argument("--gpu-policy", default="cap", choices=["yield", "cap", "exempt"])
    p_run_spk_embed.add_argument("--batch", type=int, default=5000)
    p_run_spk_embed.add_argument("--mem-fraction", type=float, default=0.15)
    p_run_spk_embed.add_argument("--limit", type=int, default=None,
                                  help="process only the first N discovered segments (testing)")
    p_run_spk_embed.add_argument("--verify-existing", action="store_true",
                                  help="also stat every existing embedding_ref file on disk and "
                                       "re-queue rows whose sidecar is missing (e.g. orphaned by "
                                       "the filtered/ tree retirement) -- slower, opt-in repair pass")
    p_run_spk_embed.set_defaults(func=cmd_run_speaker_embed)
    p_run_spk_cluster = run_sub.add_parser("speaker.cluster", help="P3: cross-file speaker clustering (CPU, whole-source recompute)")
    p_run_spk_cluster.add_argument("--threshold", type=float, default=0.25)
    p_run_spk_cluster.add_argument("--sources", default=None,
                                    help="comma-separated source allow-list (default: all sources)")
    p_run_spk_cluster.add_argument("--limit", type=int, default=None,
                                    help="cap segments loaded per source (testing)")
    p_run_spk_cluster.set_defaults(func=cmd_run_speaker_cluster)
    p_run_embed_backfill = run_sub.add_parser("embed.backfill", help="one-time (2026-07-12): migrate existing .embed.npy sidecar contents into speaker_embeddings.embedding (I/O optimization Phase 3)")
    p_run_embed_backfill.add_argument("--limit", type=int, default=None)
    p_run_embed_backfill.set_defaults(func=cmd_run_embed_backfill)
    p_run_tier = run_sub.add_parser("tier.assign", help="P4: verification-confidence tier (gold/auto_gold/silver/bronze/excluded), CPU in-supervisor")
    p_run_tier.add_argument("--batch", type=int, default=5000)
    p_run_tier.add_argument("--limit", type=int, default=None)
    p_run_tier.set_defaults(func=cmd_run_tier_assign)
    p_run_qtier = run_sub.add_parser("quality_tier.assign", help="T13: A/B acoustic-cleanliness axis (pretrain/clean) for the gold+auto_gold scope, CPU in-supervisor -- SEPARATE from tier.assign, see pipeline/nodes/quality_tier.py")
    p_run_qtier.add_argument("--batch", type=int, default=5000)
    p_run_qtier.add_argument("--limit", type=int, default=None)
    p_run_qtier.set_defaults(func=cmd_run_quality_tier_assign)
    p_run_calib_sample = run_sub.add_parser("calibrate.sample", help="P4: queue a random sample of filter-passing segments for human text-verification review (see 'pipe calibrate serve')")
    p_run_calib_sample.add_argument("--n", type=int, default=300, help="sample size to queue")
    p_run_calib_sample.add_argument("--tier", default=None,
                                     help="scope the sample to one tiers.tier value, e.g. 'auto_gold' "
                                          "(QA the statistical-confidence tier specifically)")
    p_run_calib_sample.add_argument("--min-agreement", type=float, default=None,
                                     help="scope the sample to asr_agreement.agreement >= this value "
                                          "(QA a specific --min-agreement manifest.export cut)")
    p_run_calib_sample.add_argument("--code-switch", default=None, choices=["only", "exclude"],
                                     help="'only' = filters.english_ratio > 0 (code-switched segments "
                                          "only -- pair with a larger --n, e.g. "
                                          "recommended_sample_n(..., code_switch=True), for the intended "
                                          "10x oversampled QA batch), 'exclude' = english_ratio = 0; see "
                                          "pending_task.md T18")
    p_run_calib_sample.add_argument("--order", default="random", choices=["random", "agreement_asc"],
                                     help="which segments WITHIN the scoped tier/min-agreement/code-switch "
                                          "population get picked -- 'random' (default) reproduces the "
                                          "original unbiased sample, 'agreement_asc' concentrates the batch "
                                          "on the lowest-agreement (highest-risk) segments in that population "
                                          "instead of spreading uniformly across it; see pending_task.md T21")
    p_run_calib_sample.set_defaults(func=cmd_run_calibrate_sample)
    p_run_mbuild = run_sub.add_parser("manifest.build", help="P4: build manifest entries from the catalog (in-memory, no file write)")
    p_run_mbuild.add_argument("--limit", type=int, default=None)
    p_run_mbuild.add_argument("--min-agreement", type=float, default=None,
                               help="only include entries with asr_agreement.agreement >= this value "
                                    "(see docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md)")
    p_run_mbuild.add_argument("--min-tier", default=None, choices=["gold", "auto_gold", "silver", "bronze"],
                               help="only include entries at or above this tiers.tier value "
                                    "(e.g. 'auto_gold' includes gold+auto_gold) -- see pipeline/nodes/tier.py")
    p_run_mbuild.add_argument("--code-switch", default=None, choices=["only", "exclude"],
                               help="'only' = filters.english_ratio > 0 (code-switched segments only), "
                                    "'exclude' = english_ratio = 0 (pure Cantonese only); omit for no filter "
                                    "-- see pending_task.md T18")
    p_run_mbuild.add_argument("--min-quality-tier", default=None, choices=["A", "B"],
                               help="only include entries at or above this quality_tiers.quality_tier "
                                    "value (SEPARATE axis from --min-tier -- see pipeline/nodes/quality_tier.py); "
                                    "only meaningful combined with --min-tier gold/auto_gold (or omitted), "
                                    "since quality_tiers only covers that scope -- see pending_task.md T13")
    p_run_mbuild.set_defaults(func=cmd_run_manifest_build)
    p_run_mexport = run_sub.add_parser("manifest.export", help="P4: build + write metadata/manifest.jsonl + train.jsonl + val.jsonl")
    p_run_mexport.add_argument("--limit", type=int, default=None)
    p_run_mexport.add_argument("--dry-run", action="store_true")
    p_run_mexport.add_argument("--min-agreement", type=float, default=None,
                                help="write a smaller high-confidence cut to manifest_<tag>.jsonl / "
                                     "train_<tag>.jsonl / val_<tag>.jsonl instead of the default "
                                     "files (never overwrites them) -- see docs/FINDINGS_ASR_AGREEMENT_THRESHOLDS.md "
                                     "for suggested cuts per target dataset size")
    p_run_mexport.add_argument("--min-tier", default=None, choices=["gold", "auto_gold", "silver", "bronze"],
                                help="write a cut containing only entries at or above this tiers.tier value "
                                     "(e.g. 'auto_gold' includes gold+auto_gold) to manifest_tier_<tier>.jsonl "
                                     "etc. -- combinable with --min-agreement")
    p_run_mexport.add_argument("--code-switch", default=None, choices=["only", "exclude"],
                                help="write a cut to manifest_codeswitch_<mode>.jsonl etc.: 'only' = "
                                     "filters.english_ratio > 0 (code-switched segments only, e.g. for a "
                                     "dedicated QA/eval subset), 'exclude' = english_ratio = 0 (pure "
                                     "Cantonese only) -- combinable with --min-tier/--min-agreement; see "
                                     "pending_task.md T18")
    p_run_mexport.add_argument("--min-quality-tier", default=None, choices=["A", "B"],
                                help="write a cut to manifest_qualityA.jsonl / manifest_qualityB.jsonl etc: "
                                     "SEPARATE axis from --min-tier (see pipeline/nodes/quality_tier.py) -- "
                                     "'B' for the strict clean-fine-tune subset, 'A' for the full "
                                     "pretrain scope; only meaningful combined with --min-tier gold/auto_gold "
                                     "(or omitted) -- see pending_task.md T13")
    p_run_mexport.set_defaults(func=cmd_run_manifest_export)
    p_run_report = run_sub.add_parser("report.build", help="P4: dataset-statistics + acceptance-criteria report, read live from the catalog -> metadata/DATASET_REPORT.md")
    p_run_report.add_argument("--min-tier", default=None, choices=["gold", "auto_gold", "silver", "bronze"],
                               help="scope the report to entries at or above this tiers.tier value "
                                    "(e.g. 'gold' checks the strictly human-verified subset only) -- "
                                    "see pipeline/nodes/manifest.py's min_tier convention")
    p_run_report.set_defaults(func=cmd_run_report_build)
    p_run_lcalib = run_sub.add_parser("label.calibrate", help="P4: compute rate/pitch calibration constants -> metadata/labels/calibration.json")
    p_run_lcalib.set_defaults(func=cmd_run_label_calibrate)
    p_run_lstore = run_sub.add_parser("label.store", help="P4: join+bucket label tables -> metadata/labels.jsonl (requires label.calibrate first)")
    p_run_lstore.set_defaults(func=cmd_run_label_store)

    p_run_many = sub.add_parser(
        "run-many",
        help="Run multiple DAG nodes concurrently in one process, sharing one "
             "DuckDB connection (cursor per node) — bypasses the per-process "
             "single-writer lock. Nodes: " + ", ".join(sorted(RUN_MANY_ADAPTERS)) +
             ". Usage: pipe run-many <node> [args...] -- <node> [args...] -- ...",
    )
    p_run_many.add_argument("groups", nargs=argparse.REMAINDER)

    def cmd_run_many(args: argparse.Namespace) -> int:
        import asyncio
        import logging

        from pipeline.catalog.catalog import connect

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

        groups = split_run_many_groups(args.groups)
        if len(groups) < 2:
            print(
                "run-many needs at least 2 node groups separated by '--', e.g.:\n"
                "  pipe run-many segment.diarize --devices cuda:0 -- "
                "filter.acoustic --workers 8 --threads 4"
            )
            return 1

        node_specs: list[tuple[str, argparse.Namespace]] = []
        for group in groups:
            node_name, *node_argv = group
            if node_name not in RUN_MANY_ADAPTERS:
                print(f"run-many: node '{node_name}' not yet supported for concurrent "
                      f"execution (supported: {sorted(RUN_MANY_ADAPTERS)}); "
                      f"see docs/ORCHESTRATOR_PLAN.md to extend it")
                return 1
            if node_name not in run_sub.choices:
                print(f"run-many: unknown node '{node_name}'")
                return 1
            node_args = run_sub.choices[node_name].parse_args(node_argv)
            node_specs.append((node_name, node_args))

        async def _run_all():
            conn = connect()
            try:
                coros = [
                    RUN_MANY_ADAPTERS[name](node_args, conn.cursor())
                    for name, node_args in node_specs
                ]
                return await asyncio.gather(*coros)
            finally:
                conn.close()

        results = asyncio.run(_run_all())
        for (name, _), result in zip(node_specs, results):
            print(f"\n{name}: {result}")
        return 0

    p_run_many.set_defaults(func=cmd_run_many)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
