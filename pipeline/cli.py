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


def cmd_run_ingest_download(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.ingest_download import run_ingest_download

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_ingest_download(
        source=args.source, dry_run=args.dry_run, limit=args.limit,
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


def cmd_run_asr_transcribe(args: argparse.Namespace) -> int:
    import asyncio
    import logging

    from pipeline.nodes.asr import run_asr_transcribe

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model_keys = [m.strip() for m in args.models.split(",")]
    devices = [d.strip() for d in args.devices.split(",")]
    if len(model_keys) != len(devices):
        raise SystemExit(f"--models ({len(model_keys)}) and --devices ({len(devices)}) must have the same count")
    assignments = list(zip(model_keys, devices))
    result = asyncio.run(run_asr_transcribe(
        assignments,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
        mem_fraction=args.mem_fraction,
        limit=args.limit,
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
    result = asyncio.run(run_filter_acoustic(
        n_workers=args.workers,
        threads_per_worker=args.threads,
        batch_size=args.batch,
        limit=args.limit,
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
    assignments = list(zip(model_keys, devices))
    return await run_asr_transcribe(
        assignments,
        conn=conn,
        gpu_policy=args.gpu_policy,
        batch_size=args.batch,
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


async def _run_many_adapt_recover_orphans(args: argparse.Namespace, conn) -> dict:
    from pipeline.nodes.recover_orphans import run_recover_orphans
    return await run_recover_orphans(conn=conn, limit=args.limit)


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
# see docs/ORCHESTRATOR_PLAN.md for the full call-site inventory and the
# priority order for extending this incrementally.
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
    "filter.text": _run_many_adapt_filter_text,
    "filter.decide": _run_many_adapt_filter_decide,
    "segment.vad_cut": _run_many_adapt_segment_vad_cut,
    "pregate.snr": _run_many_adapt_pregate_snr,
    "g2p": _run_many_adapt_g2p,
    "ingest.probe": _run_many_adapt_ingest_probe,
    "label.suite": _run_many_adapt_label_suite,
    "label.prosody": _run_many_adapt_label_prosody,
    "recover.orphans": _run_many_adapt_recover_orphans,
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


def cmd_run_manifest_build(args: argparse.Namespace) -> int:
    import logging

    from pipeline.nodes.manifest import run_manifest_build

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_manifest_build(limit=args.limit)
    summary = {k: v for k, v in result.items() if k != "entries"}
    print(f"\nDone: {summary}")
    return 0


def cmd_run_manifest_export(args: argparse.Namespace) -> int:
    import logging

    from pipeline.nodes.manifest import run_manifest_export

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_manifest_export(limit=args.limit, dry_run=args.dry_run)
    summary = {k: v for k, v in result.items() if k != "entries"}
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

    p_run = sub.add_parser("run", help="Run a DAG node via the orchestrator")
    run_sub = p_run.add_subparsers(dest="run_command", required=True)
    p_run_download = run_sub.add_parser("ingest.download", help="download rthk/youtube/podcast audio, native container, zero transcode (2026-07-04 policy)")
    p_run_download.add_argument("--source", default="all", choices=["rthk", "youtube", "podcast", "all"])
    p_run_download.add_argument("--dry-run", action="store_true")
    p_run_download.add_argument("--limit", type=int, default=None)
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
        help="P3: multi-model ASR across GPUs (canto_ft, whisper_v3, qwen3_asr, sense_voice)",
    )
    p_run_asr.add_argument(
        "--models", default="canto_ft,whisper_v3",
        help=(
            "comma-separated model keys, paired positionally with --devices. "
            "Valid keys: canto_ft, whisper_v3, qwen3_asr, sense_voice. "
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
    p_run_facoustic.set_defaults(func=cmd_run_filter_acoustic)
    p_run_fdecide = run_sub.add_parser("filter.decide", help="P3: merge filters_text + filters_acoustic into filters.pass")
    p_run_fdecide.add_argument("--batch", type=int, default=5000)
    p_run_fdecide.add_argument("--limit", type=int, default=None)
    p_run_fdecide.set_defaults(func=cmd_run_filter_decide)
    p_run_g2p = run_sub.add_parser("g2p", help="P3: canto-hk-g2p Cantonese text -> Jyutping (CPU, in-supervisor)")
    p_run_g2p.add_argument("--batch", type=int, default=2000)
    p_run_g2p.add_argument("--limit", type=int, default=None)
    p_run_g2p.set_defaults(func=cmd_run_g2p)
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
    p_run_tier = run_sub.add_parser("tier.assign", help="P4: verification-confidence tier (gold/silver/excluded), CPU in-supervisor")
    p_run_tier.add_argument("--batch", type=int, default=5000)
    p_run_tier.add_argument("--limit", type=int, default=None)
    p_run_tier.set_defaults(func=cmd_run_tier_assign)
    p_run_mbuild = run_sub.add_parser("manifest.build", help="P4: build manifest entries from the catalog (in-memory, no file write)")
    p_run_mbuild.add_argument("--limit", type=int, default=None)
    p_run_mbuild.set_defaults(func=cmd_run_manifest_build)
    p_run_mexport = run_sub.add_parser("manifest.export", help="P4: build + write metadata/manifest.jsonl + train.jsonl + val.jsonl")
    p_run_mexport.add_argument("--limit", type=int, default=None)
    p_run_mexport.add_argument("--dry-run", action="store_true")
    p_run_mexport.set_defaults(func=cmd_run_manifest_export)
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
