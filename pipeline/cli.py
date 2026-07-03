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
    p_run_asr = run_sub.add_parser("asr.transcribe", help="P3: dual faster-whisper models split across GPUs")
    p_run_asr.add_argument("--models", default="canto_ft,whisper_v3",
                            help="comma-separated model keys, paired positionally with --devices")
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

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
