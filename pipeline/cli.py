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

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
