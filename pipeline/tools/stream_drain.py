"""
pipeline/tools/stream_drain.py
T14 lever (4): stream downstream CPU-only nodes while a long upstream GPU
stage (typically asr.transcribe) is still running, instead of waiting for the
entire GPU stage to finish before starting the CPU chain.

Every node's discovery is an idempotent anti-join (CLAUDE.md convention), so
re-invoking a downstream node while its upstream is still landing new rows is
always safe: each poll just picks up whatever's newly available and no-ops on
the rest. This is what makes a poll loop viable without any special
coordination between the upstream and downstream processes -- they only ever
communicate through the catalog.

Mechanism:
  1. Launch the upstream node as a background subprocess (`pipe run
     <upstream> ...`), non-blocking.
  2. While that process is alive, sleep `poll_interval_s` and then run the
     downstream node(s) to drain whatever landed so far -- solo via `pipe
     run <node>` if there's one downstream node, or `pipe run-many <n1> --
     <n2> ...` if there's more than one (mirrors chain_runner.py's Round
     convention for run-many-eligible pairs).
  3. Once the upstream process exits, do exactly one more drain pass to
     catch anything that landed between the last poll and process exit, then
     stop.

This directly targets the idle-resource window CLAUDE.md/pending_task.md
documented: GPU sits ~100% idle during every CPU-only stage and CPU cores
sit ~90% idle during every GPU-only stage under a strict waterfall. Draining
`asr.agreement`/`filter.text`/`g2p` etc. while `asr.transcribe` is still
running on the GPU overlaps that CPU work into the GPU stage's wall-clock
instead of tacking it on afterward.

Usage:
    pipe chain stream --upstream asr.transcribe --upstream-args "--batch 64" \\
        --downstream asr.agreement --downstream g2p \\
        --poll-interval 300

Each poll's downstream result (processed/errors counts) is logged with a
timestamp so a long stream run is auditable after the fact -- see
metadata/logs/stream_drain_<UTC ts>.log.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import LOGS_DIR, REPO_ROOT

DEFAULT_POLL_INTERVAL_S = 300


def _launch_upstream(node: str, extra_args: list[str]) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "pipeline.cli", "run", node, *extra_args]
    return subprocess.Popen(cmd, cwd=REPO_ROOT)


def _drain_downstream(nodes: list[str], extra_args: dict[str, list[str]]) -> subprocess.CompletedProcess:
    if len(nodes) == 1:
        cmd = [sys.executable, "-m", "pipeline.cli", "run", nodes[0], *extra_args.get(nodes[0], [])]
    else:
        cmd = [sys.executable, "-m", "pipeline.cli", "run-many"]
        for i, node in enumerate(nodes):
            if i > 0:
                cmd.append("--")
            cmd.append(node)
            cmd += extra_args.get(node, [])
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)


def run_stream(
    *,
    upstream: str,
    upstream_args: list[str] | None = None,
    downstream: list[str],
    downstream_args: dict[str, list[str]] | None = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    log_path: Path | None = None,
    sleep_fn=time.sleep,
) -> dict:
    """sleep_fn is injectable for tests -- avoids real wall-clock sleeps."""
    upstream_args = upstream_args or []
    downstream_args = downstream_args or {}

    log_file = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"{stamp} {msg}"
        print(line)
        if log_file:
            log_file.write(line + "\n")
            log_file.flush()

    polls: list[dict] = []
    try:
        log(f"Launching upstream: {upstream} {' '.join(upstream_args)}")
        proc = _launch_upstream(upstream, upstream_args)

        poll_n = 0
        while proc.poll() is None:
            sleep_fn(poll_interval_s)
            if proc.poll() is not None:
                break  # exited during the sleep -- let the final drain below handle it
            poll_n += 1
            log(f"Poll {poll_n}: draining {downstream} (upstream still running)")
            result = _drain_downstream(downstream, downstream_args)
            polls.append({"poll": poll_n, "returncode": result.returncode, "stdout": result.stdout})
            log(f"Poll {poll_n} result: rc={result.returncode}")

        upstream_returncode = proc.wait()
        log(f"Upstream {upstream} exited rc={upstream_returncode} -- final drain")
        final = _drain_downstream(downstream, downstream_args)
        log(f"Final drain result: rc={final.returncode}")

        return {
            "upstream": upstream,
            "upstream_returncode": upstream_returncode,
            "polls": polls,
            "final_drain": {"returncode": final.returncode, "stdout": final.stdout},
        }
    finally:
        if log_file:
            log_file.close()


def _parse_kv_args(raw: str | None) -> list[str]:
    if not raw:
        return []
    return shlex.split(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", required=True, help="the long-running GPU node, e.g. asr.transcribe")
    parser.add_argument("--upstream-args", default=None, help="quoted extra argv for the upstream node, e.g. '--batch 64 --devices cuda:0,cuda:1'")
    parser.add_argument("--downstream", action="append", required=True, help="downstream node to drain on each poll; repeat for multiple (run together via run-many)")
    parser.add_argument("--downstream-args", action="append", default=[], help="'node=quoted argv' pairs, repeatable, e.g. 'filter.acoustic=--workers 8'")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S, help=f"seconds between drain polls while upstream runs (default {DEFAULT_POLL_INTERVAL_S})")
    args = parser.parse_args()

    downstream_args: dict[str, list[str]] = {}
    for pair in args.downstream_args:
        node, _, raw = pair.partition("=")
        downstream_args[node] = shlex.split(raw)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOGS_DIR / f"stream_drain_{stamp}.log"

    result = run_stream(
        upstream=args.upstream,
        upstream_args=_parse_kv_args(args.upstream_args),
        downstream=args.downstream,
        downstream_args=downstream_args,
        poll_interval_s=args.poll_interval,
        log_path=log_path,
    )
    print(f"\nStream done: upstream rc={result['upstream_returncode']}, "
          f"{len(result['polls'])} poll(s), final drain rc={result['final_drain']['returncode']}")
    print(f"Log: {log_path}")
    return 0 if result["upstream_returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
