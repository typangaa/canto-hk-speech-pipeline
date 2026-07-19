"""
pipeline/tools/chain_runner.py
T14 lever (3): a committed replacement for the ad-hoc `run_t7_chain.sh`-style
scripts that have been hand-written and thrown away every time the full
ingest->tier waterfall needed re-running. Those scripts always ran every node
as a strict sequential waterfall -- GPU 100% idle during every CPU-only stage,
CPU cores ~90% idle during every GPU-only stage -- even though `pipe run-many`
(all 23/23 node call sites `conn=`-injected, see docs/ORCHESTRATOR_PLAN.md)
already provides the mechanism to run independent nodes concurrently under one
shared DuckDB connection.

This script codifies the DAG as a sequence of ROUNDS. Each round is either a
single node ("solo") or a set of nodes proven/reasoned to touch disjoint
catalog tables ("run-many") -- run together via `pipe run-many`. Rounds run
strictly in order (round N+1 waits for round N to fully finish); within a
run-many round, member nodes execute concurrently.

Every node's discovery is an idempotent anti-join (CLAUDE.md convention), so
re-running this script (or any individual round) when there's nothing new to
do is always safe and fast -- it just no-ops. This is what makes "always run
the full chain" a reasonable default instead of requiring the caller to
figure out which rounds actually have work.

Round design (see docs/REARCHITECTURE_IMPLEMENTATION_PLAN.md / CLAUDE.md's
node table for the full table-level read/write map each round's grouping
was checked against):

  1. ingest.commit          solo   (raw_files -- must land before anything
                                     downstream can discover new raw ids)
  2. ingest.probe            \\
     lang_screen.auto        }-- run-many: both read raw_files ONLY,
                                     write disjoint tables (raw_probe /
                                     labels_lang). Live-validated 2026-07-17
                                     (T14): 4,086-row ingest.probe backlog +
                                     a full 3-source speaker.cluster recompute
                                     finished concurrently in 212s with zero
                                     stalling -- see the T14 pending_task.md
                                     entry for why this specific class of
                                     pairing used to stall before the
                                     upsert_rows() bulk-write fix (2026-07-16).
  3. segment.diarize         solo   (GPU; writes diarization_turns+raw_segments)
  4. segment.vad_cut         solo   (reads diarization_turns, writes segments)
  5. pregate.snr             \\
     label.suite              }-- run-many: pregate.snr is CPU-only (no
                                     --devices arg), label.suite is GPU-only
                                     (cuda:0,cuda:1) -- no device contention,
                                     unlike pairing two different GPU models
                                     on one device. Both read segments ONLY,
                                     write disjoint tables (pregate vs.
                                     labels_lang/labels_overlap/labels_music).
                                     T23 (pending_task.md, 2026-07-18): added
                                     because label.suite was previously absent
                                     from every round, so its output silently
                                     fell 15 days / 785k segments behind while
                                     every other stage kept advancing via this
                                     chain -- filter.decide's T20/T22
                                     audio-language gates (round 10) and
                                     quality_tier.assign (round 13) both
                                     depend on label.suite's output, so it
                                     must land before either.
  6. asr.transcribe          solo   (GPU; do NOT pair with speaker.embed on
                                     shared devices -- interleaving two
                                     different GPU models on one device was
                                     measured to fully starve one of them,
                                     2026-07-13, see DECISIONS.md. Solo until
                                     that's tested safe with a device split.)
  7. asr.agreement           solo   (reads asr_results, writes asr_agreement)
  8. filter.text             solo   (reads asr_agreement, writes filters_text
                                     -- sequential dependency on round 7 for
                                     the SAME newly-landed ids)
  9. filter.acoustic         solo   (reads filters_text pass=true only --
                                     sequential dependency on round 8)
 10. filter.decide           solo   (merges filters_text+filters_acoustic --
                                     sequential dependency on rounds 8+9;
                                     also reads labels_lang from round 5 for
                                     the T20/T22 audio-language gates)
 11. g2p                     \\
     tier.assign              }-- run-many: three-way, all read
     speaker.cluster         /       asr_agreement/speaker_embeddings (already
                                     landed by earlier rounds) and write
                                     disjoint tables (g2p / tiers / speakers).
                                     This is the direct T15-documented stall
                                     case's fix target (asr.transcribe was
                                     originally paired with speaker.cluster;
                                     g2p+tier.assign are safe stand-ins that
                                     exercise the same speaker.cluster
                                     large-upsert path this round).
 12. quality_tier.assign     solo   (depends on round 11's tier.assign output,
                                     and on round 5's labels_music/labels_overlap)

Usage:
    pipe chain run                       # full chain, all rounds
    pipe chain run --only 2,11           # just those rounds (comma-separated
                                          # round numbers, 1-indexed as above)
    pipe chain run --skip 3,4,5,6        # e.g. skip the GPU segmentation/ASR
                                          # rounds if nothing new was ingested
    pipe chain run --dry-run             # print the round plan, run nothing
    pipe chain run --devices cuda:0,cuda:1   # forwarded to every round that
                                          # takes --devices (diarize/asr/
                                          # lang_screen/label.suite)

Each round's stdout/stderr is teed to metadata/logs/chain_runner_<UTC ts>.log
in addition to the console, with clear round-boundary markers, so a resumed
or partial run is auditable after the fact.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import LOGS_DIR, REPO_ROOT


@dataclass
class Round:
    number: int
    name: str
    nodes: list[str]                       # node names, e.g. ["g2p", "tier.assign"]
    extra_args: dict[str, list[str]] = field(default_factory=dict)  # node -> argv

    @property
    def is_run_many(self) -> bool:
        return len(self.nodes) > 1


def build_rounds(*, devices: str | None) -> list[Round]:
    device_args = ["--devices", devices] if devices else []
    return [
        Round(1, "ingest.commit", ["ingest.commit"]),
        Round(2, "ingest.probe + lang_screen.auto", ["ingest.probe", "lang_screen.auto"],
              extra_args={"lang_screen.auto": device_args} if devices else {}),
        Round(3, "segment.diarize", ["segment.diarize"],
              extra_args={"segment.diarize": device_args} if devices else {}),
        Round(4, "segment.vad_cut", ["segment.vad_cut"]),
        Round(5, "pregate.snr + label.suite", ["pregate.snr", "label.suite"],
              extra_args={"label.suite": device_args} if devices else {}),
        Round(6, "asr.transcribe", ["asr.transcribe"],
              extra_args={"asr.transcribe": device_args} if devices else {}),
        Round(7, "asr.agreement", ["asr.agreement"]),
        Round(8, "filter.text", ["filter.text"]),
        Round(9, "filter.acoustic", ["filter.acoustic"]),
        Round(10, "filter.decide", ["filter.decide"]),
        Round(11, "g2p + tier.assign + speaker.cluster", ["g2p", "tier.assign", "speaker.cluster"]),
        Round(12, "quality_tier.assign", ["quality_tier.assign"]),
    ]


def _run_solo(node: str, extra_args: list[str], log) -> int:
    cmd = [sys.executable, "-m", "pipeline.cli", "run", node, *extra_args]
    log(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def _run_many(nodes: list[str], extra_args: dict[str, list[str]], log) -> int:
    cmd = [sys.executable, "-m", "pipeline.cli", "run-many"]
    for i, node in enumerate(nodes):
        if i > 0:
            cmd.append("--")
        cmd.append(node)
        cmd += extra_args.get(node, [])
    log(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def run_chain(
    *,
    only: set[int] | None = None,
    skip: set[int] | None = None,
    devices: str | None = None,
    dry_run: bool = False,
    log_path: Path | None = None,
) -> dict:
    rounds = build_rounds(devices=devices)
    if only:
        rounds = [r for r in rounds if r.number in only]
    if skip:
        rounds = [r for r in rounds if r.number not in skip]

    log_file = None
    if log_path and not dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"{stamp} {msg}"
        print(line)
        if log_file:
            log_file.write(line + "\n")
            log_file.flush()

    results: list[dict] = []
    try:
        for r in rounds:
            mode = "run-many" if r.is_run_many else "solo"
            log(f"=== Round {r.number}: {r.name} [{mode}] ===")
            if dry_run:
                results.append({"round": r.number, "name": r.name, "mode": mode, "dry_run": True})
                continue
            rc = _run_many(r.nodes, r.extra_args, log) if r.is_run_many else _run_solo(
                r.nodes[0], r.extra_args.get(r.nodes[0], []), log,
            )
            results.append({"round": r.number, "name": r.name, "mode": mode, "returncode": rc})
            if rc != 0:
                log(f"Round {r.number} ({r.name}) FAILED with exit {rc} -- stopping chain")
                break
    finally:
        if log_file:
            log_file.close()

    return {"rounds": results}


def _parse_round_set(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default=None, help="comma-separated round numbers to run, e.g. 2,11")
    parser.add_argument("--skip", default=None, help="comma-separated round numbers to skip")
    parser.add_argument("--devices", default=None, help="forwarded as --devices to GPU rounds (diarize/asr.transcribe/lang_screen.auto/label.suite)")
    parser.add_argument("--dry-run", action="store_true", help="print the round plan, run nothing")
    args = parser.parse_args()

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


if __name__ == "__main__":
    raise SystemExit(main())
