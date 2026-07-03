#!/usr/bin/env python3
"""
tests/golden/check_asr_parity.py
Similarity-tolerance golden-parity check for asr.transcribe (pipeline/nodes/asr.py).

Not a pytest test — loading two faster-whisper models needs a GPU and several
seconds of warmup, which doesn't belong in the regular `pytest tests/` unit-test
loop. Run this manually (or as a deliberate gate step) after touching asr.py.

Background (REARCHITECTURE_IMPLEMENTATION_PLAN.md §9.1, 2026-07-03): ASR output is
NOT expected to match tests/golden/legacy_snapshot.jsonl byte-for-byte, even with
identical uv.lock-pinned package versions and identical audio (both confirmed
unchanged since before the corpus was transcribed) — output differs stably
(deterministic, not random), most likely due to GPU-driver-level numerical drift
outside this package's control. Empirically (20-sample broader check, 2026-07-03)
the divergence is small and concentrated in a minority of harder segments:
canto_ft median=1.0 mean=0.991 min=0.929 (n=16); whisper_v3 median=1.0 mean=0.961
min=0.759 (n=20). This script re-runs that check against the frozen golden
snapshot instead of an ad-hoc sample.

Usage: python tests/golden/check_asr_parity.py [--sample-size 40] [--seed 12345]
Exit code 0 = pass (aggregate median >= AGGREGATE_MEDIAN_MIN for every model with
data); 1 = fail. Individual segments below WARN_THRESHOLD are logged but do not by
themselves fail the gate (a few hard segments diverging is expected, not a bug).
"""

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.nodes.asr import ASR_MODELS, _load_and_resample, char_agreement

GOLDEN_PATH = Path(__file__).parent / "legacy_snapshot.jsonl"

AGGREGATE_MEDIAN_MIN = 0.95
WARN_THRESHOLD = 0.75

# Frozen snapshot rows carry the pre-migration model path (e.g.
# /mnt/Drive3/Development/AI-ML/canto-corpus/...) by design — it's a point-in-time
# record, not something to remap. Match candidates to a model_key by substring
# instead of an exact model-string comparison.
_MODEL_MATCH = {
    "canto_ft": "cantonese",
    "whisper_v3": "large-v3",
}


def load_golden_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            candidates = row.get("sidecars", {}).get("transcript", {}).get("asr_candidates", [])
            if len(candidates) >= 2 and os.path.exists(row["audio_path"]):
                rows.append(row)
    return rows


def legacy_text_for(row: dict, model_key: str) -> str:
    marker = _MODEL_MATCH[model_key]
    candidates = row["sidecars"]["transcript"]["asr_candidates"]
    for cand in candidates:
        if marker in cand.get("model", ""):
            return cand.get("text", "")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=40)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    rows = load_golden_rows(GOLDEN_PATH)
    print(f"golden rows with 2 candidates + audio on disk: {len(rows)}")
    random.Random(args.seed).shuffle(rows)
    sample = rows[: args.sample_size]
    print(f"checking {len(sample)} sample(s) (seed={args.seed})")

    from faster_whisper import WhisperModel
    models = {}
    for key, cfg in ASR_MODELS.items():
        models[key] = WhisperModel(cfg["id"], device="cuda", device_index=0,
                                    compute_type="int8_float16", cpu_threads=4)

    def transcribe(model_key: str, y16) -> str:
        cfg = ASR_MODELS[model_key]
        kwargs = {"beam_size": 1, "vad_filter": False, "temperature": 0.0,
                  "language": cfg["lang"], "initial_prompt": cfg["prompt"]}
        segments, _info = models[model_key].transcribe(y16, **kwargs)
        return "".join(s.text for s in segments).strip()

    agreements: dict[str, list[float]] = {k: [] for k in ASR_MODELS}
    warned: list[tuple[str, str, float]] = []

    for row in sample:
        y16 = _load_and_resample(row["audio_path"])
        if y16 is None:
            continue
        for model_key in ASR_MODELS:
            legacy = legacy_text_for(row, model_key)
            new = transcribe(model_key, y16)
            if not legacy or not new:
                continue
            agree = char_agreement([legacy, new])
            agreements[model_key].append(agree)
            if agree < WARN_THRESHOLD:
                warned.append((row["id"], model_key, agree))

    print()
    ok = True
    for model_key, scores in agreements.items():
        if not scores:
            print(f"{model_key}: no comparable rows")
            continue
        med = statistics.median(scores)
        mean = statistics.mean(scores)
        print(f"{model_key}: n={len(scores)} median={med:.3f} mean={mean:.3f} "
              f"min={min(scores):.3f} max={max(scores):.3f}")
        if med < AGGREGATE_MEDIAN_MIN:
            ok = False

    if warned:
        print(f"\n{len(warned)} individual segment(s) below WARN_THRESHOLD={WARN_THRESHOLD} "
              f"(logged, does not fail the gate):")
        for seg_id, model_key, agree in warned:
            print(f"  {seg_id} [{model_key}] agreement={agree:.3f}")

    print(f"\n{'PASS' if ok else 'FAIL'}: aggregate median >= {AGGREGATE_MEDIAN_MIN} "
          f"required per model")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
