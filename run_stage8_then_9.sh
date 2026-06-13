#!/bin/bash
# run_stage8_then_9.sh
# Runs Stage 8 (speaker_id) for all sources, then Stage 9 (manifest).
# Run with: setsid bash run_stage8_then_9.sh &
set -e
cd "$(dirname "$0")"
VENV=.venv/bin/python
LOG=metadata/logs/stages_8_9_chain.log

mkdir -p metadata/logs
exec >> "$LOG" 2>&1
echo "=== Stage 8→9 chain started: $(date) ==="

echo "--- Stage 8 speaker_id: podcast ---"
$VENV scripts/08_speaker_id.py --source podcast
echo "--- Stage 8 speaker_id: rthk ---"
$VENV scripts/08_speaker_id.py --source rthk
echo "--- Stage 8 speaker_id: youtube ---"
$VENV scripts/08_speaker_id.py --source youtube

echo "--- Stage 9 manifest rebuild ---"
$VENV scripts/09_manifest.py

echo "--- Final counts ---"
wc -l metadata/manifest.jsonl metadata/train.jsonl metadata/val.jsonl 2>/dev/null

echo "=== Chain complete: $(date) ==="
