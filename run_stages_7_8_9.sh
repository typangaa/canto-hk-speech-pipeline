#!/bin/bash
# run_stages_7_8_9.sh
# Runs Stage 7 (G2P), 8 (speaker_id), 9 (manifest) sequentially for all sources.
# Stage 7 podcast is already running (check PID separately).
# This script handles rthk + youtube re-run (idempotent) + Stage 8 + 9.

set -e
cd "$(dirname "$0")"
LOG=metadata/logs/stages_7_8_9_chain.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Stage chain started: $(date) ==="

echo ""
echo "--- Stage 7 G2P: waiting for podcast run to finish ---"
# Wait for any running 07_g2p podcast process
while pgrep -f "07_g2p.py.*podcast" > /dev/null 2>&1; do
    DONE=$(find data/filtered/podcast -name "*.jyutping.json" 2>/dev/null | wc -l)
    echo "  $(date +%H:%M:%S) podcast G2P progress: ${DONE} done"
    sleep 60
done
echo "  Podcast G2P process finished."

echo ""
echo "--- Stage 7 G2P: rthk + youtube (idempotent, fast) ---"
.venv/bin/python scripts/07_g2p.py --source rthk
.venv/bin/python scripts/07_g2p.py --source youtube
echo "  G2P rthk+youtube done."

echo ""
echo "--- Stage 8 speaker_id: podcast ---"
.venv/bin/python scripts/08_speaker_id.py --source podcast
echo "  Speaker ID podcast done."

echo ""
echo "--- Stage 8 speaker_id: rthk + youtube ---"
.venv/bin/python scripts/08_speaker_id.py --source rthk
.venv/bin/python scripts/08_speaker_id.py --source youtube
echo "  Speaker ID rthk+youtube done."

echo ""
echo "--- Stage 9 manifest rebuild ---"
.venv/bin/python scripts/09_manifest.py
echo "  Manifest done."

echo ""
echo "=== Final stats: $(date) ==="
wc -l metadata/manifest.jsonl metadata/train.jsonl metadata/val.jsonl 2>/dev/null
echo "=== Chain complete ==="
