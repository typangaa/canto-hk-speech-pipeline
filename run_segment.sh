#!/usr/bin/env bash
# Run segmentation — expects HUGGING_FACE_HUB_TOKEN already exported in shell
set -e
cd "$(dirname "$0")"
nohup uv run python scripts/03_segment.py --source all > /tmp/segment_diarize.log 2>&1 &
echo "Segmentation PID: $!"
