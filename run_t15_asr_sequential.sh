#!/usr/bin/env bash
# T15 asr.transcribe: run each active model exclusively on both GPUs, sequentially.
# canto_ft retired 2026-07-13 (DECISIONS.md) -- slow (sequential decode, ~4.45/s/GPU
# hard ceiling) AND inaccurate (17-36% CER vs qwen3_asr's ~0.4%), same profile that
# got whisper_v3 retired 2026-07-10. Two active models remain: qwen3_asr, sense_voice.
#
# Sequential-exclusive, NOT interleaved: measured 2026-07-13 that pairing qwen3_asr
# and sense_voice on the SAME device (each device's pool has target=1, a single
# semaphore) doesn't share the GPU fairly -- nvidia-smi pmon showed sense_voice's
# workers at 0% SM across 5 consecutive samples while qwen3_asr held the device the
# whole time (starvation, not slow sharing). Running each model with exclusive
# access to both GPUs avoids the per-device semaphore contention entirely.
#
# --batch 64: the CLI default (8) badly under-fed Qwen3ASRWorker's
# max_inference_batch_size=64 (pipeline/nodes/asr.py load_model()) -- the supervisor
# only ever queued 8-item chunks, so the model never got to use its own tuned batch
# capacity. Measured 2026-07-13 at the CLI default: only ~17.4/s combined (~8.7/s/GPU,
# matching the 2026-07-07 tuning curve's batch=8 region) and only 5-6GB/24.5GB VRAM
# used per GPU -- confirms the model was starved, not GPU-bound. --batch 64 matches
# the model's own tuned cap (2026-07-07 measurement: 64=30.1/s/GPU, diminishing
# returns past 64) and should use the free VRAM headroom.
set -euo pipefail
cd /home/typangaa/Documents/canto-hk-speech-pipeline

echo "=== qwen3_asr (both GPUs) started $(date) ==="
.venv/bin/python -m pipeline.cli run asr.transcribe --models qwen3_asr,qwen3_asr --devices cuda:0,cuda:1 --batch 64

echo "=== sense_voice (both GPUs) started $(date) ==="
.venv/bin/python -m pipeline.cli run asr.transcribe --models sense_voice,sense_voice --devices cuda:0,cuda:1 --batch 64

echo "=== ALL DONE $(date) ==="
