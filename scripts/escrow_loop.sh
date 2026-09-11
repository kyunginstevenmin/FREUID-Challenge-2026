#!/usr/bin/env bash
# Checkpoint escrow (RUNBOOK-EC2.md "Window 2"): continuously deposit checkpoints,
# OOF files, logs, and mlruns to S3 so a spot interruption loses <= --save_every
# steps + one sync interval. Side effect: the S3 last-modified timestamps are a
# remotely observable heartbeat (aws s3 ls from anywhere; stale logs/ = job hung).
# Run in its own tmux window on the training box: bash scripts/escrow_loop.sh
set -u
BUCKET="${FREUID_BUCKET:-s3://freuid-ablation-839000214843/freuid}"
INTERVAL="${FREUID_ESCROW_INTERVAL:-600}"
cd "$(dirname "$0")/.."
while true; do
  for d in checkpoints oof logs mlruns; do
    [ -d "$d" ] && aws s3 sync "$d" "$BUCKET/$d" --quiet
  done
  echo "$(date -u +%H:%M:%SZ) escrow sync done"
  sleep "$INTERVAL"
done
