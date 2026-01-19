#!/usr/bin/env bash
set -euo pipefail

REPORT_DIR="${1:-/media/zudva/git1/git/piper-training/datasets/felix_mirage_prepared_sr22050_seg15/reports/validation_whisper_sharded_largev3_20260118_075419}"

if [[ ! -d "$REPORT_DIR" ]]; then
  echo "Report dir not found: $REPORT_DIR"
  exit 1
fi

RUNNER_LOG="$REPORT_DIR/runner.log"
SHARD_REPORTS="$REPORT_DIR/shard_reports"

while true; do
  clear
  echo "=== Whisper sharded status ==="
  echo "Report: $REPORT_DIR"
  echo "Time:   $(date)"
  echo

  echo "-- Runner log (tail)"
  if [[ -f "$RUNNER_LOG" ]]; then
    tail -n 5 "$RUNNER_LOG"
  else
    echo "(missing)"
  fi
  echo

  echo "-- Shard logs (sizes)"
  if [[ -d "$SHARD_REPORTS" ]]; then
    for shard in "$SHARD_REPORTS"/shard_*; do
      [[ -d "$shard" ]] || continue
      log="$shard/validator_whisper.log"
      if [[ -f "$log" ]]; then
        size=$(wc -c < "$log" | tr -d ' ')
        echo "$(basename "$shard"): ${size} bytes"
      else
        echo "$(basename "$shard"): (log missing)"
      fi
    done
  else
    echo "(shard_reports missing)"
  fi
  echo

  echo "-- Running processes"
  pgrep -af 'validate_dataset_full\.py.*--whisper' || echo "(none)"
  echo

  echo "-- GPU usage"
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits || true

  sleep 10
 done
