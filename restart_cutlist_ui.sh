#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUTLIST="${CUTLIST:-}"
AUDIO_ROOT="${AUDIO_ROOT:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7860}"
LOG_FILE="${LOG_FILE:-$DIR/cutlist_review_ui.log}"
WORK_ON_COPY="${WORK_ON_COPY:-0}"
WORK_COPY_DIR="${WORK_COPY_DIR:-}"
WORK_COPY_MODE="${WORK_COPY_MODE:-hardlink}"

if [[ -z "$CUTLIST" || -z "$AUDIO_ROOT" ]]; then
  echo "Usage: CUTLIST=path/to/cutlist.jsonl AUDIO_ROOT=path/to/audio_root $0" >&2
  echo "Optional: HOST=127.0.0.1 PORT=7860 LOG_FILE=$DIR/cutlist_review_ui.log" >&2
  exit 1
fi

# Stop existing UI (ignore if not running)
pkill -f "cutlist_review_ui.py" >/dev/null 2>&1 || true

exec "$DIR/.venv/bin/python" "$DIR/script/cutlist_review_ui.py" \
  --cutlist "$CUTLIST" \
  --audio-root "$AUDIO_ROOT" \
  $([[ "$WORK_ON_COPY" == "1" ]] && echo "--work-on-copy") \
  $([[ -n "$WORK_COPY_DIR" ]] && echo "--work-copy-dir" && echo "$WORK_COPY_DIR") \
  $([[ "$WORK_ON_COPY" == "1" ]] && echo "--work-copy-mode" && echo "$WORK_COPY_MODE") \
  --host "$HOST" \
  --port "$PORT" \
  --log-file "$LOG_FILE"
