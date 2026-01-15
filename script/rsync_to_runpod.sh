#!/bin/bash
# Quick rsync wrapper for manual sync to RunPod
# Usage: ./script/rsync_to_runpod.sh [host] [port]

set -euo pipefail

RUNPOD_HOST="${1:-${RUNPOD_HOST:-}}"
RUNPOD_PORT="${2:-${RUNPOD_PORT:-22}}"
RUNPOD_USER="${RUNPOD_USER:-root}"
REMOTE_DIR="/workspace/piper1-gpl"

if [ -z "$RUNPOD_HOST" ]; then
    echo "Usage: $0 <host> [port]"
    echo "   or: RUNPOD_HOST=ssh.runpod.io RUNPOD_PORT=12345 $0"
    exit 1
fi

echo "📤 Syncing to ${RUNPOD_USER}@${RUNPOD_HOST}:${RUNPOD_PORT}${REMOTE_DIR}"

rsync -avz --delete \
    --exclude-from=.rsyncignore \
    -e "ssh -p ${RUNPOD_PORT}" \
    ./ "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/"

echo "✓ Sync complete"
