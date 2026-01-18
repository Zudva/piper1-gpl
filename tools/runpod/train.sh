#!/bin/bash
set -euo pipefail

# Quick training start for RunPod
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -f ".venv/bin/activate" ]; then
	# shellcheck disable=SC1091
	source .venv/bin/activate
fi

python tools/runpod/runpod_launch.py "$@"
