#!/bin/bash
# Quick test script - runs inference with existing model

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

MODEL="${MODEL_PATH:-felix_mirage_epoch749.onnx}"
CONFIG="${CONFIG_PATH:-felix_mirage_epoch749.onnx.json}"

if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found: $MODEL"
    exit 1
fi

echo "Тестирование модели Феликс Мираж." | python -m piper \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output-file test_output.wav

if [ -f test_output.wav ]; then
    echo "✓ Success! Audio generated: test_output.wav"
    ls -lh test_output.wav
else
    echo "✗ Failed to generate audio"
    exit 1
fi
