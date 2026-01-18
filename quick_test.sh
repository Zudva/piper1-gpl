#!/bin/bash
# Quick test script - runs inference with existing model

source /workspace/piper1-gpl/.venv/bin/activate

MODEL="felix_mirage_epoch749.onnx"
CONFIG="felix_mirage_epoch749.onnx.json"

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
