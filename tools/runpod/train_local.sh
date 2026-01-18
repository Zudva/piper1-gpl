#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Activating virtual environment ==="
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "=== Checking GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "=== Creating test dataset directory ==="
TEST_DATA_DIR="$REPO_ROOT/test_dataset"
mkdir -p "$TEST_DATA_DIR/wavs"
mkdir -p "$TEST_DATA_DIR/.cache"

# Create minimal config.json if not exists
if [ ! -f "$TEST_DATA_DIR/config.json" ]; then
    echo "=== Creating config.json ==="
    cat > "$TEST_DATA_DIR/config.json" << 'EOF'
{
  "audio": {
    "sample_rate": 22050
  },
  "espeak": {
    "voice": "ru"
  },
  "inference": {
    "noise_scale": 0.667,
    "length_scale": 1.0,
    "noise_w": 0.8
  },
  "phoneme_type": "espeak",
  "phoneme_map": {},
  "phoneme_id_map": {}
}
EOF
fi

# Create minimal metadata CSV if not exists
if [ ! -f "$TEST_DATA_DIR/metadata.csv" ]; then
    echo "=== Creating metadata.csv ==="
    echo "test1.wav|Привет, это тестовое обучение." > "$TEST_DATA_DIR/metadata.csv"
    echo "test2.wav|Проверка работы системы обучения." >> "$TEST_DATA_DIR/metadata.csv"
    echo "test3.wav|Нейросетевой синтез речи на русском языке." >> "$TEST_DATA_DIR/metadata.csv"
fi

# Generate test audio files (silence) if not exist
if [ ! -f "$TEST_DATA_DIR/wavs/test1.wav" ]; then
    echo "=== Generating test audio files ==="
    for i in 1 2 3; do
        ffmpeg -f lavfi -i anullsrc=r=22050:cl=mono -t 1 -y "$TEST_DATA_DIR/wavs/test$i.wav" 2>/dev/null || \
        sox -n -r 22050 -c 1 "$TEST_DATA_DIR/wavs/test$i.wav" trim 0.0 1.0 2>/dev/null || \
        echo "Warning: Could not generate test$i.wav - install sox or ffmpeg"
    done
fi

echo "=== Starting training ==="
CHECKPOINT="${CHECKPOINT:-}"
if [ -n "$CHECKPOINT" ] && [ -f "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT"
    CKPT_ARG="--ckpt_path=$CHECKPOINT"
else
    echo "Starting from scratch (no checkpoint)"
    CKPT_ARG=""
fi

python -m piper.train fit \
    $CKPT_ARG \
    --data.config_path="$TEST_DATA_DIR/config.json" \
    --data.voice_name=test_voice \
    --data.csv_path="$TEST_DATA_DIR/metadata.csv" \
    --data.audio_dir="$TEST_DATA_DIR/wavs" \
    --model.sample_rate=22050 \
    --data.espeak_voice=ru \
    --data.cache_dir="$TEST_DATA_DIR/.cache" \
    --data.batch_size=${BATCH_SIZE:-32} \
    --data.num_workers=${NUM_WORKERS:-4} \
    --trainer.precision=${PRECISION:-16-mixed} \
    --trainer.max_epochs=${MAX_EPOCHS:-100} \
    --trainer.devices=${NUM_GPUS:-2} \
    --trainer.accelerator=gpu \
    --trainer.strategy=$([ "${NUM_GPUS:-2}" = "1" ] && echo "auto" || echo "ddp_find_unused_parameters_true") \
    --trainer.accumulate_grad_batches=${ACCUM:-1} \
    --trainer.check_val_every_n_epoch=1

echo "=== Training completed ==="
