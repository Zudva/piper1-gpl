#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Настройки
CHECKPOINT="${1:-lightning_logs/version_2/checkpoints/epoch=426-step=202398.ckpt}"
DATA_DIR="${DATA_DIR:-/workspace/datasets/felix_mirage}"
MAX_EPOCHS="${2:-10000}"

echo "=== Продолжение обучения Piper TTS ==="
echo "Checkpoint: $CHECKPOINT"
echo "Data dir: $DATA_DIR"
echo "Max epochs: $MAX_EPOCHS"
echo ""

# Активировать venv если нужно
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Активация venv..."
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Запуск обучения (используем только RTX 3090: GPU 0,1)
CUDA_VISIBLE_DEVICES=0,1 python -m piper.train fit \
    --ckpt_path="$CHECKPOINT" \
    --data.config_path="$DATA_DIR/config.json" \
    --data.voice_name=felix_mirage \
    --data.csv_path="$DATA_DIR/metadata_2col.csv" \
    --data.audio_dir="$DATA_DIR/wavs" \
    --model.sample_rate=22050 \
    --data.espeak_voice=ru \
    --data.cache_dir="$DATA_DIR/.cache" \
    --data.batch_size=16 \
    --trainer.max_epochs="$MAX_EPOCHS" \
    --trainer.check_val_every_n_epoch=1 \
    --trainer.strategy=ddp_find_unused_parameters_true \
    --trainer.devices=2 \
    --trainer.accelerator=gpu
