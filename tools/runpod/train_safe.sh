#!/bin/bash
# Безопасное продолжение обучения с частым сохранением

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-/workspace/datasets/felix_mirage}"
CONFIG_YAML="${CONFIG_YAML:-lightning_logs/version_3/config.yaml}"
CKPT_PATH="${CKPT_PATH:-lightning_logs/version_14/checkpoints/last.ckpt}"

echo "🚀 Запуск БЕЗОПАСНОГО обучения с чекпоинта epoch 749..."
echo "   • Автосохранение каждые 25 эпох"
echo "   • Сохранение топ-3 лучших моделей"
echo "   • Последний чекпоинт всегда сохраняется"
echo ""

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m piper.train fit \
  --config "$CONFIG_YAML" \
  --ckpt_path "$CKPT_PATH" \
  --data.csv_path "$DATA_DIR/metadata_2col.csv" \
  --data.cache_dir "$DATA_DIR/.cache" \
  --data.config_path "$DATA_DIR/config.json" \
  --data.audio_dir "$DATA_DIR/wavs" \
  --data.batch_size 64 \
  --trainer.devices 2 \
  --trainer.strategy ddp_find_unused_parameters_true \
  --trainer.precision 16-mixed \
  --trainer.max_epochs 1000 \
  --trainer.accumulate_grad_batches 1 \
  --data.num_workers 4 \
  --trainer.log_every_n_steps 50 \
  --trainer.val_check_interval 0.5
