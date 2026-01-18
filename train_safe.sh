#!/bin/bash
# Безопасное продолжение обучения с частым сохранением

echo "🚀 Запуск БЕЗОПАСНОГО обучения с чекпоинта epoch 749..."
echo "   • Автосохранение каждые 25 эпох"
echo "   • Сохранение топ-3 лучших моделей"
echo "   • Последний чекпоинт всегда сохраняется"
echo ""

cd /workspace/piper1-gpl

source .venv/bin/activate

python -m piper.train fit \
  --config lightning_logs/version_3/config.yaml \
  --ckpt_path lightning_logs/version_14/checkpoints/last.ckpt \
  --data.csv_path /workspace/datasets/felix_mirage/metadata_2col.csv \
  --data.cache_dir /workspace/datasets/felix_mirage/.cache \
  --data.config_path /workspace/datasets/felix_mirage/config.json \
  --data.audio_dir /workspace/datasets/felix_mirage/wavs \
  --data.batch_size 64 \
  --trainer.devices 2 \
  --trainer.strategy ddp_find_unused_parameters_true \
  --trainer.precision 16-mixed \
  --trainer.max_epochs 1000 \
  --trainer.accumulate_grad_batches 1 \
  --data.num_workers 4 \
  --trainer.log_every_n_steps 50 \
  --trainer.val_check_interval 0.5
