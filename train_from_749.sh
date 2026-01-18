#!/bin/bash
# Продолжение обучения с эпохи 749

echo "🚀 Запуск обучения с чекпоинта epoch 749..."
echo ""

cd /workspace/piper1-gpl

source .venv/bin/activate

python -m piper.train fit \
  --config lightning_logs/version_3/config.yaml \
  --ckpt_path lightning_logs/version_3/checkpoints/epoch=749-step=355500-val_loss=27.5963.ckpt \
  --data.csv_path /workspace/datasets/felix_mirage/metadata_2col.csv \
  --data.cache_dir /workspace/datasets/felix_mirage/.cache \
  --data.config_path /workspace/datasets/felix_mirage/config.json \
  --data.audio_dir /workspace/datasets/felix_mirage/wavs \
  --data.batch_size 48 \
  --trainer.devices 2 \
  --trainer.strategy ddp_find_unused_parameters_true \
  --trainer.precision 16-mixed \
  --trainer.max_epochs 1000 \
  --trainer.accumulate_grad_batches 1 \
  --data.num_workers 4 \
  --trainer.log_every_n_steps 50 \
  --trainer.val_check_interval 0.5
