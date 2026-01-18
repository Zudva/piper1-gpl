# RunPod Quick Commands

## 1️⃣ Проверка окружения
```bash
cd /workspace/piper1-gpl
source .venv/bin/activate
python tools/runpod/check_env.py
```

## 2️⃣ Запуск обучения

### Вариант A: Автоматический запуск (рекомендуется)
```bash
cd /workspace/piper1-gpl
source .venv/bin/activate
python tools/runpod/runpod_launch.py
```

## ✅ Dataset validation (100%) before training
```bash
cd /workspace/piper1-gpl
source .venv/bin/activate

python script/validate_dataset_full.py \
  --dataset ${DATA_DIR:-/workspace/datasets/felix_mirage} \
  --whisper --require-whisper
```

### Вариант B: С возобновлением
```bash
python tools/runpod/runpod_launch.py --resume
```

### Вариант C: Только проверка
```bash
python tools/runpod/runpod_launch.py --check-only
```

### Вариант D: С явным указанием путей
```bash
python tools/runpod/runpod_launch.py \
  --dataset /workspace/datasets/felix_mirage \
  --checkpoint lightning_logs/version_3/checkpoints/epoch=749-step=355500-val_loss=27.5963.ckpt
```

## 3️⃣ Прямой запуск (если датасет в /workspace/datasets/felix_mirage)
```bash
cd /workspace/piper1-gpl
source .venv/bin/activate

python -m piper.train fit \
  --ckpt_path=lightning_logs/version_3/checkpoints/epoch=749-step=355500-val_loss=27.5963.ckpt \
  --data.config_path=/workspace/datasets/felix_mirage/config.json \
  --data.voice_name=felix_mirage \
  --data.csv_path=/workspace/datasets/felix_mirage/metadata_2col.csv \
  --data.audio_dir=/workspace/datasets/felix_mirage/wavs \
  --model.sample_rate=22050 \
  --data.espeak_voice=ru \
  --data.cache_dir=/workspace/datasets/felix_mirage/.cache \
  --data.batch_size=80 \
  --data.num_workers=4 \
  --trainer.precision=16-mixed \
  --trainer.max_epochs=10000 \
  --trainer.devices=2 \
  --trainer.accelerator=gpu \
  --trainer.strategy=ddp_find_unused_parameters_true \
  --trainer.check_val_every_n_epoch=1
```

## 4️⃣ Тест модели
```bash
cd /workspace/piper1-gpl
source .venv/bin/activate
python tools/inference/test_model.py
```

## 5️⃣ Мониторинг
```bash
# GPU usage
watch -n 2 nvidia-smi

# TensorBoard
tensorboard --logdir=lightning_logs --host 0.0.0.0 --port 6006
```

## 6️⃣ Экспорт в ONNX
```bash
cd /workspace/piper1-gpl
source .venv/bin/activate

python -m piper.train.export_onnx \
  --checkpoint lightning_logs/version_X/checkpoints/epoch=XXX.ckpt \
  --output-file model_export.onnx
```

## ⚙️ Параметры для L40S (2x46GB)
- **batch_size**: 80-96 (оптимально)
- **num_workers**: 4
- **precision**: 16-mixed (стабильно) или bf16-mixed (быстрее)
- **devices**: 2 (для dual GPU)
- **strategy**: ddp_find_unused_parameters_true

## 🔍 Где искать датасет
Проверьте эти пути:
- `/workspace/datasets/felix_mirage`
- `/data/felix_mirage`
- `/data`
- Любой смонтированный volume

## ⚠️ Docker НЕ нужен!
Вы уже внутри контейнера RunPod - используйте Python команды напрямую.
