# 🚀 RunPod Piper Training - READY TO USE

## ✅ Вы уже внутри контейнера RunPod!

**Docker НЕ нужен** - все зависимости установлены, GPU доступны.

## 🎯 Быстрый старт (3 команды)

```bash
# 1. Проверка окружения
bash tools/runpod/check.sh

# 2. Запуск обучения
bash tools/runpod/train.sh

# Или с возобновлением
bash tools/runpod/train.sh --resume
```

## 📋 Что уже установлено

- ✅ Python 3.11 + виртуальное окружение
- ✅ PyTorch + Lightning + все зависимости
- ✅ Собранные C++ расширения (espeak-ng, monotonic_align)
- ✅ 2x NVIDIA L40S GPU (46GB VRAM каждая)
- ✅ Готовые модели felix_mirage (epoch 426, 649, 749)

## 🔍 Доступные команды

### Проверка окружения
```bash
python tools/runpod/check_env.py
```

### Запуск обучения
```bash
# Автоматический поиск датасета и чекпоинта
python tools/runpod/runpod_launch.py

# С возобновлением от последнего чекпоинта
python tools/runpod/runpod_launch.py --resume

# Только проверка (без обучения)
python tools/runpod/runpod_launch.py --check-only

# С явным указанием путей
python tools/runpod/runpod_launch.py \
  --dataset /workspace/datasets/felix_mirage \
  --checkpoint lightning_logs/version_3/checkpoints/epoch=749.ckpt
```

### Тест модели
```bash
python tools/inference/test_model.py
```

### Мониторинг
```bash
# GPU
watch -n 2 nvidia-smi

# TensorBoard
tensorboard --logdir=lightning_logs --host 0.0.0.0
```

## 📁 Где искать датасет

Скрипт `tools/runpod/runpod_launch.py` автоматически проверит:
- `/workspace/datasets/felix_mirage`
- `/data/felix_mirage`
- `/data`

Также можно явно указать путь через `DATA_DIR=/path/to/dataset`.

## 🎛️ Оптимальные параметры для 2x L40S

```bash
--data.batch_size=80          # 80-96 для максимальной скорости
--data.num_workers=4          # 4 воркера (62GB RAM)
--trainer.devices=2           # 2 GPU
--trainer.precision=16-mixed  # стабильно (или bf16-mixed)
--trainer.strategy=ddp_find_unused_parameters_true
```

## 🐛 Troubleshooting

**Терминал не отвечает?**
- Используйте Python скрипты вместо bash команд
- `python tools/runpod/check_env.py` вместо `ls`

**Dataset not found?**
- Проверьте пути в `check_env.py`
- Проверьте пути в `tools/runpod/check_env.py`
- Убедитесь, что volume смонтирован

**Out of Memory?**
- Уменьшите batch_size: `--data.batch_size=40`
- Увеличьте accumulation: `--trainer.accumulate_grad_batches=2`

## 📚 Документация

- **RUNPOD_COMMANDS.md** - все команды
- **QUICKSTART.md** - подробное руководство
- **docs/TRAINING.md** - полная документация по обучению

## ⚡ Прямой запуск (если датасет известен)

```bash
source .venv/bin/activate
python -m piper.train fit \
  --ckpt_path=lightning_logs/version_3/checkpoints/epoch=749.ckpt \
  --data.config_path=/workspace/datasets/felix_mirage/config.json \
  --data.csv_path=/workspace/datasets/felix_mirage/metadata_2col.csv \
  --data.audio_dir=/workspace/datasets/felix_mirage/wavs \
  --data.batch_size=80 \
  --trainer.devices=2 \
  --trainer.accelerator=gpu
```

---

**Вопросы?** См. [RUNPOD_COMMANDS.md](RUNPOD_COMMANDS.md)
