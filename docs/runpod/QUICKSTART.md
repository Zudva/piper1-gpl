# Быстрый старт обучения Piper TTS

## ✅ Что уже установлено

- Python 3.11 с виртуальным окружением в `.venv`
- Все зависимости (PyTorch, Lightning, и др.)
- Собранные C++ расширения (espeak-ng, monotonic_align)
- 2x NVIDIA L40S GPU (46GB VRAM каждая)

## 🚀 Варианты запуска

### Вариант 1: Тест готовой модели (быстрый)

```bash
cd /workspace/piper1-gpl
source .venv/bin/activate
python tools/inference/test_model.py
```

Или напрямую:
```bash
source .venv/bin/activate
echo "Привет, это тест" | python -m piper \
  --model felix_mirage_epoch749.onnx \
  --config felix_mirage_epoch749.onnx.json \
  --output-file output.wav
```

### Вариант 2: Обучение с тестовым датасетом

```bash
cd /workspace/piper1-gpl
source .venv/bin/activate
python tools/runpod/start_training.py
```

С параметрами:
```bash
python start_training.py \
  --batch-size 80 \
  --num-gpus 2 \
  --max-epochs 1000
```

(То же самое, но с новым путём: `python tools/runpod/start_training.py ...`)

Возобновление с чекпоинта:
```bash
python start_training.py \
  --checkpoint lightning_logs/version_3/checkpoints/epoch=749-step=355500-val_loss=27.5963.ckpt \
  --batch-size 80 \
  --num-gpus 2
```

(То же самое, но с новым путём: `python tools/runpod/start_training.py ...`)

### Вариант 3: Bash скрипт

```bash
source .venv/bin/activate
bash tools/runpod/train_local.sh
```

Параметры через переменные окружения:
```bash
BATCH_SIZE=80 NUM_GPUS=2 MAX_EPOCHS=1000 bash tools/runpod/train_local.sh
```

### Вариант 4: Прямой запуск Python CLI

```bash
source .venv/bin/activate
python -m piper.train fit \
  --data.config_path=/path/to/config.json \
  --data.voice_name=my_voice \
  --data.csv_path=/path/to/metadata.csv \
  --data.audio_dir=/path/to/wavs \
  --model.sample_rate=22050 \
  --data.espeak_voice=ru \
  --data.cache_dir=/path/to/.cache \
  --data.batch_size=80 \
  --trainer.devices=2 \
  --trainer.accelerator=gpu \
  --trainer.strategy=ddp_find_unused_parameters_true
```

## 📁 Структура датасета

Ваш датасет должен иметь структуру:
```
/path/to/dataset/
├── config.json           # Конфигурация модели
├── metadata.csv          # Список файлов и текстов (формат: file.wav|Текст)
├── wavs/                 # Аудио файлы
│   ├── utterance1.wav
│   ├── utterance2.wav
│   └── ...
└── .cache/              # Кэш (создается автоматически)
```

Формат `metadata.csv`:
```
utterance1.wav|Первое предложение для обучения.
utterance2.wav|Второе предложение для обучения.
utterance3.wav|Третье предложение для обучения.
```

## 🎯 Оптимизация для L40S

Для 2x L40S (46GB VRAM каждая):
- **Batch size**: 80-96 (максимальная скорость)
- **Precision**: `16-mixed` (стабильная) или `bf16-mixed` (быстрее)
- **Workers**: 4 (при 62GB RAM)
- **Strategy**: `ddp_find_unused_parameters_true` (для 2 GPU)

## 📊 Мониторинг

Во время обучения:
```bash
# Просмотр использования GPU
watch -n 2 nvidia-smi

# TensorBoard
tensorboard --logdir=lightning_logs
```

## 🔍 Проверка чекпоинтов

```bash
# Найти все чекпоинты
find lightning_logs -name "*.ckpt"

# Экспорт в ONNX
python -m piper.train.export_onnx \
  --checkpoint path/to/checkpoint.ckpt \
  --output-file model.onnx
```

## ⚠️ Примечания

- **Docker**: Не работает внутри контейнера (Docker-in-Docker требует привилегий)
- **Тестовый датасет**: Создается автоматически скриптом `start_training.py`
- **Реальное обучение**: Нужны настоящие аудио файлы (не silence)

## 🐛 Проблемы

**Ошибка "FileNotFoundError"**: 
- Проверьте пути к датасету
- Используйте абсолютные пути

**Out of Memory (OOM)**:
- Уменьшите `batch_size`
- Увеличьте `accumulate_grad_batches`

**Терминал не отвечает**:
- Используйте Python скрипты вместо bash команд
