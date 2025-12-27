# Инструкция по обучению голосовой модели Piper

## Требования

- NVIDIA GPU (протестировано на RTX 4090)
- Docker с поддержкой NVIDIA runtime
- Docker Compose v3.9+
- Подготовленный датасет в формате CSV

## Подготовка данных

Датасет должен находиться в директории: `D:\git\nik-v-local-talking-llm\actors\felix_mirage`

Структура данных:
```
felix_mirage/
├── datasets/
│   └── elevenlabs/
│       └── zWSsRd3J6WyZFl12aGMB/
│           ├── metadata.csv          # Формат: audio.wav|text
│           └── wavs/                 # WAV файлы (44100 Hz)
└── felix_mirage.json                 # Конфигурация модели (опционально)
```

## Запуск обучения

### Обучение с нуля

```powershell
cd E:\git\piper1-gpl
docker compose -f docker-compose.train.yml run --rm piper-train
```

### Fine-tuning с checkpoint (опционально)

1. Аутентификация в HuggingFace:
```powershell
huggingface-cli login
```

2. Скачивание checkpoint:
```powershell
huggingface-cli download rhasspy/piper-checkpoints --include "ru/ru_RU/dmitri/medium/*.ckpt" --local-dir E:\git\piper\checkpoints
```

3. Добавьте в `docker-compose.train.yml` параметр:
```yaml
--ckpt_path /checkpoints/ru/ru_RU/dmitri/medium/epoch_XXXX.ckpt \
```

## Параметры обучения

Текущая конфигурация (для RTX 4090):
- `batch_size`: 28
- `sample_rate`: 44100 Hz
- `max_epochs`: 10000
- `espeak_voice`: ru (Russian)
- `check_val_every_n_epoch`: 1

## Мониторинг

Логи обучения сохраняются в:
- `/data/.cache` - кэш датасета
- Lightning logs - метрики и чекпоинты (внутри контейнера)

## Экспорт модели

После обучения конвертируйте checkpoint в ONNX:

```bash
python3 -m piper.train.export_onnx \
  --checkpoint /path/to/checkpoint.ckpt \
  --output-file /data/felix_mirage.onnx
```

## Настройка GPU

Если нужно использовать конкретную GPU:
```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=0  # ID GPU
```

## Troubleshooting

### Network connectivity issues
Используется `network_mode: host` для доступа к PyPI и другим ресурсам.

### Out of memory
Уменьшите `batch_size` в команде обучения.

### Read-only filesystem
Venv создается в `/tmp/piper-venv` чтобы избежать конфликтов с bind mount.
