# Piper TTS Training

## Требования

- Python 3.10+
- CUDA 12.1+ (для GPU)
- 2x GPU (рекомендуется)

## Установка

```bash
# Создать и активировать venv
python3 -m venv .venv
source .venv/bin/activate

# Установить PyTorch с CUDA 12.1
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Установить piper-tts в режиме разработки
pip install -e .[train]

# Собрать C++ расширения
./build_monotonic_align.sh
python setup.py build_ext --inplace
```

## Запуск обучения

### Продолжение обучения с чекпоинта

```bash
# Активировать venv
source .venv/bin/activate

# Запустить скрипт продолжения
./train_resume.sh [CHECKPOINT_PATH] [MAX_EPOCHS]

# Пример с параметрами по умолчанию
./train_resume.sh
```

### Параметры скрипта

- `CHECKPOINT_PATH` - путь к чекпоинту (по умолчанию: `lightning_logs/version_2/checkpoints/epoch=426-step=202398.ckpt`)
- `MAX_EPOCHS` - максимальное число эпох (по умолчанию: `10000`)

## Экспорт в ONNX

```bash
source .venv/bin/activate

EXPORT_ONLY=1 \
CHECKPOINT=/workspace/piper1-gpl/lightning_logs/version_2/checkpoints/epoch=426-step=202398.ckpt \
OUTPUT_FILE=/workspace/piper1-gpl/felix_mirage_epoch426.onnx \
docker compose -f docker-compose.train.yml run --rm \
  -e EXPORT_ONLY -e CHECKPOINT -e OUTPUT_FILE piper-train
```

## Тестирование модели

```bash
source .venv/bin/activate

# Генерация одного варианта
python test_onnx_generate.py \
  --model felix_mirage_epoch426.onnx \
  --text "Привет! Это тест голоса."

# Генерация нескольких вариантов с разными параметрами
python test_onnx_variations.py \
  --text "Привет! Это демонстрация разных параметров генерации голоса."
```

## VS Code

Проект настроен на автоматическое использование venv из `.venv`.  
На RunPod обычно путь будет `/workspace/piper1-gpl/.venv`, локально — путь зависит от вашей рабочей директории.

## Мониторинг обучения

```bash
# TensorBoard
tensorboard --logdir lightning_logs

# Проверка чекпоинтов
ls -lh lightning_logs/version_*/checkpoints/
```
