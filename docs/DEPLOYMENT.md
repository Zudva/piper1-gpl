# Деплой обучения Piper на RunPod и ImmersCloud

Полное руководство по запуску обучения на облачных GPU (RunPod, ImmersCloud) с автоматической синхронизацией чекпоинтов и логов в S3.

## Оглавление

- [Требования](#требования)
- [Быстрый старт](#быстрый-старт)
- [Настройка S3 хранилища](#настройка-s3-хранилища)
- [Деплой на RunPod](#деплой-на-runpod)
- [Деплой на ImmersCloud](#деплой-на-immerscloud)
- [Оптимизация для low-RAM серверов](#оптимизация-для-low-ram-серверов)
- [Работа с Docker-образом](#работа-с-docker-образом)
- [Fabric команды](#fabric-команды)
- [Мониторинг обучения](#мониторинг-обучения)
- [Troubleshooting](#troubleshooting)

---

## Требования

**Локально:**
- Git
- Docker (для сборки образа)
- Python 3.10+ с Fabric (`pip install fabric`)
- rsync
- AWS CLI (`pip install awscli`)

**На облачной платформе:**
- GPU: H100 NVL (рекомендуется), A100 80GB, RTX 4090
- CUDA 12.x драйверы
- Docker с NVIDIA runtime
- SSH доступ

---

## Быстрый старт

### 1. Локальная подготовка

```bash
# Клонируйте репозиторий
git clone https://github.com/Zudva/piper1-gpl.git
cd piper1-gpl

# Установите Fabric и зависимости
pip install fabric awscli boto3 s3fs

# Настройте S3 credentials (скопируйте .env.example)
cp .env.example .env
# Отредактируйте .env, добавьте AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.
```

### 2. Загрузка чекпоинта в S3

```bash
# Загрузите последний чекпоинт
./script/s3_sync.sh upload-checkpoint lightning_logs/version_3/checkpoints/epoch=749-step=355500-val_loss=27.5963.ckpt

# Проверьте
./script/s3_sync.sh list-checkpoints
```

### 3. Создание GPU инстанса

**RunPod:**
- Создайте pod: H100 NVL / A100 80GB
- Включите SSH порт (22)
- Получите SSH команду: `ssh root@ssh.runpod.io -p XXXXX`

**ImmersCloud:**
- Создайте инстанс с GPU
- Настройте SSH доступ

### 4. Настройка подключения

```bash
# RunPod
export RUNPOD_HOST=ssh.runpod.io
export RUNPOD_PORT=12345  # из SSH команды
export RUNPOD_USER=root

# ИЛИ ImmersCloud
export IMMERSCLOUD_HOST=your-instance.immerscloud.com
export IMMERSCLOUD_PORT=22
export IMMERSCLOUD_USER=root
```

### 5. Деплой и запуск

```bash
# RunPod
fab setup-runpod          # Первоначальная настройка
fab sync-to-runpod         # Синхронизация кода
fab start-training --batch-size=64 --precision=16-mixed

# ImmersCloud
fab setup-immerscloud
fab sync-to-immerscloud
fab start-training-immerscloud --batch-size=64
```

### 6. Мониторинг

```bash
# Логи в реальном времени
fab ssh-runpod --cmd="docker logs -f piper1-gpl-train-1"

# Или SSH в интерактивном режиме
fab ssh-runpod
# docker logs -f piper1-gpl-train-1
# nvidia-smi
```

### 7. Скачивание результатов

```bash
# Скачать чекпоинты и логи
fab sync-from-runpod --path=lightning_logs

# Или через S3 (чекпоинты автоматически загружаются)
./script/s3_sync.sh list-checkpoints
./script/s3_sync.sh download-latest
```

---

## Настройка S3 хранилища

### Timeweb S3 (рекомендуется для РФ)

Создайте `.env` файл в корне проекта:

```bash
# Timeweb S3-compatible storage
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
AWS_DEFAULT_REGION=ru-1-hot
AWS_ENDPOINT_URL=https://s3.twcstorage.ru
S3_BUCKET=your-bucket-id
S3_PREFIX=piper-training/felix_mirage

# Training configuration
CHECKPOINT=lightning_logs/version_3/checkpoints/epoch=749-step=355500-val_loss=27.5963.ckpt
BATCH_SIZE=64
PRECISION=16-mixed
ACCUM=1
MAX_EPOCHS=10000

# AWS transfer tuning
AWS_MAX_ATTEMPTS=10
AWS_RETRY_MODE=standard
AWS_S3_MAX_CONCURRENCY=10
```

### Проверка подключения

```bash
source .env
aws s3 ls s3://$S3_BUCKET/ --endpoint-url=$AWS_ENDPOINT_URL --region=$AWS_DEFAULT_REGION
```

### Автоматическая синхронизация при обучении

Установите `ENABLE_S3_SYNC=1` — чекпоинты будут автоматически загружаться в S3 после каждого сохранения (каждые 5000 шагов).

---

## Деплой на RunPod

### Вариант A: Прямое использование ghcr.io (рекомендуется)

```bash
# 1. Соберите и запушьте образ в GitHub Container Registry
docker build -f Dockerfile.train -t ghcr.io/zudva/piper-train:latest .
echo $GITHUB_TOKEN | docker login ghcr.io -u Zudva --password-stdin
docker push ghcr.io/zudva/piper-train:latest

# 2. На RunPod используйте образ напрямую
# В docker-compose.runpod.yml уже указан ghcr.io/zudva/piper-train:latest
```

### Вариант B: Загрузка образа из S3 (экономия трафика ghcr.io)

```bash
# 1. Экспорт образа в S3
docker build -f Dockerfile.train -t ghcr.io/zudva/piper-train:latest .
./script/docker_to_s3.sh export  # ~12 GB сжатый архив

# 2. На RunPod загрузите образ при старте
fab ssh-runpod --cmd="cd /workspace/piper1-gpl && ./script/docker_to_s3.sh load"
```

### Настройка Pod Template (RunPod UI)

**Container Image:** `ghcr.io/zudva/piper-train:latest`

**Environment Variables:**
```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_ENDPOINT_URL=https://s3.twcstorage.ru
AWS_DEFAULT_REGION=ru-1-hot
S3_BUCKET=your-bucket
S3_PREFIX=piper-training/felix_mirage
ENABLE_S3_SYNC=1
BATCH_SIZE=64
PRECISION=16-mixed
```

**Volume Mounts:**
- Container Path: `/workspace/piper1-gpl`
- Size: 100 GB

**Ports:**
- SSH: 22 (для удаленного доступа)
- TensorBoard: 6006 (опционально)

---

## Деплой на ImmersCloud

Процесс идентичен RunPod, используйте команды с суффиксом `-immerscloud`:

```bash
export IMMERSCLOUD_HOST=your-instance.com
fab setup-immerscloud
fab sync-to-immerscloud
fab start-training-immerscloud --batch-size=64
```

---

## Работа с Docker-образом

### Локальная сборка

```bash
docker build -f Dockerfile.train -t ghcr.io/zudva/piper-train:latest .
```

### Экспорт в S3

```bash
./script/docker_to_s3.sh export
# Создаст piper-train-latest.tar.gz (~12 GB)
# Загрузит в s3://bucket/piper-training/felix_mirage/docker-images/
```

### Загрузка из S3

```bash
./script/docker_to_s3.sh load
# Скачает и загрузит образ в Docker
```

### Список образов в S3

```bash
./script/docker_to_s3.sh list
```

---

## Fabric команды

### Общие команды

```bash
fab --list  # Показать все доступные команды
```

### RunPod

```bash
fab sync-to-runpod                     # Rsync кода на RunPod
fab sync-from-runpod                   # Скачать lightning_logs
fab sync-from-runpod --path=.env       # Скачать конкретный файл
fab ssh-runpod                         # Интерактивный SSH
fab ssh-runpod --cmd="nvidia-smi"     # Выполнить команду
fab setup-runpod                       # Первоначальная настройка
fab start-training --batch-size=64 --precision=16-mixed  # Запуск обучения
```

### ImmersCloud

Замените `runpod` на `immerscloud`:

```bash
fab sync-to-immerscloud
fab ssh-immerscloud
fab setup-immerscloud
fab start-training-immerscloud --batch-size=64
```

### Docker

```bash
fab build                              # Собрать образ локально
fab push                               # Запушить в REGISTRY
```

---

## Мониторинг обучения

### Логи Docker

```bash
# Через Fabric
fab ssh-runpod --cmd="docker logs -f piper1-gpl-train-1"

# Напрямую через SSH
ssh root@ssh.runpod.io -p XXXXX
docker logs -f piper1-gpl-train-1
```

### TensorBoard (опционально)

```bash
# На удаленном сервере
ssh root@ssh.runpod.io -p XXXXX
tensorboard --logdir /workspace/piper1-gpl/lightning_logs --host 0.0.0.0 --port 6006

# Локально (после sync-from-runpod)
tensorboard --logdir lightning_logs
```

### GPU мониторинг

```bash
fab ssh-runpod --cmd="nvidia-smi"
fab ssh-runpod --cmd="watch -n 1 nvidia-smi"
```

### Проверка чекпоинтов в S3

```bash
./script/s3_sync.sh list-checkpoints
```

---

## Параметры обучения

### Рекомендуемые настройки по GPU

| GPU | BATCH_SIZE | PRECISION | ACCUM | Примечание |
|-----|------------|-----------|-------|------------|
| H100 NVL (94GB) | 96-128 | 16-mixed | 1 | Попробуйте bf16-mixed |
| A100 80GB | 64-96 | 16-mixed | 1 | Стабильно |
| RTX 4090 (24GB) | 24-32 | 16-mixed | 1 | Локальная разработка |
| 2x RTX 4090 | 16-24 | 16-mixed | 1 | DDP на 2 GPU |

### Precision

- **16-mixed** — рекомендуется, совместимо с cuFFT
- **bf16-mixed** — может работать на H100, но cuFFT может не поддерживать
- **32** — точнее, но медленнее и больше памяти

### Gradient accumulation

Если не хватает памяти на большой batch, используйте `ACCUM=2` или `ACCUM=4`:

```bash
# Эффективный batch_size = BATCH_SIZE * ACCUM
BATCH_SIZE=32 ACCUM=2  # эффективно 64
```

---

## Troubleshooting

### Ошибка cuFFT doesn't support BFloat16

**Решение:** Используйте `PRECISION=16-mixed` вместо `bf16-mixed`

```bash
export PRECISION=16-mixed
fab start-training --precision=16-mixed
```

### Out of Memory

**Решение 1:** Уменьшите batch size

```bash
BATCH_SIZE=32 fab start-training
```

**Решение 2:** Включите gradient accumulation

```bash
BATCH_SIZE=16 ACCUM=4 fab start-training
```

### S3 upload failed

**Проверьте credentials:**

```bash
source .env
aws s3 ls s3://$S3_BUCKET/ --endpoint-url=$AWS_ENDPOINT_URL
```

**Увеличьте retry:**

```bash
export AWS_MAX_ATTEMPTS=20
```

### SSH connection refused

**RunPod:** Убедитесь что SSH порт включен в pod template

**Проверьте порт:**

```bash
echo $RUNPOD_PORT  # должен быть указан из RunPod UI
```

### Docker image not found

**Вариант A:** Загрузите из S3

```bash
fab ssh-runpod --cmd="cd /workspace/piper1-gpl && ./script/docker_to_s3.sh load"
```

**Вариант B:** Pull из ghcr.io

```bash
fab ssh-runpod --cmd="docker pull ghcr.io/zudva/piper-train:latest"
```

### Dataset not found

**Убедитесь что датасет загружен:**

```bash
# Проверьте наличие датасета в S3
./script/s3_sync.sh list  # (нужно добавить эту команду)

# Или загрузите вручную
fab ssh-runpod --cmd="cd /workspace/piper1-gpl && ./script/s3_sync.sh download-dataset"
```

---

## Полезные ссылки

- **GitHub репозиторий:** https://github.com/Zudva/piper1-gpl
- **RunPod документация:** https://docs.runpod.io
- **Timeweb S3:** https://timeweb.cloud/storage/s3
- **Original Piper:** https://github.com/rhasspy/piper

---

## Changelog

**v1.0.0 (2026-01-15)**
- ✅ S3 интеграция (Timeweb Cloud)
- ✅ Автоматическая синхронизация чекпоинтов
- ✅ Fabric команды для RunPod/ImmersCloud
- ✅ Docker образ с CUDA 12.4, PyTorch 2.3.1
- ✅ Поддержка H100 NVL, A100, RTX 4090
- ✅ Rsync синхронизация кода
- ✅ TensorBoard логи в S3
