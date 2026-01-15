# S3 интеграция для Piper Training

Автоматическая синхронизация чекпоинтов, логов и датасетов с S3-совместимым хранилищем (Timeweb Cloud, AWS S3, MinIO).

## Возможности

- ✅ **Автоматическая загрузка чекпоинтов** после каждого сохранения (каждые 5000 шагов)
- ✅ **Автоматическая загрузка TensorBoard логов** после каждой эпохи
- ✅ **Ручные команды** для загрузки/скачивания чекпоинтов и датасетов
- ✅ **Загрузка Docker образов** в S3 для быстрого развертывания
- ✅ **Поддержка resume** из последнего чекпоинта в S3

---

## Быстрая настройка

### 1. Создайте `.env` файл

```bash
# Timeweb S3 (рекомендуется для РФ)
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=ru-1-hot
AWS_ENDPOINT_URL=https://s3.twcstorage.ru
S3_BUCKET=your-bucket-id
S3_PREFIX=piper-training/felix_mirage

# AWS transfer optimizations
AWS_MAX_ATTEMPTS=10
AWS_RETRY_MODE=standard
AWS_S3_MAX_CONCURRENCY=10

# Training parameters
ENABLE_S3_SYNC=1  # Включить автоматическую загрузку
CHECKPOINT=s3  # Автоматически скачать последний чекпоинт из S3
BATCH_SIZE=64
PRECISION=16-mixed
```

### 2. Проверьте подключение

```bash
source .env
aws s3 ls s3://$S3_BUCKET/ --endpoint-url=$AWS_ENDPOINT_URL --region=$AWS_DEFAULT_REGION
```

### 3. Включите автосинхронизацию при обучении

```bash
export ENABLE_S3_SYNC=1
docker compose -f docker-compose.train.yml up
```

---

## Автоматическая синхронизация

### S3CheckpointCallback

Автоматически загружает чекпоинты в S3 после каждого сохранения:

```python
from piper.train.s3_callbacks import S3CheckpointCallback

trainer = Trainer(
    callbacks=[
        S3CheckpointCallback(
            s3_bucket="your-bucket",
            s3_prefix="piper-training/felix_mirage/checkpoints",
            endpoint_url="https://s3.twcstorage.ru"
        )
    ]
)
```

**Путь в S3:**
```
s3://bucket/piper-training/felix_mirage/checkpoints/
├── epoch=749-step=355500-val_loss=27.5963.ckpt
├── epoch=750-step=356000-val_loss=27.4821.ckpt
└── last.ckpt
```

### S3LogsCallback

Загружает TensorBoard логи после каждой эпохи:

```python
from piper.train.s3_callbacks import S3LogsCallback

trainer = Trainer(
    callbacks=[
        S3LogsCallback(
            s3_bucket="your-bucket",
            s3_prefix="piper-training/felix_mirage/logs",
            endpoint_url="https://s3.twcstorage.ru"
        )
    ]
)
```

**Путь в S3:**
```
s3://bucket/piper-training/felix_mirage/logs/
└── version_3/
    ├── events.out.tfevents.1705334400.hostname.12345.0
    └── hparams.yaml
```

---

## Ручные операции с S3

### script/s3_sync.sh

Скрипт для ручного управления чекпоинтами, логами и датасетами.

#### Загрузка чекпоинта

```bash
./script/s3_sync.sh upload-checkpoint lightning_logs/version_3/checkpoints/epoch=749-step=355500-val_loss=27.5963.ckpt
```

#### Список чекпоинтов

```bash
./script/s3_sync.sh list-checkpoints

# Вывод:
# Checkpoints in s3://bucket/piper-training/felix_mirage/checkpoints/:
# 2026-01-15 12:34:56  1234567890 epoch=749-step=355500-val_loss=27.5963.ckpt
# 2026-01-15 13:45:00  1234567890 last.ckpt
```

#### Скачивание чекпоинта

```bash
# Скачать конкретный чекпоинт
./script/s3_sync.sh download-checkpoint epoch=749-step=355500-val_loss=27.5963.ckpt

# Скачать последний
./script/s3_sync.sh download-latest
```

#### Загрузка логов

```bash
./script/s3_sync.sh upload-logs lightning_logs/version_3
```

#### Загрузка/скачивание датасета

```bash
# Загрузить датасет в S3
./script/s3_sync.sh upload-dataset ./felix_mirage

# Скачать датасет из S3
./script/s3_sync.sh download-dataset
```

---

## Работа с Docker образами

### script/docker_to_s3.sh

Загрузка/скачивание Docker образов через S3.

#### Экспорт образа в S3

```bash
./script/docker_to_s3.sh export

# Что происходит:
# 1. docker save ghcr.io/zudva/piper-train:latest > piper-train-latest.tar
# 2. gzip piper-train-latest.tar → piper-train-latest.tar.gz (~12 GB)
# 3. aws s3 cp → s3://bucket/piper-training/felix_mirage/docker-images/
```

#### Загрузка образа из S3

```bash
./script/docker_to_s3.sh load

# Что происходит:
# 1. aws s3 cp s3://bucket/.../piper-train-latest.tar.gz → /tmp/
# 2. gunzip /tmp/piper-train-latest.tar.gz
# 3. docker load < /tmp/piper-train-latest.tar
```

#### Список образов в S3

```bash
./script/docker_to_s3.sh list
```

---

## Resume обучения из S3

### Автоматический resume

Установите `CHECKPOINT=s3` в `.env`:

```bash
CHECKPOINT=s3
ENABLE_S3_SYNC=1
```

При старте обучения будет автоматически:
1. Скачан последний чекпоинт из S3
2. Обучение продолжено с последнего шага

### Ручной resume

```bash
# 1. Скачайте конкретный чекпоинт
./script/s3_sync.sh download-checkpoint epoch=749-step=355500-val_loss=27.5963.ckpt

# 2. Укажите путь в CHECKPOINT
export CHECKPOINT=checkpoints/epoch=749-step=355500-val_loss=27.5963.ckpt
docker compose -f docker-compose.train.yml up
```

---

## Настройка для разных провайдеров

### Timeweb Cloud (рекомендуется для РФ)

```bash
AWS_ENDPOINT_URL=https://s3.twcstorage.ru
AWS_DEFAULT_REGION=ru-1-hot
S3_BUCKET=your-bucket-uuid
```

**Как получить credentials:**
1. Зайдите в [Timeweb панель](https://timeweb.cloud/my/storage)
2. Создайте S3 хранилище
3. Получите Access Key и Secret Key
4. Скопируйте bucket ID из URL

### AWS S3

```bash
AWS_ENDPOINT_URL=https://s3.amazonaws.com
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=your-bucket-name
```

### MinIO

```bash
AWS_ENDPOINT_URL=http://localhost:9000
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=piper-training
```

---

## Оптимизация производительности

### Параллельная загрузка

```bash
# Увеличьте количество параллельных потоков
export AWS_S3_MAX_CONCURRENCY=20

# Увеличьте размер частей для multipart upload
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 16MB
```

### Retry настройки

```bash
# Увеличьте количество повторных попыток
export AWS_MAX_ATTEMPTS=20
export AWS_RETRY_MODE=adaptive
```

### Сжатие логов

TensorBoard логи автоматически загружаются без сжатия для совместимости с TensorBoard.dev. Для экономии места:

```bash
# Сжать логи перед загрузкой (ручной режим)
tar czf logs.tar.gz lightning_logs/version_3/
aws s3 cp logs.tar.gz s3://$S3_BUCKET/$S3_PREFIX/logs/
```

---

## Безопасность

### Защита credentials

```bash
# Убедитесь что .env в .gitignore
echo ".env" >> .gitignore

# Никогда не коммитьте credentials!
git add .env  # ❌ НЕ ДЕЛАЙТЕ ТАК
```

### IAM Policy (для AWS)

Минимальные права для S3 синхронизации:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket/*",
        "arn:aws:s3:::your-bucket"
      ]
    }
  ]
}
```

---

## Troubleshooting

### Ошибка "An error occurred (SignatureDoesNotMatch)"

**Проблема:** Неправильные credentials или endpoint

**Решение:**
```bash
# Проверьте credentials
source .env
echo $AWS_ACCESS_KEY_ID
echo $AWS_ENDPOINT_URL

# Проверьте подключение
aws s3 ls --endpoint-url=$AWS_ENDPOINT_URL --region=$AWS_DEFAULT_REGION
```

### Медленная загрузка чекпоинтов

**Проблема:** Большие чекпоинты (~1.2 GB) загружаются долго

**Решение:**
```bash
# Увеличьте параллелизм
export AWS_S3_MAX_CONCURRENCY=20

# Увеличьте chunk size
aws configure set default.s3.multipart_chunksize 32MB
```

### Чекпоинт не загружается автоматически

**Проверьте:**
1. `ENABLE_S3_SYNC=1` установлен
2. S3 credentials корректны
3. Логи обучения на наличие ошибок S3

```bash
docker logs piper1-gpl-train-1 | grep S3
```

---

## Примеры использования

### Локальное обучение с S3

```bash
# 1. Загрузите последний чекпоинт из S3
./script/s3_sync.sh download-latest

# 2. Запустите обучение с автосинхронизацией
export ENABLE_S3_SYNC=1
docker compose -f docker-compose.train.yml up

# 3. Чекпоинты будут автоматически загружаться в S3 каждые 5000 шагов
```

### Облачное обучение (RunPod/ImmersCloud)

```bash
# 1. На облачном сервере клонируйте репо
git clone https://github.com/Zudva/piper1-gpl.git
cd piper1-gpl

# 2. Создайте .env с S3 credentials
cat > .env << EOF
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_ENDPOINT_URL=https://s3.twcstorage.ru
S3_BUCKET=your-bucket
S3_PREFIX=piper-training/felix_mirage
ENABLE_S3_SYNC=1
CHECKPOINT=s3
BATCH_SIZE=64
PRECISION=16-mixed
EOF

# 3. Запустите обучение (автоматически скачает последний чекпоинт)
docker compose -f docker-compose.runpod.yml up -d

# 4. Мониторьте логи
docker logs -f piper1-gpl-train-1
```

### Скачивание результатов на локальную машину

```bash
# Вариант 1: Через S3
./script/s3_sync.sh list-checkpoints
./script/s3_sync.sh download-checkpoint epoch=850-step=403000-val_loss=26.1234.ckpt

# Вариант 2: Через Fabric
fab sync-from-runpod --path=lightning_logs
```

---

## Структура S3 bucket

```
s3://your-bucket/piper-training/felix_mirage/
├── checkpoints/
│   ├── epoch=749-step=355500-val_loss=27.5963.ckpt  (~1.2 GB)
│   ├── epoch=750-step=356000-val_loss=27.4821.ckpt
│   └── last.ckpt
├── logs/
│   └── version_3/
│       ├── events.out.tfevents.1705334400.hostname.12345.0
│       └── hparams.yaml
├── docker-images/
│   └── piper-train-latest.tar.gz  (~12 GB)
└── datasets/
    └── felix_mirage/
        ├── config.json
        ├── dataset.jsonl
        └── wavs/
            └── *.wav
```

---

## FAQ

**Q: Чекпоинты загружаются автоматически?**  
A: Да, если `ENABLE_S3_SYNC=1`. Загрузка происходит после каждого сохранения (каждые 5000 шагов).

**Q: Как resume обучение из S3?**  
A: Установите `CHECKPOINT=s3` — автоматически скачается последний чекпоинт.

**Q: Сколько места занимает один чекпоинт?**  
A: ~1.2 GB несжатый. В S3 хранится несжатым для быстрой загрузки.

**Q: Можно ли использовать AWS S3?**  
A: Да, измените `AWS_ENDPOINT_URL` и `AWS_DEFAULT_REGION`.

**Q: Логи TensorBoard тоже загружаются?**  
A: Да, после каждой эпохи (если `ENABLE_S3_SYNC=1`).

**Q: Как удалить старые чекпоинты из S3?**  
A: Используйте AWS CLI:
```bash
aws s3 rm s3://$S3_BUCKET/$S3_PREFIX/checkpoints/epoch=old.ckpt --endpoint-url=$AWS_ENDPOINT_URL
```

---

## См. также

- [DEPLOYMENT.md](DEPLOYMENT.md) — Деплой на RunPod/ImmersCloud
- [TRAINING.md](TRAINING.md) — Обучение новых голосов
- [script/s3_sync.sh](../script/s3_sync.sh) — Скрипт для ручных операций
- [src/piper/train/s3_callbacks.py](../src/piper/train/s3_callbacks.py) — S3 callbacks для Lightning
