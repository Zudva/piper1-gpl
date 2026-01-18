# RunPod L40S Quick Start — Максимальная скорость обучения

Быстрый старт с **L40S** ($0.71/hr) — самым быстрым GPU для обучения Piper при разумной цене.

## Почему L40S?

✅ **48 GB VRAM** → `BATCH_SIZE=80-96` (в 2-3× больше чем RTX 4090)  
✅ **62 GB RAM** → `NUM_WORKERS=4` без проблем  
✅ **~2× быстрее RTX 4090** при обучении  
✅ **$0.71/hr** → дешевле чем RTX 5090 ($0.78) и H100 ($2.39)  
✅ **Epoch за 10 часов** (vs 20 часов на RTX 4090)  

**Стоимость:** ~$17/день, ~$512/месяц, ~$7/эпоха

---

## 1. Создание Pod на RunPod

1. Откройте [RunPod Console](https://runpod.io/console/pods)
2. **Deploy** → **Pods** → **GPU Pods**
3. Выберите **L40S** (48 GB VRAM, 62 GB RAM)
4. Настройте:
   - **Template**: Docker (оставьте стандартный или выберите PyTorch)
   - **Container Disk**: **150 GB** (минимум)
   - **Volume Disk**: **50 GB** (опционально, для персистентных данных)
   - **Expose Ports**: SSH (22), TensorBoard (6006)

5. **Deploy On-Demand** или **Deploy Spot** (дешевле, но может прерваться)

---

## 2. SSH подключение

После создания Pod:

1. **Connect** → **Start SSH over exposed TCP port**
2. Скопируйте команду SSH:
   ```bash
   ssh root@XXX.XXX.XXX.XXX -p XXXXX -i ~/.ssh/id_ed25519
   ```

3. Извлеките параметры:
   ```bash
   export RUNPOD_HOST=XXX.XXX.XXX.XXX  # или ssh.runpod.io
   export RUNPOD_PORT=XXXXX
   export RUNPOD_USER=root
   ```

---

## 3. Быстрая настройка (3 команды)

```bash
# 1. Настройте подключение (вставьте свои значения)
export RUNPOD_HOST=ssh.runpod.io
export RUNPOD_PORT=12345

# 2. Первоначальная настройка
fab setup-runpod

# 3. Синхронизация кода
fab sync-to-runpod
```

---

## 4. Создание .env с S3 credentials

```bash
# SSH в pod
fab ssh-runpod

# Создайте .env
cat > /workspace/piper1-gpl/.env << 'EOF'
# Timeweb S3
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_ENDPOINT_URL=https://s3.twcstorage.ru
AWS_DEFAULT_REGION=ru-1-hot
S3_BUCKET=your-bucket-id
S3_PREFIX=piper-training/felix_mirage

# S3 sync
ENABLE_S3_SYNC=1
CHECKPOINT=s3

# L40S optimized settings (48GB VRAM, 62GB RAM)
BATCH_SIZE=80
NUM_WORKERS=4
NUM_DEVICES=1
PRECISION=16-mixed
ACCUM=1
MAX_EPOCHS=10000

# AWS tuning
AWS_MAX_ATTEMPTS=10
AWS_RETRY_MODE=standard
AWS_S3_MAX_CONCURRENCY=10
EOF

exit
```

---

## 5. Загрузка датасета из S3

```bash
fab ssh-runpod

cd /workspace/piper1-gpl

# Скачать датасет из S3 (если уже загружен)
./script/s3_sync.sh download-dataset

# Или загрузите в S3 с локальной машины (один раз)
# Локально: ./script/s3_sync.sh upload-dataset /path/to/felix_mirage
```

---

## 6. Запуск обучения

```bash
# Вариант 1: Через Fabric (рекомендуется)
fab start-training --batch-size=80

# Вариант 2: Прямо в pod
fab ssh-runpod

docker compose -f deploy/compose/docker-compose.runpod.yml up -d

# Проверить логи
docker logs -f piper1-gpl-train-1
```

---

## 7. Мониторинг

```bash
# Логи в реальном времени
fab ssh-runpod --cmd="docker logs -f piper1-gpl-train-1"

# GPU utilization (должно быть 95-100%)
fab ssh-runpod --cmd="watch -n 1 nvidia-smi"

# RAM usage (должно быть ~40-50 GB из 62 GB)
fab ssh-runpod --cmd="free -h"

# Docker stats
fab ssh-runpod --cmd="docker stats piper1-gpl-train-1"

# Список чекпоинтов в S3
./script/s3_sync.sh list-checkpoints
```

### Ожидаемая производительность L40S:

```
GPU Utilization: 95-100%
RAM Usage: ~45 GB / 62 GB
VRAM Usage: ~40 GB / 48 GB
Steps/sec: ~2.5-3.0 (vs ~1.2-1.5 на RTX 4090)
Epoch time: ~10 hours (vs ~20 hours на RTX 4090)
```

---

## 8. TensorBoard (опционально)

```bash
# На RunPod pod
fab ssh-runpod

tensorboard --logdir /workspace/piper1-gpl/lightning_logs --host 0.0.0.0 --port 6006

# Локально (в браузере)
# Откройте http://RUNPOD_HOST:6006
```

---

## 9. Остановка и возобновление

### Остановка:

```bash
fab ssh-runpod --cmd="docker compose -f /workspace/piper1-gpl/deploy/compose/docker-compose.runpod.yml down"
```

### Возобновление:

```bash
# Чекпоинт автоматически загрузится из S3
fab ssh-runpod --cmd="docker compose -f /workspace/piper1-gpl/deploy/compose/docker-compose.runpod.yml up -d"
```

---

## 10. Скачивание результатов

```bash
# Вариант 1: Из S3 (автоматически загружается)
./script/s3_sync.sh list-checkpoints
./script/s3_sync.sh download-checkpoint epoch=850-step=403000-val_loss=26.1234.ckpt

# Вариант 2: Через rsync
fab sync-from-runpod --path=lightning_logs
```

---

## Оптимизация для максимальной скорости

### 1. Увеличьте BATCH_SIZE до предела

```bash
# Попробуйте максимальный batch size
export BATCH_SIZE=96
fab start-training --batch-size=96

# Если OOM → уменьшите на 10-20%
export BATCH_SIZE=80
```

### 2. Попробуйте bf16-mixed (может быть быстрее)

```bash
export PRECISION=bf16-mixed
fab start-training --batch-size=80 --precision=bf16-mixed

# Если ошибки cuFFT → вернитесь на 16-mixed
export PRECISION=16-mixed
```

### 3. Увеличьте NUM_WORKERS (если CPU не bottleneck)

```bash
export NUM_WORKERS=6  # было 4
# Проверьте CPU usage - должно быть <90%
fab ssh-runpod --cmd="top"
```

### 4. Используйте tmpfs для кеша (если достаточно RAM)

```yaml
# В deploy/compose/docker-compose.runpod.yml добавьте:
volumes:
  - type: tmpfs
    target: /data/.cache
    tmpfs:
      size: 20G  # RAM-based cache
```

---

## Troubleshooting

### OOM (Out of Memory) в VRAM

```bash
# Уменьшите batch size
export BATCH_SIZE=64  # было 80

# Или включите gradient accumulation
export BATCH_SIZE=40
export ACCUM=2  # эффективный BS = 40 × 2 = 80
```

### Медленная загрузка данных

```bash
# Увеличьте num_workers
export NUM_WORKERS=6

# Проверьте что датасет на быстром диске (не NFS)
fab ssh-runpod --cmd="df -h /data"
```

### S3 upload failed

```bash
# Увеличьте retry
export AWS_MAX_ATTEMPTS=20

# Проверьте credentials
fab ssh-runpod --cmd="aws s3 ls s3://your-bucket/ --endpoint-url=https://s3.twcstorage.ru"
```

---

## Стоимость и время обучения

### От epoch=749 до epoch=1000 (251 эпоха):

**L40S @ $0.71/hr:**
```
Время: 251 эпохи × 10 часов = 2510 часов = ~104 дня непрерывно
НО: обычно останавливаем раньше при достижении целевого val_loss
```

**Реалистичный сценарий (100 эпох):**
```
Время: 100 × 10 = 1000 часов = ~42 дня
Стоимость: 1000 × $0.71 = $710
```

**Сравнение с RTX 4090:**
```
L40S: 100 эпох × 10 часов × $0.71 = $710
RTX 4090: 100 эпох × 20 часов × $0.50 = $1000
→ L40S на 29% дешевле при 2× скорости! ✅
```

---

## Полный пример команд (копипаста)

```bash
# 1. Настройка подключения
export RUNPOD_HOST=ssh.runpod.io
export RUNPOD_PORT=12345

# 2. Первоначальная настройка
fab setup-runpod
fab sync-to-runpod

# 3. Создайте .env (SSH в pod, скопируйте блок из шага 4)
fab ssh-runpod
# ... создайте .env ...
exit

# 4. Загрузите датасет
fab ssh-runpod --cmd="cd /workspace/piper1-gpl && ./script/s3_sync.sh download-dataset"

# 5. Запустите обучение
fab start-training --batch-size=80

# 6. Мониторинг
fab ssh-runpod --cmd="docker logs -f piper1-gpl-train-1"
```

---

## Best Practices для L40S

1. **Всегда используйте S3 sync** — автосохранение каждые 5000 шагов
2. **Мониторьте GPU utilization** — должно быть 95-100%, если меньше → увеличьте NUM_WORKERS
3. **Используйте Spot instances** — на 30-40% дешевле, но могут прерваться
4. **Backup в S3** — при прерывании Spot обучение продолжится с последнего чекпоинта
5. **TensorBoard** — запускайте локально после sync-from-runpod, экономит VRAM

---

## Сравнение с другими GPU

| GPU | $/час | Эпоха (часы) | $/эпоха | Скорость | RAM |
|-----|-------|--------------|---------|----------|-----|
| L4 | $0.32 | 20 | $6.40 | 1.0× | 55 GB ✅ |
| RTX 4090 | $0.50 | 18 | $9.00 | 1.1× | 31 GB ⚠️ |
| RTX 5090 | $0.78 | 12 | $9.36 | 1.6× | 92 GB ✅ |
| **L40S** | **$0.71** | **10** | **$7.10** | **2.0×** | **62 GB ✅** |
| H100 PCIe | $2.03 | 6 | $12.18 | 3.5× | 176 GB 💎 |

**Вывод:** L40S — лучший баланс скорость/цена для серьезного обучения! 🏆

---

## См. также

- [DEPLOYMENT.md](DEPLOYMENT.md) — Общее руководство по деплою
- [S3_INTEGRATION.md](S3_INTEGRATION.md) — S3 синхронизация
- [LOW_RAM_OPTIMIZATION.md](LOW_RAM_OPTIMIZATION.md) — Оптимизация для low-RAM (не нужно для L40S!)
