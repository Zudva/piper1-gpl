# Оптимизация для серверов с низким RAM

Руководство по настройке обучения Piper на серверах с ограниченной оперативной памятью (ImmersCloud, бюджетные GPU серверы).

## Проблема

На некоторых GPU платформах доступно очень мало RAM:
- **ImmersCloud**: 32 GB RAM на 1 GPU, 44 GB RAM на 4 GPU
- **Бюджетные провайдеры**: 32-48 GB RAM

При стандартной конфигурации обучение может падать с **OOM (Out of Memory)** в системной памяти (не VRAM!).

## Почему не хватает RAM?

### Потребление RAM при обучении:

1. **Системная память**: 4-8 GB (Ubuntu, Docker, SSH)
2. **PyTorch runtime**: 8-12 GB (CUDA libs, PyTorch core)
3. **DataLoader workers**: 
   - `num_workers=1`: ~2-4 GB
   - `num_workers=4`: ~8-16 GB ⚠️
4. **Batch данных**: зависит от `BATCH_SIZE`
   - `BATCH_SIZE=64`: ~10-15 GB
   - `BATCH_SIZE=16`: ~4-6 GB
5. **Multi-GPU (DDP)**: каждый GPU процесс дублирует память
   - 4 GPU × 10 GB = **40 GB только на процессы!**

### Критические сценарии:

**4 GPU + 44 GB RAM:**
```
44 GB / 4 GPU процесса = 11 GB на процесс
```
При стандартной конфигурации (`num_workers=4`, `BATCH_SIZE=64`):
```
Системная память: 8 GB
PyTorch per process: 3 GB × 4 = 12 GB
DataLoader workers: 4 GB × 4 = 16 GB
Batch data: 4 GB × 4 = 16 GB
──────────────────────────────────
ИТОГО: 52 GB (нужно) > 44 GB (есть) ❌
```

## Оптимизированные конфигурации

### 1 GPU + 32 GB RAM (ImmersCloud)

**Рекомендуемая конфигурация:**
```bash
export NUM_DEVICES=1
export BATCH_SIZE=24
export NUM_WORKERS=1
export PRECISION=16-mixed
export ACCUM=1
```

**Потребление RAM:**
```
Системная память: 6 GB
PyTorch runtime: 10 GB
DataLoader (1 worker): 3 GB
Batch data (BS=24): 6 GB
Буферы: 4 GB
──────────────────────
ИТОГО: ~29 GB ✅
```

**Эффективный batch size:** 24

### 4 GPU + 44 GB RAM (ImmersCloud)

**Рекомендуемая конфигурация:**
```bash
export NUM_DEVICES=4
export BATCH_SIZE=16      # per-GPU
export NUM_WORKERS=1      # КРИТИЧНО!
export PRECISION=16-mixed
export ACCUM=1
```

**Потребление RAM:**
```
Системная память: 6 GB
PyTorch runtime: 2 GB × 4 = 8 GB
DataLoader (1 worker): 2 GB × 4 = 8 GB
Batch data (BS=16): 4 GB × 4 = 16 GB
Буферы: 4 GB
──────────────────────────────────
ИТОГО: ~42 GB ✅
```

**Эффективный batch size:** 16 × 4 = **64**

### Агрессивная оптимизация (4 GPU + 44 GB RAM)

Если всё равно OOM:

```bash
export NUM_DEVICES=4
export BATCH_SIZE=12      # уменьшить per-GPU BS
export NUM_WORKERS=1
export PRECISION=16-mixed
export ACCUM=2            # компенсировать через accumulation
```

**Эффективный batch size:** 12 × 4 × 2 = **96**

## Docker Compose конфигурации

### ImmersCloud 1 GPU

`deploy/compose/docker-compose.immerscloud.yml`:
```yaml
environment:
  - BATCH_SIZE=24
  - NUM_WORKERS=1
  - NUM_DEVICES=1
  - PRECISION=16-mixed
  - ACCUM=1
```

### ImmersCloud 4 GPU

```yaml
environment:
  - BATCH_SIZE=16
  - NUM_WORKERS=1
  - NUM_DEVICES=4
  - PRECISION=16-mixed
  - ACCUM=1
```

## Параметры оптимизации

### 1. num_workers (самый важный!)

**Что делает:** Количество процессов для загрузки данных

**Потребление RAM:**
- `num_workers=1`: ~2-3 GB на GPU процесс
- `num_workers=4`: ~8-12 GB на GPU процесс

**Рекомендация для low-RAM:**
```bash
--data.num_workers=1  # ВСЕГДА ставьте 1 при low RAM!
```

**Влияние на скорость:**
- `num_workers=1`: DataLoader может стать узким местом (~5-10% медленнее)
- `num_workers=4`: Оптимально, но требует RAM

**Компромисс:** На современных NVMe SSD `num_workers=1` даёт приемлемую скорость.

### 2. BATCH_SIZE

**Что делает:** Количество примеров в одном батче

**Потребление RAM:** ~0.3-0.5 GB на пример

**Рекомендации:**
- **1 GPU + 32 GB RAM**: `BATCH_SIZE=24-32`
- **4 GPU + 44 GB RAM**: `BATCH_SIZE=12-16` (per-GPU)

### 3. ACCUM (gradient accumulation)

**Что делает:** Накапливает градиенты N шагов перед обновлением весов

**Потребление RAM:** Минимальное (~100-200 MB)

**Преимущества:**
- Эффективный batch size = `BATCH_SIZE × num_GPU × ACCUM`
- Не увеличивает потребление RAM (почти)

**Когда использовать:**
```bash
# Если уменьшили BATCH_SIZE из-за RAM, компенсируйте через ACCUM
BATCH_SIZE=12 ACCUM=2  # эффективно = 24 на GPU
BATCH_SIZE=12 ACCUM=4  # эффективно = 48 на GPU
```

### 4. PRECISION

**Что делает:** Точность вычислений (float16 vs float32)

**Потребление RAM:**
- `16-mixed`: ~50% экономии RAM и VRAM
- `32`: Полная точность, но 2× больше памяти

**Рекомендация:** Всегда используйте `16-mixed` для low-RAM.

## Проверка потребления RAM

### Во время обучения

```bash
# SSH в контейнер
fab ssh-immerscloud

# Мониторинг RAM
watch -n 1 free -h

# Детальный мониторинг процессов
watch -n 1 'ps aux --sort=-%mem | head -20'

# Мониторинг конкретного процесса
docker stats piper1-gpl-train-1
```

### Признаки OOM

**Логи Docker:**
```
Killed
OOM killed process
Out of memory
```

**Система:**
```bash
# Проверить kernel logs
dmesg | grep -i oom
dmesg | tail -50
```

## Swap память (emergency solution)

Если всё равно не хватает RAM, можно добавить swap:

```bash
# На сервере (ImmersCloud)
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Проверить
free -h
```

⚠️ **Внимание:** Swap очень медленный, используйте только как временное решение.

## Сравнение конфигураций

| Конфигурация | RAM | Эфф. BS | Скорость | Рекомендация |
|--------------|-----|---------|----------|--------------|
| 1 GPU, BS=32, NW=4 | 35 GB | 32 | 100% | ❌ OOM на 32GB |
| 1 GPU, BS=24, NW=1 | 29 GB | 24 | 95% | ✅ Оптимально для 32GB |
| 4 GPU, BS=16, NW=4 | 52 GB | 64 | 100% | ❌ OOM на 44GB |
| 4 GPU, BS=16, NW=1 | 42 GB | 64 | 95% | ✅ Оптимально для 44GB |
| 4 GPU, BS=12, NW=1, AC=2 | 38 GB | 96 | 95% | ✅ Лучший вариант! |

**BS** = BATCH_SIZE, **NW** = NUM_WORKERS, **AC** = ACCUM, **Эфф. BS** = эффективный batch size

## Fabric команды для ImmersCloud

### Запуск с оптимизированными параметрами

```bash
# 1 GPU
export IMMERSCLOUD_HOST=your-instance.com
export NUM_DEVICES=1
export BATCH_SIZE=24
export NUM_WORKERS=1

fab sync-to-immerscloud
fab start-training-immerscloud

# 4 GPU с gradient accumulation
export NUM_DEVICES=4
export BATCH_SIZE=12
export NUM_WORKERS=1
export ACCUM=2

fab start-training-immerscloud
```

### Мониторинг

```bash
# Логи
fab ssh-immerscloud --cmd="docker logs -f piper1-gpl-train-1"

# RAM usage
fab ssh-immerscloud --cmd="free -h"
fab ssh-immerscloud --cmd="docker stats piper1-gpl-train-1"
```

## Troubleshooting

### OOM после старта обучения

**Симптомы:**
- Процесс killed через 1-5 минут
- `dmesg` показывает OOM killer

**Решение:**
```bash
# Уменьшите num_workers
export NUM_WORKERS=1

# Уменьшите batch size
export BATCH_SIZE=12  # было 16

# Добавьте gradient accumulation
export ACCUM=2
```

### Медленная загрузка данных

**Симптомы:**
- GPU utilization <80%
- `nvidia-smi` показывает низкую загрузку

**Решение:**
```bash
# Попробуйте увеличить num_workers (если есть RAM)
export NUM_WORKERS=2

# Или переместите датасет на NVMe SSD
# Или используйте tmpfs для кеша
docker run -v /dev/shm:/data/.cache ...
```

### Не хватает VRAM (не RAM!)

**Симптомы:**
```
CUDA out of memory
torch.cuda.OutOfMemoryError
```

**Решение:**
```bash
# Уменьшите BATCH_SIZE
export BATCH_SIZE=12

# Включите gradient accumulation
export ACCUM=2

# Используйте 16-mixed precision
export PRECISION=16-mixed
```

## Best Practices

1. **ВСЕГДА** используйте `NUM_WORKERS=1` на low-RAM серверах
2. **Начинайте** с малого `BATCH_SIZE` и увеличивайте постепенно
3. **Используйте** `ACCUM` для компенсации малого batch size
4. **Мониторьте** RAM через `docker stats` в реальном времени
5. **Проверяйте** `dmesg` после каждого OOM
6. **Используйте** S3 для автосохранения — если упадёт, не потеряете чекпоинт

## Примеры команд

### ImmersCloud 1 GPU (32 GB RAM)

```bash
export IMMERSCLOUD_HOST=gpu1.immerscloud.com
export NUM_DEVICES=1
export BATCH_SIZE=24
export NUM_WORKERS=1
export PRECISION=16-mixed

fab sync-to-immerscloud
fab start-training-immerscloud
```

### ImmersCloud 4 GPU (44 GB RAM) — агрессивная оптимизация

```bash
export IMMERSCLOUD_HOST=gpu4.immerscloud.com
export NUM_DEVICES=4
export BATCH_SIZE=12
export NUM_WORKERS=1
export PRECISION=16-mixed
export ACCUM=2

fab sync-to-immerscloud
fab start-training-immerscloud
```

## См. также

- [DEPLOYMENT.md](DEPLOYMENT.md) — Общее руководство по деплою
- [S3_INTEGRATION.md](S3_INTEGRATION.md) — S3 синхронизация
- [deploy/compose/docker-compose.immerscloud.yml](../deploy/compose/docker-compose.immerscloud.yml) — Конфигурация для ImmersCloud
