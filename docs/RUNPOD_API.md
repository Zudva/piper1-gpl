# RunPod API Integration

Автоматизация создания и управления RunPod pods через GraphQL API.

## Получение API ключа

1. Зайдите в [RunPod Settings](https://www.runpod.io/console/user/settings)
2. **API Keys** → **+ API Key** → **Read & Write**
3. Скопируйте ключ

```bash
export RUNPOD_API_KEY="your_api_key_here"

# Или добавьте в .env
echo "RUNPOD_API_KEY=your_key" >> .env
```

## Установка зависимостей

```bash
pip install requests
```

## Использование

### 1. Создание Pod с L40S (автоматически)

```bash
# Создать pod с настройками из .env
python script/runpod_api.py create --gpu L40S --name piper-training

# Или с кастомными параметрами
python script/runpod_api.py create \
  --gpu L40S \
  --count 1 \
  --name piper-training-749 \
  --container-disk 150 \
  --volume-disk 100 \
  --image ghcr.io/zudva/piper-train:latest \
  --spot

# On-demand (дороже, но стабильнее)
python script/runpod_api.py create --gpu L40S --on-demand
```

**Вывод:**
```
✅ Pod created: abc123xyz
Name: piper-training
GPU: NVIDIA L40S

Get SSH command: python script/runpod_api.py ssh abc123xyz
```

### 2. Список всех pods

```bash
python script/runpod_api.py list
```

**Вывод:**
```
ID                             Name                 GPU                            Uptime
----------------------------------------------------------------------------------------------------
abc123xyz                      piper-training       NVIDIA L40S                    2h 15m
def456uvw                      test-pod             NVIDIA RTX 4090                0h 45m
```

### 3. Получить SSH команду

```bash
python script/runpod_api.py ssh abc123xyz
```

**Вывод:**
```
SSH command:
ssh root@123.45.67.89 -p 12345

Or set env vars:
export RUNPOD_HOST=123.45.67.89
export RUNPOD_PORT=12345
```

### 4. Детали pod

```bash
python script/runpod_api.py get abc123xyz
```

### 5. Остановить pod (сохраняет данные)

```bash
python script/runpod_api.py stop abc123xyz
```

### 6. Удалить pod (полностью)

```bash
python script/runpod_api.py terminate abc123xyz
```

---

## Автоматизированный workflow

### Полный цикл через API:

```bash
# 1. Создайте .env с credentials
cat > .env << 'EOF'
RUNPOD_API_KEY=your_runpod_api_key
AWS_ACCESS_KEY_ID=your_s3_key
AWS_SECRET_ACCESS_KEY=your_s3_secret
AWS_ENDPOINT_URL=https://s3.twcstorage.ru
AWS_DEFAULT_REGION=ru-1-hot
S3_BUCKET=your-bucket-id
S3_PREFIX=piper-training/felix_mirage
ENABLE_S3_SYNC=1
CHECKPOINT=s3
BATCH_SIZE=80
NUM_WORKERS=4
NUM_DEVICES=1
PRECISION=16-mixed
MAX_EPOCHS=1000
EOF

# 2. Создайте pod (env vars из .env автоматически загрузятся)
POD_ID=$(python script/runpod_api.py create --gpu L40S --name piper-749 | grep "Pod created:" | awk '{print $4}')
echo "Created pod: $POD_ID"

# 3. Получите SSH credentials
python script/runpod_api.py ssh $POD_ID

# 4. Установите env vars из вывода
export RUNPOD_HOST=...
export RUNPOD_PORT=...

# 5. Настройте и запустите обучение
fab setup-runpod
fab sync-to-runpod
fab start-training --batch-size=80

# 6. Мониторинг
fab ssh-runpod --cmd="docker logs -f piper1-gpl-train-1"

# 7. Когда закончите - остановите pod
python script/runpod_api.py stop $POD_ID
```

---

## Интеграция с Fabric

Добавим команды в `fabfile.py`:

```python
@task
def runpod_create(c, gpu="L40S", name="piper-training"):
    """Create RunPod pod via API."""
    result = c.run(f"python script/runpod_api.py create --gpu {gpu} --name {name}", pty=True)
    # Parse pod ID from output
    # Set RUNPOD_HOST and RUNPOD_PORT

@task
def runpod_list(c):
    """List RunPod pods."""
    c.run("python script/runpod_api.py list", pty=True)

@task  
def runpod_ssh_info(c, pod_id):
    """Get SSH info for pod."""
    c.run(f"python script/runpod_api.py ssh {pod_id}", pty=True)
```

**Использование:**
```bash
fab runpod-create --gpu=L40S
fab runpod-list
fab runpod-ssh-info --pod-id=abc123
```

---

## Доступные GPU типы

| Тип | Код | VRAM | RAM | $/час (spot) |
|-----|-----|------|-----|--------------|
| L4 | `L4` | 24 GB | 55 GB | $0.32 |
| L40S | `L40S` | 48 GB | 62 GB | $0.71 |
| RTX 4090 | `RTX4090` | 24 GB | 31 GB | $0.50 |
| RTX 5090 | `RTX5090` | 32 GB | 92 GB | $0.78 |
| A100 | `A100` | 80 GB | 128 GB | $1.50 |
| H100 | `H100` | 80 GB | 176 GB | $2.03 |

---

## Environment Variables передаваемые в pod

При создании pod через API, переменные из `.env` автоматически передаются:

```python
# Скрипт читает .env и передает все переменные в pod
env_vars = {
    "AWS_ACCESS_KEY_ID": "...",
    "AWS_SECRET_ACCESS_KEY": "...",
    "BATCH_SIZE": "80",
    # ... и т.д.
}
```

**Безопасность:** Credentials передаются через API, но не сохраняются в RunPod template.

---

## Troubleshooting

### API Key не работает

```bash
# Проверьте что ключ установлен
echo $RUNPOD_API_KEY

# Проверьте права ключа (должен быть Read & Write)
python script/runpod_api.py list
```

### Pod не создается

```bash
# Проверьте доступность GPU
python script/runpod_api.py create --gpu L40S

# Попробуйте другой GPU
python script/runpod_api.py create --gpu RTX4090
```

### GPU type ID неправильный

Скрипт использует упрощенные имена. Если нужны реальные GPU type IDs:

```bash
# Получите список через RunPod API
curl -X POST https://api.runpod.io/graphql \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{"query": "query { gpuTypes { id displayName } }"}'
```

---

## Примеры

### Быстрый старт для epoch 749→850

```bash
# 1. Создать pod
python script/runpod_api.py create --gpu L40S --name piper-749-850

# 2. Дождаться создания (1-2 минуты)
sleep 120

# 3. Получить SSH info
POD_ID="abc123"  # из вывода create
python script/runpod_api.py ssh $POD_ID

# 4. Экспортировать credentials
export RUNPOD_HOST=...
export RUNPOD_PORT=...

# 5. Автоматический деплой
fab setup-runpod sync-to-runpod start-training

# 6. Проверка статуса через API
python script/runpod_api.py get $POD_ID
```

### Multi-GPU pod

```bash
python script/runpod_api.py create \
  --gpu L40S \
  --count 2 \
  --name piper-2xL40S \
  --container-disk 200

# В .env установите:
# NUM_DEVICES=2
# BATCH_SIZE=40  # per-GPU
```

---

## См. также

- [RunPod API Docs](https://docs.runpod.io/reference/graphql-api)
- [RUNPOD_L40S_QUICKSTART.md](RUNPOD_L40S_QUICKSTART.md)
- [DEPLOYMENT.md](DEPLOYMENT.md)

---

## Ограничения текущей реализации

⚠️ **TODO:**
- Получение реальных GPU Type IDs через API (сейчас используются названия)
- Автоматическое ожидание готовности pod после создания
- Интеграция с `fabfile.py` для one-command deployment
- Resume после Spot interruption

**Вклады приветствуются!** 🚀
