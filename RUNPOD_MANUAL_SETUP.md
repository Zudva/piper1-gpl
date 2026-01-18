# Пошаговая настройка RunPod вручную

## 1. Подключитесь к RunPod:

```bash
ssh <user>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

## 2. После подключения выполните команды по порядку:

### Проверка GPU:
```bash
nvidia-smi
```
Должны увидеть **NVIDIA L40S** с **48 GB VRAM** ✅

### Клонирование репозитория:
```bash
cd /workspace
git clone https://github.com/Zudva/piper1-gpl.git
cd piper1-gpl
```

### Установка зависимостей:
```bash
apt-get update
apt-get install -y build-essential cmake ninja-build espeak-ng rsync
```

### Установка Python пакетов:
```bash
pip install --upgrade pip
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install scikit-build boto3 s3fs awscli
pip install -e .[train]
```

### Сборка monotonic align:
```bash
chmod +x build_monotonic_align.sh
./build_monotonic_align.sh
```

## 3. Создайте .env файл:

```bash
cat > .env << 'EOF'
AWS_ACCESS_KEY_ID=ВАШ_КЛЮЧ
AWS_SECRET_ACCESS_KEY=ВАШ_СЕКРЕТ
AWS_ENDPOINT_URL=https://s3.twcstorage.ru
AWS_DEFAULT_REGION=ru-1-hot
S3_BUCKET=ВАШ_BUCKET
S3_PREFIX=piper-training/felix_mirage

ENABLE_S3_SYNC=1
CHECKPOINT=s3
BATCH_SIZE=80
NUM_WORKERS=4
NUM_DEVICES=1
PRECISION=16-mixed
MAX_EPOCHS=1000

AWS_MAX_ATTEMPTS=10
AWS_RETRY_MODE=standard
EOF
```

## 4. Скачайте датасет из S3:

```bash
./script/s3_sync.sh download-dataset
```

Или создайте символическую ссылку если датасет уже есть на volume.

## 5. Запустите обучение:

Если вы **уже внутри training-контейнера** (типичный сценарий, Docker-in-Docker недоступен), запускайте напрямую:

```bash
bash check.sh
bash train.sh
```

Если у вас есть доступ к Docker daemon (не внутри контейнера), тогда можно использовать docker-compose сценарии из `docs/DEPLOYMENT.md`.

## 6. Мониторинг:

```bash
# GPU
watch -n 1 nvidia-smi

# Чекпоинты
./script/s3_sync.sh list-checkpoints
```

---

## Копипаста для быстрого выполнения (всё сразу):

```bash
cd /workspace && \
git clone https://github.com/Zudva/piper1-gpl.git && \
cd piper1-gpl && \
apt-get update && apt-get install -y build-essential cmake ninja-build espeak-ng rsync && \
pip install --upgrade pip && \
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121 && \
pip install scikit-build boto3 s3fs awscli && \
pip install -e .[train] && \
chmod +x build_monotonic_align.sh && ./build_monotonic_align.sh && \
echo "✅ Setup complete! Now create .env and start training."
```

После этого создайте `.env` и запустите обучение!
