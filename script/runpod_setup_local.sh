#!/bin/bash
# RunPod Setup Script - запустите на локальной машине
# Этот скрипт подключится к RunPod и настроит всё автоматически

set -e

RUNPOD_SSH="7q78ektzn8qnzr-6441186d@ssh.runpod.io"
SSH_KEY="~/.ssh/id_ed25519"

echo "🚀 Starting RunPod setup..."
echo "Connecting to: $RUNPOD_SSH"

# Подключаемся и выполняем setup
ssh -i $SSH_KEY $RUNPOD_SSH bash -s << 'REMOTE_SCRIPT'
set -e

echo "✅ Connected to RunPod"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /workspace

# Clone or update repo
if [ ! -d "piper1-gpl" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/Zudva/piper1-gpl.git
else
    echo "📥 Updating repository..."
    cd piper1-gpl
    git pull
    cd /workspace
fi

cd /workspace/piper1-gpl

echo "📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq build-essential cmake ninja-build espeak-ng rsync > /dev/null

echo "🐍 Installing Python packages..."
pip install --upgrade pip -q
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121 -q
pip install scikit-build -q
pip install -e .[train] -q

echo "🔨 Building monotonic align..."
chmod +x build_monotonic_align.sh
./build_monotonic_align.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Create .env file with your S3 credentials"
echo "2. Download dataset from S3"
echo "3. Start training"
echo ""

REMOTE_SCRIPT

echo ""
echo "🎉 RunPod setup finished!"
echo ""
echo "Now run this to connect:"
echo "  ssh $RUNPOD_SSH -i $SSH_KEY"
