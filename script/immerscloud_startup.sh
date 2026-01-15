#!/bin/bash
# ImmersCloud startup script: setup environment and start training
# Place this in ImmersCloud instance startup scripts

set -euo pipefail

echo "=== ImmersCloud Piper Training Setup ==="

WORKSPACE="/workspace/piper1-gpl"
REPO_URL="https://github.com/Zudva/piper1-gpl.git"

# Install dependencies if needed
if ! command -v git &> /dev/null; then
    echo "Installing git..."
    apt-get update && apt-get install -y git rsync
fi

# Clone or update repository
if [ ! -d "$WORKSPACE/.git" ]; then
    echo "📦 Cloning repository..."
    git clone "$REPO_URL" "$WORKSPACE"
else
    echo "📦 Updating repository..."
    cd "$WORKSPACE" && git pull origin main
fi

cd "$WORKSPACE"

# Download .env from S3 (contains credentials)
if [ -f "script/s3_sync.sh" ]; then
    echo "📥 Downloading .env from S3..."
    ./script/s3_sync.sh download-env || echo "⚠️ No .env in S3, using system env"
fi

# Install AWS CLI if needed
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    pip install awscli boto3 s3fs
fi

# Download Docker image from S3 if specified
if [ "${USE_S3_IMAGE:-0}" = "1" ]; then
    echo "📥 Loading Docker image from S3..."
    ./script/docker_to_s3.sh load
fi

# Start training
echo "🚀 Starting training..."
ENABLE_S3_SYNC=1 docker compose -f docker-compose.immerscloud.yml up -d

echo "✓ Training started. Monitor with:"
echo "  docker logs -f piper1-gpl-train-1"
echo "  docker compose -f docker-compose.immerscloud.yml logs -f"
