#!/bin/bash
# RunPod entrypoint: download Docker image from S3 if needed, then start training
set -euo pipefail

echo "=== Piper Training on RunPod (S3-backed) ==="

# Load S3 image if not already loaded
IMAGE_NAME="${DOCKER_IMAGE:-ghcr.io/zudva/piper-train:latest}"
if ! docker images "$IMAGE_NAME" | grep -q "$IMAGE_NAME"; then
    echo "📥 Image not found locally, downloading from S3..."
    
    if [ -f "/workspace/script/docker_to_s3.sh" ]; then
        /workspace/script/docker_to_s3.sh load
    else
        echo "❌ Error: docker_to_s3.sh not found"
        echo "Manual load: aws s3 cp s3://\$S3_BUCKET/\$S3_PREFIX/docker-images/piper-train-latest.tar.gz - | gunzip | docker load"
        exit 1
    fi
else
    echo "✓ Image already loaded: $IMAGE_NAME"
fi

# Now run the actual training container
echo "🚀 Starting training container..."
exec docker run --rm --gpus all \
    --env-file /workspace/.env \
    -v /workspace:/workspace \
    "$IMAGE_NAME" \
    "$@"
