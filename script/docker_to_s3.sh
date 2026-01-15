#!/bin/bash
# Export Docker image to S3 and load it on remote machines (e.g., RunPod)
# Usage:
#   ./docker_to_s3.sh export   - Export local image to tar.gz and upload to S3
#   ./docker_to_s3.sh load     - Download from S3 and load into Docker

set -euo pipefail

# Load .env if present
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

: "${AWS_ACCESS_KEY_ID:?Missing AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Missing AWS_SECRET_ACCESS_KEY}"
: "${AWS_ENDPOINT_URL:?Missing AWS_ENDPOINT_URL}"
: "${S3_BUCKET:?Missing S3_BUCKET}"
: "${S3_PREFIX:?Missing S3_PREFIX}"

AWS_REGION="${AWS_DEFAULT_REGION:-ru-1-hot}"
AWS_CLI="aws --endpoint-url=$AWS_ENDPOINT_URL --region=$AWS_REGION"

IMAGE_NAME="${DOCKER_IMAGE:-ghcr.io/zudva/piper-train:latest}"
ARCHIVE_NAME="piper-train-latest.tar.gz"
S3_KEY="$S3_PREFIX/docker-images/$ARCHIVE_NAME"

ACTION="${1:-help}"

export_to_s3() {
    echo "📦 Exporting Docker image: $IMAGE_NAME"
    
    if [ -f "$ARCHIVE_NAME" ]; then
        echo "⚠️  Archive already exists: $ARCHIVE_NAME"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping export, using existing archive"
        else
            rm "$ARCHIVE_NAME"
            echo "Creating archive (this may take 5-10 minutes for 25GB image)..."
            docker save "$IMAGE_NAME" | gzip > "$ARCHIVE_NAME"
        fi
    else
        echo "Creating archive (this may take 5-10 minutes for 25GB image)..."
        docker save "$IMAGE_NAME" | gzip > "$ARCHIVE_NAME"
    fi
    
    ARCHIVE_SIZE=$(du -h "$ARCHIVE_NAME" | cut -f1)
    echo "✓ Archive created: $ARCHIVE_NAME ($ARCHIVE_SIZE)"
    
    echo "📤 Uploading to S3: s3://$S3_BUCKET/$S3_KEY"
    $AWS_CLI s3 cp "$ARCHIVE_NAME" "s3://$S3_BUCKET/$S3_KEY" \
        --storage-class STANDARD
    
    echo "✓ Upload complete!"
    echo ""
    echo "To load on RunPod or remote machine:"
    echo "  curl -o $ARCHIVE_NAME '${AWS_ENDPOINT_URL}/${S3_BUCKET}/${S3_KEY}'"
    echo "  gunzip -c $ARCHIVE_NAME | docker load"
}

load_from_s3() {
    echo "📥 Downloading from S3: s3://$S3_BUCKET/$S3_KEY"
    $AWS_CLI s3 cp "s3://$S3_BUCKET/$S3_KEY" "$ARCHIVE_NAME"
    
    ARCHIVE_SIZE=$(du -h "$ARCHIVE_NAME" | cut -f1)
    echo "✓ Downloaded: $ARCHIVE_NAME ($ARCHIVE_SIZE)"
    
    echo "🐳 Loading image into Docker..."
    gunzip -c "$ARCHIVE_NAME" | docker load
    
    echo "✓ Image loaded: $IMAGE_NAME"
    echo ""
    echo "Cleanup archive? (y/N): "
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm "$ARCHIVE_NAME"
        echo "✓ Archive removed"
    fi
}

list_s3_images() {
    echo "Docker images in S3 (s3://$S3_BUCKET/$S3_PREFIX/docker-images/):"
    $AWS_CLI s3 ls "s3://$S3_BUCKET/$S3_PREFIX/docker-images/" \
        | awk '{printf "  %s %s  %s  %s\n", $1, $2, $3, $4}'
}

case "$ACTION" in
    export)
        export_to_s3
        ;;
    load)
        load_from_s3
        ;;
    list)
        list_s3_images
        ;;
    help|*)
        cat <<EOF
Usage: $0 <action>

Actions:
  export    Export Docker image to tar.gz and upload to S3
  load      Download from S3 and load into Docker
  list      List Docker images in S3

Environment variables (from .env or set manually):
  DOCKER_IMAGE (default: ghcr.io/zudva/piper-train:latest)
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL
  S3_BUCKET, S3_PREFIX

Example workflow:
  # On build machine:
  ./script/docker_to_s3.sh export

  # On RunPod (or any remote machine):
  ./script/docker_to_s3.sh load
EOF
        ;;
esac
