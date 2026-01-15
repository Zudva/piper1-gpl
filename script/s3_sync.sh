#!/bin/bash
# S3 sync script for Piper training (Timeweb S3-compatible storage)
# Usage: 
#   ./s3_sync.sh upload-checkpoint <path>
#   ./s3_sync.sh download-checkpoint <name>
#   ./s3_sync.sh upload-logs
#   ./s3_sync.sh download-dataset
#   ./s3_sync.sh list-checkpoints

set -euo pipefail

# Load environment variables from .env if present
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Required env vars
: "${AWS_ACCESS_KEY_ID:?Missing AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Missing AWS_SECRET_ACCESS_KEY}"
: "${AWS_ENDPOINT_URL:?Missing AWS_ENDPOINT_URL}"
: "${S3_BUCKET:?Missing S3_BUCKET}"
: "${S3_PREFIX:?Missing S3_PREFIX}"

AWS_REGION="${AWS_DEFAULT_REGION:-ru-1-hot}"
AWS_CLI="aws --endpoint-url=$AWS_ENDPOINT_URL --region=$AWS_REGION"

ACTION="${1:-help}"

upload_checkpoint() {
    local ckpt_path="$1"
    if [ ! -f "$ckpt_path" ]; then
        echo "ERROR: Checkpoint not found: $ckpt_path"
        exit 1
    fi
    
    local ckpt_name=$(basename "$ckpt_path")
    local s3_key="$S3_PREFIX/checkpoints/$ckpt_name"
    
    echo "Uploading checkpoint: $ckpt_path -> s3://$S3_BUCKET/$s3_key"
    $AWS_CLI s3 cp "$ckpt_path" "s3://$S3_BUCKET/$s3_key"
    echo "✓ Upload complete"
}

download_checkpoint() {
    local ckpt_name="$1"
    local s3_key="$S3_PREFIX/checkpoints/$ckpt_name"
    local local_dir="lightning_logs"
    
    mkdir -p "$local_dir"
    echo "Downloading checkpoint: s3://$S3_BUCKET/$s3_key"
    $AWS_CLI s3 cp "s3://$S3_BUCKET/$s3_key" "$local_dir/$ckpt_name"
    echo "✓ Downloaded to: $local_dir/$ckpt_name"
}

download_latest_checkpoint() {
    local latest=$($AWS_CLI s3 ls "s3://$S3_BUCKET/$S3_PREFIX/checkpoints/" \
        | grep '\.ckpt$' \
        | sort -k1,2 \
        | tail -1 \
        | awk '{print $4}')
    
    if [ -z "$latest" ]; then
        echo "No checkpoints found in S3"
        return 1
    fi
    
    echo "Latest checkpoint: $latest"
    download_checkpoint "$latest"
}

upload_logs() {
    if [ ! -d "lightning_logs" ]; then
        echo "No lightning_logs directory found"
        exit 1
    fi
    
    echo "Uploading TensorBoard logs..."
    $AWS_CLI s3 sync lightning_logs/ "s3://$S3_BUCKET/$S3_PREFIX/lightning_logs/" \
        --exclude "*.ckpt" \
        --exclude "*.tmp"
    echo "✓ Logs uploaded"
}

download_dataset() {
    local data_dir="${DATA_DIR:-/data}"
    local s3_dataset_prefix="${S3_DATASET_PREFIX:-$S3_PREFIX/dataset}"
    
    if [ -f "$data_dir/.dataset_downloaded" ]; then
        echo "Dataset already downloaded (marker file exists)"
        return 0
    fi
    
    echo "Downloading dataset from s3://$S3_BUCKET/$s3_dataset_prefix/"
    mkdir -p "$data_dir"
    $AWS_CLI s3 sync "s3://$S3_BUCKET/$s3_dataset_prefix/" "$data_dir/" \
        --exclude ".cache/*"
    
    touch "$data_dir/.dataset_downloaded"
    echo "✓ Dataset downloaded to $data_dir"
}

list_checkpoints() {
    echo "Checkpoints in S3 (s3://$S3_BUCKET/$S3_PREFIX/checkpoints/):"
    $AWS_CLI s3 ls "s3://$S3_BUCKET/$S3_PREFIX/checkpoints/" \
        | grep '\.ckpt$' \
        | awk '{printf "  %s %s  %s\n", $1, $2, $4}'
}

case "$ACTION" in
    upload-checkpoint)
        upload_checkpoint "${2:?Missing checkpoint path}"
        ;;
    download-checkpoint)
        download_checkpoint "${2:?Missing checkpoint name}"
        ;;
    download-latest)
        download_latest_checkpoint
        ;;
    upload-logs)
        upload_logs
        ;;
    download-dataset)
        download_dataset
        ;;
    list-checkpoints|list)
        list_checkpoints
        ;;
    help|*)
        cat <<EOF
Usage: $0 <action> [args]

Actions:
  upload-checkpoint <path>      Upload checkpoint to S3
  download-checkpoint <name>    Download specific checkpoint from S3
  download-latest               Download latest checkpoint from S3
  upload-logs                   Sync TensorBoard logs to S3 (excludes .ckpt)
  download-dataset              Download training dataset from S3
  list-checkpoints              List all checkpoints in S3
  help                          Show this help

Environment variables (required):
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL
  S3_BUCKET, S3_PREFIX

Optional:
  AWS_DEFAULT_REGION (default: ru-1-hot)
  DATA_DIR (default: /data)
  S3_DATASET_PREFIX (default: \$S3_PREFIX/dataset)
EOF
        exit 0
        ;;
esac
