# Piper TTS Training - Quick Start Script
# Запуск обучения голосовой модели felix_mirage

Write-Host "=== Piper TTS Training ===" -ForegroundColor Cyan
Write-Host "Voice: felix_mirage (Russian)" -ForegroundColor Yellow
Write-Host "GPU: NVIDIA RTX 4090" -ForegroundColor Yellow
Write-Host "Dataset: 694 samples @ 44.1kHz" -ForegroundColor Yellow
Write-Host ""

# Проверка директорий
$dataPath = "D:\git\nik-v-local-talking-llm\actors\felix_mirage"
if (-not (Test-Path $dataPath)) {
    Write-Host "ERROR: Dataset not found at $dataPath" -ForegroundColor Red
    exit 1
}

$csvPath = "$dataPath\datasets\elevenlabs\zWSsRd3J6WyZFl12aGMB\metadata.csv"
if (-not (Test-Path $csvPath)) {
    Write-Host "ERROR: metadata.csv not found at $csvPath" -ForegroundColor Red
    exit 1
}

# Подсчет количества строк в датасете
$lineCount = (Get-Content $csvPath | Measure-Object -Line).Lines
Write-Host "Found $lineCount training samples" -ForegroundColor Green
Write-Host ""

# Запуск обучения
Write-Host "Starting training..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host ""

Set-Location E:\git\piper1-gpl
docker compose -f docker-compose.train.yml run --rm piper-train
