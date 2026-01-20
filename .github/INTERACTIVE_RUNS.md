# Interactive Runs (Required)

All long-running tasks **must be started via interactive scripts** so that progress is visible in real time.
Do **not** run raw `python ...` commands directly for long Whisper validations.

## Required Scripts

- Rich UI runner for Whisper validation:
  - `script/run_whisper_validate_rich.py`

This script shows live progress, GPU usage, and shard log tails.

## Example (recommended)

```bash
# Activate venv
source .venv/bin/activate

# Run with Rich UI (2x3090)
python script/run_whisper_validate_rich.py \
  --dataset ../piper-training/datasets/felix_mirage_prepared_sr22050_seg15 \
  --gpus 0,1 \
  --workers-per-gpu 2 \
  --whisper-model large-v3 \
  --whisper-backend faster-whisper \
  --whisper-batch-size 16 \
  --whisper-num-workers 4 \
  --whisper-compute-type float16 \
  --whisper-beam-size 1 \
  --whisper-vad-filter \
  --progress-mode whisper \
  --progress-every 200
```

## Notes

- The Rich UI is required so the user can see progress. This is mandatory for Whisper validations.
- Reports are written under: `<dataset>/reports/validation_whisper_sharded_rich_YYYYMMDD_HHMMSS/`.
- You can adjust `--workers-per-gpu`, `--whisper-batch-size`, and `--whisper-num-workers` to trade speed vs VRAM.

## Запуск команд
- Никакие команды не запускаются скрыто. Перед запуском требуется явное указание запуска и краткое описание.
