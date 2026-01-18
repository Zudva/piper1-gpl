# Dataset Layout (Training)

This document defines the **canonical dataset structure** used by training scripts and validators.

## Recommended dataset root
Use one of these mount locations:
- `/workspace/datasets/<voice_name>` (RunPod)
- `/data/<voice_name>`

Avoid machine-specific paths (e.g. `/media/...`).

You can override dataset location with env vars:
- `DATA_DIR=/path/to/dataset`
- `PIPER_DATASET_DIR=/path/to/dataset`

## Required files
Dataset directory must contain:
- `config.json`
- `metadata_2col.csv` (pipe-delimited: `wav|text`)
- `wavs/` (audio files)

Example:
```
felix_mirage/
  config.json
  metadata_2col.csv
  wavs/
    felix_000001.wav
    ...
```

## Validation (100%)
Before training, dataset must pass validation:
- Policy: `reports/DATA_VALIDATION_POLICY.md`
- Template: `reports/VALIDATION_TEMPLATE.md`
- Script: `script/validate_dataset_full.py`

## Notes
- Keep dataset out of the git repo (ignored by `.gitignore`).
- Cache directory is stored inside dataset root as `.cache/`.
