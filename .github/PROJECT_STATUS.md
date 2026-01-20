# Project Status Report (Jan 19, 2026)

This is a team handoff note for the “Felix Mirage” Piper training dataset readiness pipeline.

## What’s Ready Now
- Canonical prepared dataset exists and passes strict non‑Whisper validation.
- Whisper validation is operationalized for **live progress** (Rich UI) and **multi‑GPU sharding**.
- Repo policy docs exist to avoid “silent long runs”.

## Canonical Dataset (single source of truth)
- Dataset root (workspace‑relative): `../piper-training/datasets/felix_mirage_prepared_sr22050_seg15`
- Expected contents: `config.json`, `metadata_2col.csv`, `wavs/`, `PREPARE_INFO.txt`, `reports/`
- Scale: 18,377 clips (metadata lines == wav count at time of PASS)

## Latest Validation Results
- Strict full validation (no Whisper): **PASS**
  - Report directory: `../piper-training/datasets/felix_mirage_prepared_sr22050_seg15/reports/validation_clean_20260119_192031`
- Full Whisper validation (large‑v3): **not yet recorded here as completed**
  - The pipeline is ready to run; once finished, paste the report directory path here.

## Tools Implemented (repo: piper1-gpl)
- Full validator (structure/text/audio + optional Whisper): `script/validate_dataset_full.py`
  - Produces: `reports/validation_*.md` + `reports/validation_*.tsv`
  - Supports real‑time progress (`--progress-every`, `--progress-mode`) and faster‑whisper tuning.
- Whisper sharded runner: `script/validate_dataset_whisper_sharded.py`
  - Shards metadata and runs parallel validator processes across multiple GPUs.
  - Writes `SUMMARY.txt` and per‑shard logs under `<report>/shard_reports/`.
- Rich interactive runner (recommended entrypoint): `script/run_whisper_validate_rich.py`
  - Live dashboard: shard log tail + nvidia-smi snapshot.

## Dataset Readiness Pipeline (team standard)
Canonical description lives in: `.github/DATASET_READINESS_PIPELINE.md`.

### Stage 1 — Base validation (required, fast)
**Goal:** verify structure, text constraints, audio format, file consistency.

Command (from piper1-gpl):
```bash
python script/validate_dataset_full.py \
  --dataset ../piper-training/datasets/felix_mirage_prepared_sr22050_seg15 \
  --report-dir ../piper-training/datasets/felix_mirage_prepared_sr22050_seg15/reports
```

Artifacts:
- `validation_*.md` (PASS/FAIL summary)
- `validation_*.tsv` (problem lines)

### Stage 2 — Quality report (fast, before Whisper)
**Goal:** human + stats signal before expensive full ASR.
Artifacts: quality markdown + suspects list + (optional) quick ASR sample report.

### Stage 3 — Full Whisper validation (only if needed)
**Goal:** full ASR-alignment with large‑v3, sharded across GPUs, live progress.

Command (recommended):
```bash
python script/run_whisper_validate_rich.py \
  --dataset ../piper-training/datasets/felix_mirage_prepared_sr22050_seg15 \
  --gpus 0,1 \
  --workers-per-gpu 1 \
  --whisper-model large-v3 \
  --whisper-backend auto \
  --progress-mode whisper \
  --progress-every 200
```

### Stage 4 — Mini-train sanity check (optional)
**Goal:** 1–2 epochs + listen to generated samples to catch gross issues early.

### Stage 5 — Full training
**Goal:** full run; archive logs/checkpoints and link back to dataset report.

## Ongoing Support / “Data readiness” policy
- All long validations must be started via interactive scripts (Rich UI) per `.github/INTERACTIVE_RUNS.md`.
- Reports must be stored under `<dataset>/reports/` and never overwrite previous runs.
- Any dataset edits must be followed by Stage 1 PASS; Stage 2 is recommended after significant edits.
- For team sync: update this file with the latest Stage results + report dir paths.

## Cleanup Already Done
- Removed obsolete dataset copies (kept only the canonical dataset under `../piper-training/datasets/`).
- Removed redundant historical reports to reduce confusion.

## Handoff Checklist
- [x] One canonical dataset only
- [x] Stage 1 PASS report path recorded
- [x] Interactive sharded Whisper validation runner available
- [ ] Stage 2 quality report created and reviewed
- [ ] Stage 3 Whisper validation completed and report path recorded
- [ ] Stage 4 mini-train sanity check (optional)

## Запуск команд
- Никакие команды не запускаются скрыто. Перед запуском требуется явное указание запуска и краткое описание.
