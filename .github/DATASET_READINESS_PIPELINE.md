# Dataset Readiness Pipeline (Piper)

This document defines the standard, reproducible pipeline to decide whether a prepared dataset is ready for training.

Principles:
- Prefer **fast, deterministic checks** first.
- Use **interactive progress** for long-running stages.
- All artifacts go under `<dataset>/reports/` and are append-only.
- No paid APIs.

## Stages (1–5)

### 1) Base validation of structure and format (required, fast)
**When:** immediately after dataset preparation / any edits.

**Checks:**
- Presence of `config.json`, `metadata_2col.csv`, `wavs/`.
- Metadata integrity (2 columns, non-empty).
- Referenced wav files exist; no extra wavs.
- Audio format and constraints (sample rate, mono, PCM16, duration).
- Text constraints (min/max length, character class, Cyrillic ratio, non-printable).

**Artifacts:**
- `validation_*.md` (PASS/FAIL)
- `validation_*.tsv` (problem rows)

**Command (from piper1-gpl):**
```bash
python script/validate_dataset_full.py \
  --dataset <DATASET_DIR> \
  --report-dir <DATASET_DIR>/reports
```

---

### 2) Quality report (fast, before Whisper)
**When:** after Stage 1 PASS.

**Goal:** catch obvious quality problems early, before expensive full ASR alignment.

**Checks / outputs:**
- Random sample list (10–50 clips) for listening.
- Summary statistics (durations, text lengths).
- Heuristics:
  - duplicate normalized texts
  - code-switch / unexpected Latin presence
  - repeated tokens / abnormal repetition
- Optional quick ASR sample (N=200–500) to estimate rough alignment quality.

**Artifacts:**
- `quality_report.md` (summary)
- `samples.tsv` (list to listen)
- `suspects.tsv` (rows to investigate)
- `stats.json` (machine-readable summary)

**Command (from piper1-gpl):**
```bash
python script/dataset_quality_report.py \
  --dataset <DATASET_DIR> \
  --report-dir <DATASET_DIR>/reports/quality_<TIMESTAMP> \
  --sample-count 30 \
  --suspect-top 200
```

Optional quick ASR sample:
```bash
python script/dataset_quality_report.py \
  --dataset <DATASET_DIR> \
  --report-dir <DATASET_DIR>/reports/quality_<TIMESTAMP> \
  --asr-sample 300 \
  --asr-model medium \
  --asr-device cuda
```

---

### 3) Full Whisper validation run (only if needed)
**When:** after Stage 2 review.

**Goal:** full ASR alignment using Whisper `large-v3` (or team-approved model) to detect mislabels, wrong text, bad audio.

**Requirements:**
- Use sharding across available GPUs.
- Run in interactive mode (Rich UI) to see progress.

**Artifacts:**
- Sharded report directory with:
  - `SUMMARY.txt`
  - per-shard `validator_whisper.log`
  - per-shard `validation_*.md` and `validation_*.tsv`

**Command (recommended):**
```bash
python script/run_whisper_validate_rich.py \
  --dataset <DATASET_DIR> \
  --gpus 0,1 \
  --workers-per-gpu 1 \
  --whisper-model large-v3 \
  --whisper-backend auto \
  --progress-mode whisper \
  --progress-every 200
```

---

### 4) Mini-train (sanity check)
**When:** after Stage 3 PASS.

**Goal:** verify the training pipeline end-to-end and listen to generated samples after 1–2 epochs.

**Artifacts:**
- short training log
- checklist of listened samples + verdict

---

### 5) Full training
**When:** after all prior stages PASS.

**Artifacts:**
- full training logs
- checkpoints
- link back to dataset + report directories

## Where we record the latest state
- Project handoff / current status: `.github/PROJECT_STATUS.md`
- Interactive run policy: `.github/INTERACTIVE_RUNS.md`

## Запуск команд
- Никакие команды не запускаются скрыто. Перед запуском требуется явное указание запуска и краткое описание.
