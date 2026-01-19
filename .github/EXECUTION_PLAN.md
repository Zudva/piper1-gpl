# Execution Plan: Felix Mirage Dataset → Production Model

**Target Dataset:** `../piper-training/datasets/felix_mirage_prepared_sr22050_seg15`  
**Current Status:** Stage 1 PASS ✅  
**Goal:** Complete Stages 2–5 and produce production-ready checkpoint

---

## Phase 1: Quality Gate (Stage 2) — READY TO START

### Objective
Fast pre-Whisper quality check: statistics, heuristics, human review sample.

### Prerequisites
- [x] Stage 1 PASS (structure/format validation completed)
- [x] `script/dataset_quality_report.py` implemented and tested

### Command
```bash
cd ${workspaceFolder}/piper1-gpl

python script/dataset_quality_report.py \
  --dataset ../piper-training/datasets/felix_mirage_prepared_sr22050_seg15 \
  --sample-count 30 \
  --suspect-top 100
```

### Expected Artifacts
- `../piper-training/datasets/felix_mirage_prepared_sr22050_seg15/reports/quality_<TIMESTAMP>/`
  - `quality_report.md` (summary + histograms)
  - `samples.tsv` (30 clips for listening)
  - `suspects.tsv` (top 100 suspect rows)
  - `stats.json` (machine-readable statistics)

### Success Criteria (PASS)
- [ ] Script completes without errors
- [ ] Duration histogram shows reasonable distribution (no spikes at extremes)
- [ ] Text length p95 < 300 chars
- [ ] Duplicate normalized text count < 5%
- [ ] Human review of `samples.tsv`: no obvious mislabels or corrupted audio

### Decision Point
- **PASS** → Proceed to Phase 2 (optional quick ASR sample or directly to Stage 3)
- **FAIL** → Fix dataset issues, re-run Stage 1, repeat Phase 1

---

## Phase 2a (Optional): Quick ASR Sample — IF NEEDED

### Objective
Test ASR alignment quality on small subset (300 clips) before expensive full Whisper run.

### Command
```bash
cd ${workspaceFolder}/piper1-gpl

python script/dataset_quality_report.py \
  --dataset ../piper-training/datasets/felix_mirage_prepared_sr22050_seg15 \
  --sample-count 30 \
  --suspect-top 100 \
  --asr-sample 300 \
  --asr-model medium \
  --asr-device cuda \
  --asr-language ru
```

### Expected Additional Artifact
- `asr_sample.tsv` (300 rows with ref/hyp/similarity)

### Success Criteria (PASS)
- [ ] Mean similarity ≥ 0.85
- [ ] Low similarity count (sim < 0.8) < 10%
- [ ] No systematic errors (e.g., all numbers/names failing)

### Decision Point
- **PASS** → Proceed to Phase 2b (full Whisper)
- **FAIL** → Investigate low-similarity outliers, consider dataset cleanup

---

## Phase 2b: Full Whisper Validation (Stage 3) — INTERACTIVE REQUIRED

### Objective
Full ASR alignment check with `large-v3` on all 18,377 clips using 2×RTX3090.

### Prerequisites
- [x] Stage 2 quality report reviewed and PASS
- [x] `script/run_whisper_validate_rich.py` available
- [ ] No other heavy GPU tasks running (check `nvidia-smi`)

### Command (Interactive Rich UI)
```bash
cd ${workspaceFolder}/piper1-gpl

python script/run_whisper_validate_rich.py \
  --dataset ../piper-training/datasets/felix_mirage_prepared_sr22050_seg15 \
  --gpus 0,1 \
  --workers-per-gpu 1 \
  --whisper-model large-v3 \
  --whisper-backend auto \
  --whisper-language ru \
  --whisper-batch-size 8 \
  --whisper-num-workers 2 \
  --whisper-compute-type float16 \
  --whisper-beam-size 5 \
  --progress-mode whisper \
  --progress-every 200
```

### Expected Runtime
~6–12 hours (depends on GPU utilization and file durations)

### Expected Artifacts
- `../piper-training/datasets/felix_mirage_prepared_sr22050_seg15/reports/validation_whisper_sharded_rich_<TIMESTAMP>/`
  - `SUMMARY.txt`
  - `runner.log`
  - `shard_reports/shard_0/` and `shard_reports/shard_1/`
    - `validator_whisper.log` (per-row progress lines)
    - `validation_*.md` (PASS/FAIL per shard)
    - `validation_*.tsv` (problem rows)

### Success Criteria (PASS)
- [ ] Both shards exit code 0
- [ ] `SUMMARY.txt` shows no failed shards
- [ ] Low similarity issues < 2% of total clips
- [ ] Mean similarity ≥ 0.90

### Monitoring During Run
Watch Rich UI for:
- GPU utilization (should be >50% on both GPUs)
- Per-row similarity scores (scroll shard logs)
- Progress rate (~300-500 rows/hour expected)

### Decision Point
- **PASS** → Proceed to Phase 3 (mini-train)
- **FAIL** → Review low-similarity TSV, fix/remove bad clips, re-run Stage 1+3

---

## Phase 3: Mini-Train Sanity Check (Stage 4) — 1–2 Epochs

### Objective
Verify training pipeline and listen to generated samples before committing to full training.

### Prerequisites
- [x] Stage 3 Whisper validation PASS
- [ ] Python 3.10+ venv with `piper-tts[train]` installed
- [ ] C++ extensions built (`build_monotonic_align.sh`, `setup.py build_ext`)

### Command
```bash
cd ${workspaceFolder}/piper1-gpl
source .venv/bin/activate

# Mini-train (1-2 epochs)
./train_resume.sh "" 2
```

### Expected Runtime
~30–60 minutes per epoch (depends on dataset size and GPU)

### Expected Artifacts
- `lightning_logs/version_<N>/checkpoints/epoch=0-*.ckpt`
- `lightning_logs/version_<N>/checkpoints/epoch=1-*.ckpt`
- TensorBoard logs

### Success Criteria (PASS)
- [ ] Training starts without errors
- [ ] Loss decreases from epoch 0 to epoch 1
- [ ] Export to ONNX succeeds
- [ ] Generated samples (5-10 sentences) sound intelligible
- [ ] No crashes or OOM errors

### Generate Test Samples
```bash
# Export epoch 1 checkpoint to ONNX
EXPORT_ONLY=1 \
CHECKPOINT=lightning_logs/version_<N>/checkpoints/epoch=1-*.ckpt \
OUTPUT_FILE=felix_mirage_epoch1_test.onnx \
python -c "from piper_train.vits.export_onnx import main; main()"

# Generate samples
python test_onnx_generate.py \
  --model felix_mirage_epoch1_test.onnx \
  --text "Привет, это тестовый голос."
```

### Human Review Checklist
- [ ] Listen to 5–10 generated samples
- [ ] Voice is recognizable (not just noise/gibberish)
- [ ] Prosody is acceptable for epoch 1–2
- [ ] No obvious glitches/artifacts

### Decision Point
- **PASS** → Proceed to Phase 4 (full training)
- **FAIL** → Debug training config, check dataset preparation, review logs

---

## Phase 4: Full Training (Stage 5) — Production Run

### Objective
Train to convergence (~200-500 epochs) and produce production checkpoint.

### Prerequisites
- [x] Stage 4 mini-train PASS
- [ ] Clear schedule for GPU usage (multi-day run)

### Command
```bash
cd ${workspaceFolder}/piper1-gpl
source .venv/bin/activate

# Full training (resume from epoch 2 or start fresh)
./train_resume.sh "" 10000
```

### Monitoring
- TensorBoard: `tensorboard --logdir lightning_logs`
- Periodic checkpoint exports + listening tests every 50 epochs
- Loss convergence: watch for plateau

### Expected Runtime
2–7 days (depends on target epochs and early stopping)

### Success Criteria (Production Ready)
- [ ] Validation loss converged (plateau for 20+ epochs)
- [ ] Samples at final epoch sound natural and clear
- [ ] No overfitting (train/val loss gap reasonable)
- [ ] Final checkpoint exported to ONNX successfully

### Final Artifacts
- `lightning_logs/version_<N>/checkpoints/epoch=<FINAL>-*.ckpt`
- `felix_mirage_epoch<FINAL>.onnx`
- `felix_mirage_epoch<FINAL>.onnx.json`

---

## Phase 5: Production Deployment & Handoff

### Tasks
- [ ] Archive final checkpoint + config + training logs
- [ ] Update `PROJECT_STATUS.md` with final report paths
- [ ] Create deployment package (ONNX + config + samples + README)
- [ ] Tag release in Git: `git tag v1.0.0-felix-mirage`
- [ ] Push to GitHub: `git push origin v1.0.0-felix-mirage`

### Handoff Checklist
- [ ] All Stage 1–5 reports archived under `<dataset>/reports/`
- [ ] Final ONNX model tested on production inference script
- [ ] Team documentation updated with model location and usage examples

---

## Quick Reference Commands

| Phase | Stage | Command Entry Point |
|-------|-------|---------------------|
| 1 | 2 (Quality) | `python script/dataset_quality_report.py --dataset <DATASET>` |
| 2a | 2 (ASR sample) | `python script/dataset_quality_report.py --dataset <DATASET> --asr-sample 300` |
| 2b | 3 (Whisper) | `python script/run_whisper_validate_rich.py --dataset <DATASET> --gpus 0,1` |
| 3 | 4 (Mini-train) | `./train_resume.sh "" 2` |
| 4 | 5 (Full train) | `./train_resume.sh "" 10000` |

---

## Current Phase Status

- [x] Phase 1 prerequisites met (Stage 1 PASS)
- [ ] Phase 1 execution (Quality report)
- [ ] Phase 2a execution (optional ASR sample)
- [ ] Phase 2b execution (Full Whisper)
- [ ] Phase 3 execution (Mini-train)
- [ ] Phase 4 execution (Full training)
- [ ] Phase 5 (Deployment)

**Next Action:** Execute Phase 1 (Quality Report) command and review artifacts.
