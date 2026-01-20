# Felix Mirage v2 (Piper) — dataset pipeline

This document describes the **Felix Mirage v2** dataset preparation workflow for Piper, with a strong separation between:

- **Source of truth** (original recordings + original text)
- **Build artifacts** (generated intermediate files and final training dataset)

The goal is to avoid version confusion and prevent repeating the earlier "1-second fragments" problem caused by bad segmentation.

## Status / Checklist

For a step-by-step checklist with acceptance criteria, see: `docs/TODO_FELIX_MIRAGE_V2.md`.

Run policy reminder:

- Avoid heavy processing unless explicitly requested.
- Keep artifacts under `piper-training/datasets/felix_mirage_v2_work/`.
- Keep `nik-v-local-talking-llm/actors/...` as **source of truth** only.

## Repos and responsibilities

- `nik-v-local-talking-llm/`
  - Holds the original ElevenLabs-exported dataset under `actors/felix_mirage/...`.
  - **Do not** store generated artifacts here (except the original manifest).

- `piper1-gpl/`
  - Holds the scripts used to prepare and validate datasets.
  - This documentation lives here.

- `piper-training/`
  - Holds the **working directory** and the **final Piper dataset** for training.

## Canonical inputs (source of truth)

- Manifest JSONL (canonical text + audio references):
  - `nik-v-local-talking-llm/actors/felix_mirage/datasets/elevenlabs/<voice_id>/manifest.jsonl`
- Audio files referenced by the manifest:
  - `.../wavs/*.wav`

The manifest is the only file we trust for pairing `audio_path` ↔ `text`.

## Output locations (build artifacts)

Create two directories in `piper-training/datasets/`:

- Work directory (intermediate artifacts):
  - `piper-training/datasets/felix_mirage_v2_work/`
- Final Piper dataset directory (training-ready):
  - `piper-training/datasets/felix_mirage_v2_sr22050/`

Rationale:
- Work dir changes often and can be regenerated.
- Final dataset is the stable input to training.

## Phase 1 — Stage A (smart text split, text-only)

### Purpose

Split long monologues into shorter, TTS-friendly chunks **without touching audio**. This produces a `to_align` file used by Phase 2.

### Policy (approved)

- Target chunk length: ~160 characters
- Min chunk length: 35 characters
- Tail chunks are merged into the previous chunk when possible
- Do not attempt to "fix" quotes/abbreviations via splitting heuristics; prefer stable text

Note: a hard limit (e.g. 230 chars) is desirable, but must be backed by logic that can split a single overlong sentence.

### Output contract

Stage A produces a JSON file (array) where each item ties one original audio file to multiple target chunks:

- `audio_path`: path from manifest (e.g. `wavs/xxx.wav`)
- `sentences`: list of chunk strings (order matters)
- `original_full_text`: cleaned/normalized full text

This file is the **input** to Stage B alignment.

### Run (dry-run first)

From the `piper1-gpl/` repo:

```bash
python script/text_splitter/01_text_splitter.py \
  --input nik-v-local-talking-llm/actors/felix_mirage/datasets/elevenlabs/zWSsRd3J6WyZFl12aGMB/manifest.jsonl \
  --output piper-training/datasets/felix_mirage_v2_work/to_align.sample20.json \
  --limit 20 --overwrite --print-samples 2
```

If output looks good, run full:

```bash
python script/text_splitter/01_text_splitter.py \
  --input nik-v-local-talking-llm/actors/felix_mirage/datasets/elevenlabs/zWSsRd3J6WyZFl12aGMB/manifest.jsonl \
  --output piper-training/datasets/felix_mirage_v2_work/to_align.json \
  --overwrite
```

### Quality control (fast)

Compute chunk length stats:

```bash
python - <<'PY'
import json, statistics

path = 'piper-training/datasets/felix_mirage_v2_work/to_align.sample20.json'
items = json.load(open(path, 'r', encoding='utf-8'))
lengths = [len(s) for it in items for s in it.get('sentences', []) if s]
print('chunks:', len(lengths))
print('min/max:', min(lengths), max(lengths))
print('mean/median:', round(statistics.mean(lengths), 1), statistics.median(lengths))
print('too_short(<35):', sum(l < 35 for l in lengths))
print('too_long(>230):', sum(l > 230 for l in lengths))
PY
```

## Phase 2 — Stage B (alignment + cutlist, WhisperX)

### Purpose

Convert `to_align.json` into precise time ranges per chunk and build a **cutlist**:

- Input: long wav + list of target chunks
- Output: `{src_audio, start, end, text}` records

Why WhisperX:
- Word-level boundaries reduce cut artifacts ("chomp" at start/end).

Important note:
- WhisperX aligns words for its recognized transcript. If the manifest text significantly differs from spoken audio, include a matching step (text normalization + fuzzy matching) between "desired chunks" and ASR transcript.

### Implementation (script)

Stage B script (this repo):

- `script/felix_mirage_v2_align_whisperx.py`

Suggested dry-run (limits work):

```bash
python script/felix_mirage_v2_align_whisperx.py \
  --to-align piper-training/datasets/felix_mirage_v2_work/to_align.sample20.json \
  --audio-root nik-v-local-talking-llm/actors/felix_mirage/datasets/elevenlabs/zWSsRd3J6WyZFl12aGMB/ \
  --out piper-training/datasets/felix_mirage_v2_work/alignment/cutlist.sample20.jsonl \
  --limit 2 --dry-run
```

Expected artifacts (suggested):

- `piper-training/datasets/felix_mirage_v2_work/alignment/` (work dir)
  - `cutlist.jsonl` (or `.json`) with `{src_audio, start, end, text}`
  - logs and alignment debug outputs

## Phase 3 — Build final Piper dataset (sr22050)

### Purpose

Produce the training-ready dataset in:

- `piper-training/datasets/felix_mirage_v2_sr22050/`

Expected output layout:

- `wavs/*.wav` (mono, 22050 Hz, PCM)
- `metadata_2col.csv` (`wav|text`)
- `config.json`

## Run policy (important)

- Avoid heavy processing unless explicitly requested.
- Keep artifacts under `piper-training/datasets/felix_mirage_v2_work/`.
- Avoid rewriting text aggressively before alignment; preserve stable text (or store both raw and normalized forms).
