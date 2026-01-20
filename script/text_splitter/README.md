# Text Splitter (Stage A)

Goal: turn long monologues from `manifest.jsonl` into short, natural phrases suitable for later **forced alignment** (WhisperX/MFA).

This stage is **text-only**: it does not cut audio.

For the end-to-end Felix Mirage v2 pipeline (including where to keep build artifacts), see: `piper1-gpl/docs/FELIX_MIRAGE_V2_PIPELINE.md`.

## Input

An ElevenLabs-style JSONL where each line contains at least:

```json
{"audio_path":"wavs/file.wav","text":"..."}
```

## Output

A single JSON file (not JSONL):

```json
[
  {
    "audio_path": "wavs/file.wav",
    "sentences": ["...", "..."],
    "original_full_text": "...",
    "manifest_meta": {"voice_id": "...", "lang": "ru", "line": 123}
  }
]
```

## Install

Python 3.10+ recommended.

```bash
python -m pip install razdel
```

## Run

### Recommended paths (to avoid version confusion)

- **Source of truth** stays in `nik-v-local-talking-llm/.../manifest.jsonl`.
- **Build artifacts** (like `to_align.json`) go into `piper-training/datasets/felix_mirage_v2_work/`.

```bash
python script/text_splitter/01_text_splitter.py \
  --input nik-v-local-talking-llm/actors/felix_mirage/datasets/elevenlabs/zWSsRd3J6WyZFl12aGMB/manifest.jsonl \
  --output piper-training/datasets/felix_mirage_v2_work/to_align.json
```

Dry-run (strongly recommended first):

```bash
python script/text_splitter/01_text_splitter.py \
  --input nik-v-local-talking-llm/actors/felix_mirage/datasets/elevenlabs/zWSsRd3J6WyZFl12aGMB/manifest.jsonl \
  --output piper-training/datasets/felix_mirage_v2_work/to_align.sample20.json \
  --limit 20 --overwrite --print-samples 2
```

Optional tuning:

- `--target-length 160` (default)
- `--min-length 35` (default)

Policy notes (current Felix Mirage v2 decisions):

- Target length ~160 chars
- Min length 35 chars (avoid tiny/noisy clips)
- If you later add a hard limit (e.g. 230 chars), ensure the script can split a single overlong sentence further; otherwise “too long” chunks are unavoidable.

## Algorithm summary

- Normalization: collapse whitespace/newlines, remove spaces before punctuation, normalize dash, strip `|`.
- Sentence segmentation: `razdel.sentenize`.
- Long sentence fallback: if a single sentence exceeds `target-length`, it is split further on `; : , —` (or whitespace as fallback).
- Grouping: pack sentences into chunks up to `target-length`.
- Anti-orphan tail rule: if the last chunk is shorter than `min-length`, try to attach it to previous chunk (allowing a small overflow).

## Quick QC (chunk length stats)

This prints basic stats for a produced `to_align*.json` file:

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
