# TODO — Felix Mirage v2 dataset pipeline

This is the operational checklist for building **felix_mirage_v2_sr22050**.

Constraints:

- Do not store generated artifacts under `nik-v-local-talking-llm/actors/...`.
- Do not run heavy processing unless explicitly requested.
- Keep work artifacts under `piper-training/datasets/felix_mirage_v2_work/`.

## Phase 0 — Setup

- [ ] Create directories:
  - [ ] `piper-training/datasets/felix_mirage_v2_work/`
  - [ ] `piper-training/datasets/felix_mirage_v2_sr22050/` (empty until Phase 3)

Acceptance:
- Work dir exists and is safe to delete/regenerate.
- Final dir exists but stays empty until build.

## Phase 1 — Stage A (text-only split)

Input:
- `nik-v-local-talking-llm/actors/felix_mirage/datasets/elevenlabs/<voice_id>/manifest.jsonl`

Output:
- `piper-training/datasets/felix_mirage_v2_work/to_align.json`

Steps:

- [ ] Dry-run sample (e.g. `--limit 20 --print-samples 2`) and manually review a couple of outputs.
- [ ] Quick QC stats on the sample JSON.
- [ ] Full run to produce `to_align.json`.

Acceptance:
- Chunk lengths are mostly within target (e.g. mean/median near 160).
- Very short chunks (<35 chars) are rare (tail-merge is working).
- Very long chunks (>230 chars) are either rare or explained (true long sentences).
- A few random samples sound plausible as standalone phrases (text-only judgement).

## Phase 2 — Stage B (WhisperX alignment + cutlist)

Goal:
- Align Stage A chunks to audio and produce a stable cutlist.

Output (suggested):
- `piper-training/datasets/felix_mirage_v2_work/alignment/cutlist.jsonl`

Steps:

- [ ] Decide alignment/matching strategy when manifest text != spoken transcript.
- [ ] Use `script/felix_mirage_v2_align_whisperx.py` to generate a first cutlist on a small sample.
- [ ] Generate a cutlist with `{src_audio, start, end, text}`.
- [ ] Spot-check alignment on at least 10 random entries.

Acceptance:
- Cut boundaries are not obviously clipping phonemes.
- Minimum segment duration is sane (e.g. >2s unless intentionally shorter).
- Text in cutlist matches the intended chunk text.

## Phase 3 — Build final Piper dataset (cut + resample)

Output:
- `piper-training/datasets/felix_mirage_v2_sr22050/`
  - `wavs/*.wav` (mono, 22050 Hz, PCM)
  - `metadata_2col.csv` (`wav|text`)
  - `config.json`

Steps:

- [ ] Cut audio per cutlist and write WAVs.
- [ ] Write `metadata_2col.csv`.
- [ ] Write `config.json` (audio.sample_rate=22050).

Acceptance:
- No clips are ~1.0s unless explicitly intended.
- WAVs decode cleanly (no ffmpeg decode errors) and have consistent format.

## Phase 4 — Validate

- [ ] Run basic structural validation (files exist, metadata matches wavs).
- [ ] Optional: run Whisper validation on a sample if needed.

Acceptance:
- Validation reports no systemic issues (missing wavs, massive duplicates, wrong SR, etc.).
- Any remaining low-similarity items are explainable (e.g., paraphrase) or flagged for review.
