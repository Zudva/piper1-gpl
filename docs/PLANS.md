# Plans

Plans for future Piper development.

## Alignments

Experimental [alignment support][ALIGNMENTS.md] has been added. If useful, all of the existing voices need to be patched.

## Felix Mirage v2

Tracking doc:

- `docs/FELIX_MIRAGE_V2_PIPELINE.md` (canonical pipeline)
- `docs/TODO_FELIX_MIRAGE_V2.md` (checklist + acceptance criteria)

Near-term plan:

- Stage A: keep using `script/text_splitter/01_text_splitter.py` (text-only) and review chunk distribution.
- Stage B: implement a WhisperX-based alignment + matching tool that consumes `to_align.json` and emits a stable `cutlist.jsonl`.
- Stage C: cut/resample to 22050 Hz mono PCM WAV and write `metadata_2col.csv` + `config.json` into `piper-training/datasets/felix_mirage_v2_sr22050/`.

## piper.exe

Compiled executables for Piper need to be ported from the old repo. These should use `libpiper` now.
