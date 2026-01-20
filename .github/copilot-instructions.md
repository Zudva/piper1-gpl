# Copilot Instructions (piper1-gpl)

## Project Overview
This repository contains tools and scripts for training and fine-tuning Piper TTS models, specifically focused on the `felix_mirage` voice. It includes utilities for dataset preparation, alignment review, and training orchestration.

## Key Components

### Data Review UI (`script/cutlist_review_ui.py`)
A Gradio-based tool for manually reviewing and correcting dataset alignment (`cutlist.jsonl`).
- **Input**: `cutlist.jsonl` (JSONL file with text, audio path, start/end times).
- **Features**:
    - **Smart Gap Detection**: For `unmatched` segments (where start/end are null), the tool calculates the gap between the previous and next segment to isolate the missing audio.
    - **FFmpeg Integration**: Uses `ffmpeg` to physically cut audio segments for playback and transcription.
    - **Whisper Transcription**: Integrates `faster-whisper` to auto-recognize speech in the selected segment.
- **Dependencies**: `gradio`, `soundfile`, `faster-whisper`, `ffmpeg` (system binary).

### Environment
- **Python**: 3.10+
- **Virtual Environment**: `.venv`
- **Dependencies**: Listed in `requirements.txt` (or installed ad-hoc: `gradio`, `soundfile`, `faster-whisper`).

## Coding Rules & Patterns

1.  **Path Handling**:
    - Use `pathlib.Path` for all file operations.
    - Be aware of cross-repository paths (audio files may reside in `nik-v-local-talking-llm`).
    - The review UI includes logic (`_resolve_audio_path`) to hunt for audio files if relative paths don't match.

2.  **Audio Processing**:
    - Always use `soundfile` for reading audio metadata and data.
    - When sending audio to Gradio, cast to `float32` to avoid serialization issues (`data.astype("float32")`).
    - Use `ffmpeg` for precise cutting, especially when `start`/`end` timestamps are involved.

3.  **Logging**:
    - Use the custom `_log` function in scripts to ensure dual output to console and log files.

4.  **Hardware**:
    - Scripts are designed to run on a machine with NVIDIA RTX 3090.
    - Whisper models default to `device="cuda", compute_type="float16"`.

## Common Tasks

- **Fixing Alignment**: Use the UI to fill in missing text or correct timestamps for `unmatched` entries.
- **Training**: Refer to `TRAINING_RU.md` or `docker-compose.train.yml` for training commands.

## Запуск команд
- Никакие команды не запускаются скрыто. Перед запуском требуется явное указание запуска и краткое описание.
