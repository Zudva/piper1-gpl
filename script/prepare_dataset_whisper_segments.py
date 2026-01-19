#!/usr/bin/env python3
"""Prepare a Piper-style dataset by segmenting long recordings using Whisper timestamps.

This script is designed for long-form recordings (e.g., monologues) where each WAV may be
60+ seconds. It runs Whisper ASR, takes its segment timecodes, and then uses ffmpeg to
cut/resample each segment into short clips (default: <= 15s) with matching text.

Output dataset layout:
  <out_dataset>/
    config.json
    metadata_2col.csv   # wav|text
    wavs/
      <basename>_<seg>.wav

Notes:
- Uses local Whisper (no paid APIs).
- Keeps input dataset untouched.
- Requires `ffmpeg` in PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("|", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _ffmpeg_cut_resample(
    *,
    src_wav: Path,
    out_wav: Path,
    start: float,
    end: float,
    sample_rate: int,
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # -ss/-to before -i is faster for many cuts, but less accurate for some formats.
    # For WAV, accuracy is fine.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(src_wav),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        "-y",
        str(out_wav),
    ]
    _run(cmd)


def _iter_wavs(dataset_dir: Path) -> list[Path]:
    wavs_dir = dataset_dir / "wavs"
    if not wavs_dir.is_dir():
        raise FileNotFoundError(f"Missing wavs/ directory: {wavs_dir}")
    return sorted([p for p in wavs_dir.iterdir() if p.suffix.lower() == ".wav"])


def _read_wavs_file(path: Path) -> list[Path]:
    """Read a list of wav paths or filenames (one per line)."""
    wavs: list[Path] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            wavs.append(Path(line))
    return wavs


def _load_config(dataset_dir: Path) -> dict:
    config_path = dataset_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_config(out_dir: Path, base_config: dict, sample_rate: int) -> None:
    cfg = json.loads(json.dumps(base_config))
    cfg.setdefault("audio", {})
    cfg["audio"]["sample_rate"] = int(sample_rate)
    out_path = out_dir / "config.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_metadata(out_dir: Path, rows: Iterable[tuple[str, str]]) -> Path:
    out_path = out_dir / "metadata_2col.csv"
    with out_path.open("w", encoding="utf-8") as f:
        for wav_name, text in rows:
            f.write(f"{wav_name}|{text}\n")
    return out_path


def _read_metadata(dataset_dir: Path) -> dict[str, str]:
    meta_path = dataset_dir / "metadata_2col.csv"
    if not meta_path.is_file():
        return {}

    mapping: dict[str, str] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "|" not in line:
                continue
            wav, text = line.split("|", 1)
            wav = wav.strip()
            if not wav:
                continue
            mapping[wav] = text.strip()
    return mapping


def _whisper_segments(
    *,
    model,
    wav_path: Path,
    language: str,
) -> list[Segment]:
    result = model.transcribe(
        str(wav_path),
        language=language,
        task="transcribe",
        temperature=0.0,
        verbose=False,
    )

    segments: list[Segment] = []
    for seg in result.get("segments", []):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = _normalize_text(str(seg.get("text", "")))
        if not text:
            continue
        if end <= start:
            continue
        segments.append(Segment(start=start, end=end, text=text))

    return segments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Input dataset dir (has config.json, wavs/, metadata_2col.csv)")
    parser.add_argument(
        "--out-dataset",
        default=None,
        help="Output dataset dir (default: <dataset>_prepared)",
    )
    parser.add_argument(
        "--wavs-file",
        default=None,
        help="Optional file with WAV paths (absolute or relative) or WAV filenames, one per line.",
    )
    parser.add_argument(
        "--metadata-out",
        default=None,
        help="Optional metadata output path (default: <out_dataset>/metadata_2col.csv)",
    )
    parser.add_argument("--whisper-model", default="medium", help="Whisper model name (tiny/base/small/medium/large)")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Whisper device (auto selects cuda if available)",
    )
    parser.add_argument("--language", default="ru", help="Whisper language code")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate for output audio")

    parser.add_argument("--min-seg-seconds", type=float, default=1.0)
    parser.add_argument("--max-seg-seconds", type=float, default=15.0)
    parser.add_argument("--min-text-chars", type=int, default=2)
    parser.add_argument("--max-text-chars", type=int, default=300)

    parser.add_argument("--limit-files", type=int, default=0, help="If >0, only process first N WAVs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output dir if it exists")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset).expanduser().resolve()
    out_dir = Path(args.out_dataset).expanduser().resolve() if args.out_dataset else Path(f"{dataset_dir}_prepared")
    metadata_out = Path(args.metadata_out).expanduser().resolve() if args.metadata_out else out_dir / "metadata_2col.csv"

    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise SystemExit(f"Output dir exists (use --overwrite): {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)

    base_config = _load_config(dataset_dir)
    _write_config(out_dir, base_config, sample_rate=args.sample_rate)

    # Optional: keep original metadata for traceability (not used for segment text)
    src_meta = dataset_dir / "metadata_2col.csv"
    if src_meta.is_file():
        shutil.copy2(src_meta, out_dir / "metadata_source_2col.csv")

    # For progress output
    if args.wavs_file:
        wavs_file = Path(args.wavs_file).expanduser().resolve()
        raw = _read_wavs_file(wavs_file)

        wavs: list[Path] = []
        for p in raw:
            if p.is_absolute():
                wavs.append(p)
            else:
                # Treat as relative to dataset wavs/ first, then relative to wavs_file.
                candidate = (dataset_dir / "wavs" / p)
                if candidate.exists():
                    wavs.append(candidate)
                else:
                    wavs.append((wavs_file.parent / p).resolve())
    else:
        wavs = _iter_wavs(dataset_dir)
    if args.limit_files and args.limit_files > 0:
        wavs = wavs[: args.limit_files]

    # Import whisper only when needed (keeps help fast)
    import torch
    import whisper

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model(args.whisper_model, device=device)

    meta_rows: list[tuple[str, str]] = []

    for wav_index, wav_path in enumerate(wavs, start=1):
        segments = _whisper_segments(model=model, wav_path=wav_path, language=args.language)

        seg_counter = 0
        for seg in segments:
            duration = seg.end - seg.start
            if duration < args.min_seg_seconds or duration > args.max_seg_seconds:
                continue

            text = _normalize_text(seg.text)
            if len(text) < args.min_text_chars or len(text) > args.max_text_chars:
                continue

            out_name = f"{wav_path.stem}_{seg_counter:04d}.wav"
            out_wav = out_dir / "wavs" / out_name

            _ffmpeg_cut_resample(
                src_wav=wav_path,
                out_wav=out_wav,
                start=seg.start,
                end=seg.end,
                sample_rate=args.sample_rate,
            )

            meta_rows.append((out_name, text))
            seg_counter += 1

        print(f"[{wav_index}/{len(wavs)}] {wav_path.name}: {seg_counter} segments kept", flush=True)

    # Write metadata (can be redirected for sharded/multi-process runs)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    with metadata_out.open("w", encoding="utf-8") as f:
        for wav_name, text in meta_rows:
            f.write(f"{wav_name}|{text}\n")

    info_path = out_dir / "PREPARE_INFO.txt"
    with info_path.open("w", encoding="utf-8") as f:
        f.write(f"source_dataset={dataset_dir}\n")
        f.write(f"whisper_model={args.whisper_model}\n")
        f.write(f"device={device}\n")
        f.write(f"language={args.language}\n")
        f.write(f"sample_rate={args.sample_rate}\n")
        f.write(f"min_seg_seconds={args.min_seg_seconds}\n")
        f.write(f"max_seg_seconds={args.max_seg_seconds}\n")
        f.write(f"min_text_chars={args.min_text_chars}\n")
        f.write(f"max_text_chars={args.max_text_chars}\n")

    print(f"\nWrote dataset: {out_dir}", flush=True)
    print(f"Clips: {len(meta_rows)}", flush=True)
    print(f"Metadata: {metadata_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
