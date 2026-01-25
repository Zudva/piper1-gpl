#!/usr/bin/env python3
"""Prepare a Piper-style dataset from an ElevenLabs-style manifest.jsonl.

This converts a dataset like:
  nik-v-local-talking-llm/actors/felix_mirage/datasets/elevenlabs/<voice_id>/manifest.jsonl
into a Piper training/validation dataset layout:
  <out_dataset>/
    config.json
    metadata_2col.csv          # wav|text
    wavs/
      <name>.wav

Key properties:
- Local-only (no paid APIs)
- Uses ffmpeg to convert audio to mono PCM WAV at target sample rate
- Normalizes text (newlines -> spaces, collapses whitespace, strips '|')

It does NOT do semantic phrase splitting. This step is about establishing a clean
"source of truth" dataset (one audio file per original utterance).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


_WS_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("|", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text


def _run_ffmpeg_convert(*, src: Path, dst: Path, sample_rate: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found in PATH")

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-c:a",
        "pcm_s16le",
        "-y",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def _safe_out_name(stem: str) -> str:
    # Keep it filesystem-safe and deterministic.
    stem = stem.strip().replace(" ", "_")
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        stem = "utt"
    return stem + ".wav"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to manifest.jsonl (lines of JSON with at least audio_path and text).",
    )
    p.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Root directory for audio_path. Default: directory containing manifest.jsonl.",
    )
    p.add_argument(
        "--out-dataset",
        required=True,
        type=Path,
        help="Output dataset dir to create.",
    )
    p.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate (Hz).")
    p.add_argument(
        "--config-template",
        type=Path,
        default=None,
        help="Optional config.json template to copy. If omitted, writes a minimal config.",
    )
    p.add_argument("--limit", type=int, default=0, help="If >0, only process first N rows (debug).")
    p.add_argument("--progress-every", type=int, default=200, help="Print progress every N rows.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="If output wav already exists, keep it and still write metadata.",
    )

    args = p.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    audio_root = (args.audio_root or manifest_path.parent).expanduser().resolve()
    out_dataset = args.out_dataset.expanduser().resolve()

    if out_dataset.exists():
        raise SystemExit(f"Output dataset already exists: {out_dataset}")

    out_wavs = out_dataset / "wavs"
    out_wavs.mkdir(parents=True, exist_ok=True)

    # config.json
    if args.config_template:
        tpl = args.config_template.expanduser().resolve()
        if not tpl.is_file():
            raise SystemExit(f"Missing config template: {tpl}")
        shutil.copy2(tpl, out_dataset / "config.json")
    else:
        minimal = {
            "audio": {"sample_rate": int(args.sample_rate)},
            "espeak": {"voice": "ru"},
        }
        (out_dataset / "config.json").write_text(
            json.dumps(minimal, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    meta_lines: list[str] = []

    total = 0
    kept = 0
    missing = 0

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            total += 1
            if args.limit and total > args.limit:
                break

            try:
                obj = json.loads(line)
            except Exception:
                continue

            audio_path = obj.get("audio_path")
            text = obj.get("text")
            if not audio_path or not text:
                continue

            src = audio_root / str(audio_path)
            if not src.is_file():
                missing += 1
                continue

            stem = Path(str(audio_path)).stem
            out_name = _safe_out_name(stem)
            dst = out_wavs / out_name

            if dst.exists() and args.skip_existing:
                pass
            else:
                _run_ffmpeg_convert(src=src, dst=dst, sample_rate=args.sample_rate)

            norm_text = _normalize_text(str(text))
            if not norm_text:
                continue

            meta_lines.append(f"{out_name}|{norm_text}")
            kept += 1

            if args.progress_every and kept % args.progress_every == 0:
                print(f"processed={total} kept={kept} missing_audio={missing}")

    (out_dataset / "metadata_2col.csv").write_text(
        "\n".join(meta_lines) + ("\n" if meta_lines else ""),
        encoding="utf-8",
    )

    print(f"Wrote dataset: {out_dataset}")
    print(f"Rows kept: {kept}")
    if missing:
        print(f"Missing audio files: {missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
