#!/usr/bin/env python3
"""Multi-GPU launcher for prepare_dataset_whisper_segments.py.

Splits input WAVs into shards and runs multiple workers in parallel (typically one per GPU).
Each worker writes its own shard dataset; then this launcher merges shards into a single
Piper-ready output dataset.

Local-only: uses Whisper + ffmpeg, no paid APIs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _read_config(dataset_dir: Path) -> dict:
    with (dataset_dir / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_config(out_dir: Path, cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _iter_wavs(dataset_dir: Path) -> list[Path]:
    wavs_dir = dataset_dir / "wavs"
    if not wavs_dir.is_dir():
        raise FileNotFoundError(f"Missing wavs/: {wavs_dir}")
    return sorted([p for p in wavs_dir.iterdir() if p.suffix.lower() == ".wav"])


def _chunk_round_robin(items: list[Path], n: int) -> list[list[Path]]:
    shards: list[list[Path]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        shards[i % n].append(item)
    return shards


def _write_wavs_file(path: Path, wavs: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for w in wavs:
            f.write(str(w) + "\n")


def _merge_shards(final_dir: Path, shard_dirs: list[Path]) -> None:
    (final_dir / "wavs").mkdir(parents=True, exist_ok=True)

    seen = set()
    rows: list[str] = []

    for shard in shard_dirs:
        meta = shard / "metadata_2col.csv"
        if not meta.is_file():
            continue

        # Move wavs
        wavs_dir = shard / "wavs"
        if wavs_dir.is_dir():
            for wav in wavs_dir.iterdir():
                if wav.suffix.lower() != ".wav":
                    continue
                name = wav.name
                if name in seen:
                    raise RuntimeError(f"Name collision when merging shards: {name}")
                seen.add(name)
                shutil.move(str(wav), str(final_dir / "wavs" / name))

        # Collect metadata lines
        with meta.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                rows.append(line)

    # Write merged metadata
    rows.sort()  # deterministic
    with (final_dir / "metadata_2col.csv").open("w", encoding="utf-8") as f:
        for line in rows:
            f.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Input dataset dir")
    parser.add_argument("--out-dataset", required=True, help="Final output dataset dir")
    parser.add_argument(
        "--gpus",
        default="0,1,2",
        help="Comma-separated GPU ids to use (e.g. 0,1,2).",
    )
    parser.add_argument("--whisper-model", default="medium")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--max-seg-seconds", type=float, default=15.0)
    parser.add_argument("--min-seg-seconds", type=float, default=1.0)
    parser.add_argument("--max-text-chars", type=int, default=300)
    parser.add_argument("--min-text-chars", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-shards", action="store_true")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset).expanduser().resolve()
    final_dir = Path(args.out_dataset).expanduser().resolve()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one gpu id")

    if final_dir.exists():
        if args.overwrite:
            shutil.rmtree(final_dir)
        else:
            raise SystemExit(f"Output exists (use --overwrite): {final_dir}")

    # Prepare shard dirs
    work_dir = final_dir.parent / (final_dir.name + "_work")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = work_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    wavs = _iter_wavs(dataset_dir)
    shards = _chunk_round_robin(wavs, len(gpus))

    script_path = Path(__file__).resolve().parent / "prepare_dataset_whisper_segments.py"
    python = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"

    shard_dirs: list[Path] = []
    procs: list[subprocess.Popen] = []

    for idx, gpu in enumerate(gpus):
        shard_dir = work_dir / f"shard_{idx}"
        shard_dirs.append(shard_dir)

        # Important: wavs_file must NOT live under shard_dir, because the worker
        # script will delete shard_dir on startup when --overwrite is used.
        wavs_file = splits_dir / f"wavs_shard_{idx}.txt"
        _write_wavs_file(wavs_file, shards[idx])

        if shard_dir.exists():
            shutil.rmtree(shard_dir)

        cmd = [
            str(python),
            str(script_path),
            "--dataset",
            str(dataset_dir),
            "--out-dataset",
            str(shard_dir),
            "--wavs-file",
            str(wavs_file),
            "--metadata-out",
            str(shard_dir / "metadata_2col.csv"),
            "--whisper-model",
            args.whisper_model,
            "--device",
            "cuda",
            "--language",
            args.language,
            "--sample-rate",
            str(args.sample_rate),
            "--min-seg-seconds",
            str(args.min_seg_seconds),
            "--max-seg-seconds",
            str(args.max_seg_seconds),
            "--min-text-chars",
            str(args.min_text_chars),
            "--max-text-chars",
            str(args.max_text_chars),
            "--overwrite",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu

        log_path = splits_dir / f"worker_{idx}.log"
        log_f = open(log_path, "w", encoding="utf-8")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT))

    # Wait for workers
    exit_codes = [p.wait() for p in procs]
    if any(code != 0 for code in exit_codes):
        raise SystemExit(f"One or more workers failed: {exit_codes}. See shard logs under {work_dir}.")

    # Write final config
    cfg = _read_config(dataset_dir)
    cfg.setdefault("audio", {})
    cfg["audio"]["sample_rate"] = int(args.sample_rate)
    _write_config(final_dir, cfg)

    # Merge
    _merge_shards(final_dir, shard_dirs)

    # Keep a small trace
    with (final_dir / "PREPARE_INFO.txt").open("w", encoding="utf-8") as f:
        f.write(f"source_dataset={dataset_dir}\n")
        f.write(f"gpus={','.join(gpus)}\n")
        f.write(f"whisper_model={args.whisper_model}\n")
        f.write(f"language={args.language}\n")
        f.write(f"sample_rate={args.sample_rate}\n")
        f.write(f"min_seg_seconds={args.min_seg_seconds}\n")
        f.write(f"max_seg_seconds={args.max_seg_seconds}\n")
        f.write(f"min_text_chars={args.min_text_chars}\n")
        f.write(f"max_text_chars={args.max_text_chars}\n")

    if not args.keep_shards:
        shutil.rmtree(work_dir)

    print(f"Wrote dataset: {final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
