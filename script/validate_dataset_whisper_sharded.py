#!/usr/bin/env python3
"""Run validate_dataset_full.py Whisper checks in parallel across multiple GPUs.

Whisper itself is not multi-GPU for a single transcription run, so we shard the
metadata and run multiple validator processes (one per GPU) against temporary
shard datasets with symlinked wavs.

This script is local-only and does not use paid APIs.

Example:
  .venv/bin/python script/validate_dataset_whisper_sharded.py \
    --dataset ../piper-training/datasets/felix_mirage_prepared_sr22050_seg15 \
    --gpus 0,1 \
    --whisper-model large-v3
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ShardResult:
    shard_index: int
    report_dir: Path
    exit_code: int


def _read_metadata_lines(metadata_path: Path) -> list[str]:
    lines: list[str] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # keep as-is
            lines.append(line)
    return lines


def _round_robin(lines: list[str], n: int) -> list[list[str]]:
    shards: list[list[str]] = [[] for _ in range(n)]
    for i, line in enumerate(lines):
        shards[i % n].append(line)
    return shards


def _make_shard_dataset(
    *,
    dataset_dir: Path,
    shard_dir: Path,
    metadata_lines: list[str],
) -> None:
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Copy config.json
    shutil.copy2(dataset_dir / "config.json", shard_dir / "config.json")

    # Symlink wavs
    (shard_dir / "wavs").mkdir(parents=True, exist_ok=True)
    src_wavs = dataset_dir / "wavs"
    if not src_wavs.is_dir():
        raise FileNotFoundError(f"Missing wavs/: {src_wavs}")

    for line in metadata_lines:
        wav_name = line.split("|", 1)[0]
        src = src_wavs / wav_name
        dst = shard_dir / "wavs" / wav_name
        if not src.is_file():
            # If file is missing, validator should catch; but we want this explicit.
            raise FileNotFoundError(f"Missing wav referenced by metadata: {src}")
        dst.symlink_to(src)

    # Write shard metadata
    (shard_dir / "metadata_2col.csv").write_text(
        "\n".join(metadata_lines) + ("\n" if metadata_lines else ""),
        encoding="utf-8",
    )


def _run_validator(
    *,
    venv_python: Path,
    validator_script: Path,
    dataset_dir: Path,
    report_dir: Path,
    gpu: str,
    whisper_model: str,
    whisper_device: str,
    require_whisper: bool,
    progress_every: int,
    progress_mode: str,
    whisper_backend: str,
    whisper_language: str | None,
    whisper_batch_size: int,
    whisper_num_workers: int,
    whisper_compute_type: str,
    whisper_beam_size: int,
    whisper_vad_filter: bool,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    cmd = [
        str(venv_python),
        str(validator_script),
        "--dataset",
        str(dataset_dir),
        "--report-dir",
        str(report_dir),
        "--whisper",
        "--whisper-model",
        whisper_model,
        "--whisper-device",
        whisper_device,
        "--progress-every",
        str(progress_every),
        "--progress-mode",
        progress_mode,
        "--whisper-backend",
        whisper_backend,
        "--whisper-batch-size",
        str(whisper_batch_size),
        "--whisper-num-workers",
        str(whisper_num_workers),
        "--whisper-compute-type",
        whisper_compute_type,
        "--whisper-beam-size",
        str(whisper_beam_size),
    ]
    if whisper_language:
        cmd.extend(["--whisper-language", whisper_language])
    if whisper_vad_filter:
        cmd.append("--whisper-vad-filter")
    if require_whisper:
        cmd.append("--require-whisper")

    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "validator_whisper.log"
    log_f = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Prepared dataset dir (config.json, wavs/, metadata_2col.csv)")
    parser.add_argument("--gpus", default="0,1", help="Comma-separated GPU ids to use (e.g. 0,1)")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model name (e.g. large-v3)")
    parser.add_argument("--whisper-device", default="cuda", help="Whisper device for validator (cuda/cpu)")
    parser.add_argument("--require-whisper", action="store_true", help="Fail if whisper check cannot run")
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Output report dir (default: <dataset>/reports/validation_whisper_sharded_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep temporary shard datasets under the report dir.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N rows inside each shard validator.",
    )
    parser.add_argument(
        "--progress-mode",
        choices=["none", "count", "whisper", "all"],
        default="whisper",
        help="Progress verbosity passed to shard validators.",
    )
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="Number of validator processes per GPU.")
    parser.add_argument("--whisper-backend", choices=["auto", "faster-whisper", "whisper"], default="auto")
    parser.add_argument("--whisper-language", default=None)
    parser.add_argument("--whisper-batch-size", type=int, default=8)
    parser.add_argument("--whisper-num-workers", type=int, default=2)
    parser.add_argument("--whisper-compute-type", default="float16")
    parser.add_argument("--whisper-beam-size", type=int, default=5)
    parser.add_argument("--whisper-vad-filter", action="store_true")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset).expanduser().resolve()
    metadata_path = dataset_dir / "metadata_2col.csv"
    if not metadata_path.is_file():
        raise SystemExit(f"Missing metadata_2col.csv: {metadata_path}")

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if len(gpus) < 1:
        raise SystemExit("--gpus must contain at least one gpu id")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = (
        Path(args.report_dir).expanduser().resolve()
        if args.report_dir
        else (dataset_dir / "reports" / f"validation_whisper_sharded_{ts}")
    )

    # Resolve paths to validator + venv python relative to this repo.
    repo_root = Path(__file__).resolve().parents[1]
    venv_python = repo_root / ".venv" / "bin" / "python"
    validator_script = repo_root / "script" / "validate_dataset_full.py"

    lines = _read_metadata_lines(metadata_path)
    total_workers = max(1, len(gpus) * max(1, args.workers_per_gpu))
    shards = _round_robin(lines, total_workers)

    shard_root = report_dir / "shards"
    shard_reports_root = report_dir / "shard_reports"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_reports_root.mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen] = []
    shard_results: list[ShardResult] = []

    for idx in range(len(shards)):
        gpu = gpus[idx % len(gpus)]
        shard_dataset_dir = shard_root / f"dataset_shard_{idx}"
        _make_shard_dataset(dataset_dir=dataset_dir, shard_dir=shard_dataset_dir, metadata_lines=shards[idx])

        shard_report_dir = shard_reports_root / f"shard_{idx}"
        proc = _run_validator(
            venv_python=venv_python,
            validator_script=validator_script,
            dataset_dir=shard_dataset_dir,
            report_dir=shard_report_dir,
            gpu=gpu,
            whisper_model=args.whisper_model,
            whisper_device=args.whisper_device,
            require_whisper=args.require_whisper,
            progress_every=args.progress_every,
            progress_mode=args.progress_mode,
            whisper_backend=args.whisper_backend,
            whisper_language=args.whisper_language,
            whisper_batch_size=args.whisper_batch_size,
            whisper_num_workers=args.whisper_num_workers,
            whisper_compute_type=args.whisper_compute_type,
            whisper_beam_size=args.whisper_beam_size,
            whisper_vad_filter=args.whisper_vad_filter,
        )
        procs.append(proc)

    # Wait with periodic heartbeat
    start_time = time.time()
    last_heartbeat = 0.0
    while True:
        running = [p for p in procs if p.poll() is None]
        now = time.time()
        if now - last_heartbeat >= 60:
            elapsed = int(now - start_time)
            print(
                f"[heartbeat] elapsed={elapsed}s running={len(running)}/{len(procs)}",
                flush=True,
            )
            last_heartbeat = now
        if not running:
            break
        time.sleep(2)

    for idx, proc in enumerate(procs):
        code = proc.wait()
        shard_results.append(
            ShardResult(shard_index=idx, report_dir=shard_reports_root / f"shard_{idx}", exit_code=code)
        )

    # Summarize
    summary_path = report_dir / "SUMMARY.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"dataset={dataset_dir}\n")
        f.write(f"gpus={','.join(gpus)}\n")
        f.write(f"whisper_model={args.whisper_model}\n")
        f.write(f"whisper_device={args.whisper_device}\n")
        f.write(f"require_whisper={bool(args.require_whisper)}\n")
        f.write(f"progress_every={args.progress_every}\n")
        f.write(f"progress_mode={args.progress_mode}\n")
        f.write(f"workers_per_gpu={args.workers_per_gpu}\n")
        f.write(f"whisper_backend={args.whisper_backend}\n")
        f.write(f"whisper_language={args.whisper_language}\n")
        f.write(f"whisper_batch_size={args.whisper_batch_size}\n")
        f.write(f"whisper_num_workers={args.whisper_num_workers}\n")
        f.write(f"whisper_compute_type={args.whisper_compute_type}\n")
        f.write(f"whisper_beam_size={args.whisper_beam_size}\n")
        f.write(f"whisper_vad_filter={bool(args.whisper_vad_filter)}\n")
        for r in shard_results:
            f.write(f"shard_{r.shard_index}_exit_code={r.exit_code} report_dir={r.report_dir}\n")

    if not args.keep_shards:
        shutil.rmtree(shard_root, ignore_errors=True)

    any_failed = any(r.exit_code != 0 for r in shard_results)
    if any_failed:
        print(f"Whisper sharded validation: FAIL (see {report_dir})")
        return 2

    print(f"Whisper sharded validation: PASS (see {report_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
