#!/usr/bin/env python3
"""Run sharded Whisper validation with a Rich live status UI.

This wraps validate_dataset_whisper_sharded.py and shows real-time progress
from shard logs. Intended for interactive terminal use.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


def _pick_python() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _read_last_line(path: Path) -> str:
    if not path.exists():
        return "(no log yet)"
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        return lines[-1] if lines else "(empty)"
    except Exception:
        return "(read error)"


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return 0


def _gpu_status() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        return out or "(no output)"
    except Exception:
        return "(nvidia-smi unavailable)"


def _render_table(
    *,
    report_dir: Path,
    shard_count: int,
    start_time: float,
    pid: Optional[int],
) -> Table:
    table = Table(title="Whisper Validation (Rich)")
    table.add_column("Item")
    table.add_column("Value", overflow="fold")

    elapsed = int(time.time() - start_time)
    table.add_row("Report", str(report_dir))
    table.add_row("PID", str(pid) if pid else "(unknown)")
    table.add_row("Elapsed", f"{elapsed}s")

    table.add_row("GPU", _gpu_status())

    for i in range(shard_count):
        log_path = report_dir / "shard_reports" / f"shard_{i}" / "validator_whisper.log"
        size = _file_size(log_path)
        last_line = _read_last_line(log_path)
        table.add_row(f"shard_{i} log size", f"{size} bytes")
        table.add_row(f"shard_{i} last", last_line)

    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to prepared dataset (config.json, wavs/, metadata_2col.csv).",
    )
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="Comma-separated GPU ids to use (e.g. 0,1).",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Number of validator processes per GPU (parallel shards).",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        help="Whisper model name (e.g. large-v3).",
    )
    parser.add_argument(
        "--whisper-backend",
        default="auto",
        help="Whisper backend: auto, faster-whisper, or whisper.",
    )
    parser.add_argument(
        "--whisper-batch-size",
        type=int,
        default=8,
        help="Batch size for faster-whisper (higher = faster, more VRAM).",
    )
    parser.add_argument(
        "--whisper-num-workers",
        type=int,
        default=2,
        help="Number of CPU workers for faster-whisper decoding.",
    )
    parser.add_argument(
        "--whisper-compute-type",
        default="float16",
        help="Compute type for faster-whisper (float16/int8/int8_float16).",
    )
    parser.add_argument(
        "--whisper-beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (lower = faster).",
    )
    parser.add_argument(
        "--whisper-vad-filter",
        action="store_true",
        help="Enable VAD filter (faster-whisper).",
    )
    parser.add_argument(
        "--whisper-language",
        default=None,
        help="Force language code (e.g. ru).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N rows per shard.",
    )
    parser.add_argument(
        "--progress-mode",
        default="whisper",
        help="Progress mode: none|count|whisper|all.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Override report directory (default: <dataset>/reports/validation_whisper_sharded_rich_TIMESTAMP).",
    )
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=2.0,
        help="Rich UI refresh interval in seconds.",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_dir = (
        Path(args.report_dir).expanduser().resolve()
        if args.report_dir
        else dataset / "reports" / f"validation_whisper_sharded_rich_{ts}"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    shard_count = max(1, len(args.gpus.split(",")) * max(1, args.workers_per_gpu))

    runner_log = report_dir / "runner.log"

    cmd = [
        _pick_python(),
        str(Path(__file__).resolve().parent / "validate_dataset_whisper_sharded.py"),
        "--dataset",
        str(dataset),
        "--gpus",
        args.gpus,
        "--workers-per-gpu",
        str(args.workers_per_gpu),
        "--whisper-model",
        args.whisper_model,
        "--whisper-backend",
        args.whisper_backend,
        "--whisper-batch-size",
        str(args.whisper_batch_size),
        "--whisper-num-workers",
        str(args.whisper_num_workers),
        "--whisper-compute-type",
        args.whisper_compute_type,
        "--whisper-beam-size",
        str(args.whisper_beam_size),
        "--progress-every",
        str(args.progress_every),
        "--progress-mode",
        args.progress_mode,
        "--report-dir",
        str(report_dir),
        "--require-whisper",
    ]
    if args.whisper_vad_filter:
        cmd.append("--whisper-vad-filter")
    if args.whisper_language:
        cmd.extend(["--whisper-language", args.whisper_language])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with runner_log.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)

    console = Console()
    start_time = time.time()

    with Live(console=console, refresh_per_second=max(1, int(1 / args.refresh_seconds))):
        while True:
            table = _render_table(
                report_dir=report_dir,
                shard_count=shard_count,
                start_time=start_time,
                pid=proc.pid,
            )
            console.clear()
            console.print(Panel(table))
            if proc.poll() is not None:
                break
            time.sleep(args.refresh_seconds)

    exit_code = proc.wait()
    console.print(f"\nRunner finished with code {exit_code}. Report: {report_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
