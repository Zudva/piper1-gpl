#!/usr/bin/env python3
"""Quick environment check"""
import os
import subprocess
import sys
from pathlib import Path


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
            return parent
    return here.parent


print("QUICK ENVIRONMENT CHECK")
print("="*60)

# Basic info
try:
    user = subprocess.check_output("whoami", shell=True).decode().strip()
    hostname = subprocess.check_output("hostname", shell=True).decode().strip()
    print(f"User: {user}")
    print(f"Host: {hostname}")
    print(f"Working dir: {Path.cwd()}")
except:
    print("Could not get system info")

# Python
print(f"\nPython: {sys.version}")
print(f"Executable: {sys.executable}")

# PyTorch
try:
    import torch
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    print("\n✗ PyTorch not installed")

# Project structure
print(f"\nProject structure:")
repo_root = find_repo_root()
piper_dir = repo_root
print(f"  piper1-gpl: {'✓' if piper_dir.exists() else '✗'}")
print(f"  .venv: {'✓' if (piper_dir / '.venv').exists() else '✗'}")
print(f"  lightning_logs: {'✓' if (piper_dir / 'lightning_logs').exists() else '✗'}")

# Dataset locations to check
print(f"\nDataset check:")
env_dataset = os.environ.get("DATA_DIR") or os.environ.get("PIPER_DATASET_DIR")
dataset_paths = [
    env_dataset,
    "/workspace/datasets/felix_mirage",
    "/data/felix_mirage", 
    "/data",
    str(repo_root / "datasets" / "felix_mirage")
]

for path in [p for p in dataset_paths if p]:
    p = Path(path)
    if p.exists():
        print(f"  ✓ {path}")
        if (p / "wavs").exists():
            wav_count = len(list((p / "wavs").glob("*.wav")))
            print(f"    WAV files: {wav_count}")
    else:
        print(f"  ✗ {path}")

# Checkpoints
print(f"\nCheckpoints:")
ckpt_dir = piper_dir / "lightning_logs"
if ckpt_dir.exists():
    checkpoints = list(ckpt_dir.rglob("*.ckpt"))
    print(f"  Found: {len(checkpoints)}")
    if checkpoints:
        latest = sorted(checkpoints)[-1]
        print(f"  Latest: {latest.relative_to(piper_dir)}")
else:
    print("  ✗ No lightning_logs")

print("\n" + "="*60)
print("\nTo start training:")
print("  python tools/runpod/runpod_launch.py")
print("\nOr direct command:")
print("  python tools/runpod/start_training.py --batch-size 80 --num-gpus 2")
