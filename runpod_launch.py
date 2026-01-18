#!/usr/bin/env python3
"""
Environment check and training launcher for RunPod
"""
import os
import sys
import subprocess
from pathlib import Path
import json

def run_cmd(cmd):
    """Run command and return output"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except:
        return "ERROR"

def check_environment():
    """Check current environment"""
    print("="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)
    
    print(f"\n1. User: {run_cmd('whoami')}")
    print(f"2. Hostname: {run_cmd('hostname')}")
    print(f"3. OS: {run_cmd('cat /etc/os-release | head -1')}")
    print(f"4. Python: {run_cmd('python3 --version')}")
    print(f"5. PyTorch: {run_cmd('python3 -c \"import torch; print(torch.__version__)\"')}")
    print(f"6. CUDA Available: {run_cmd('python3 -c \"import torch; print(torch.cuda.is_available())\"')}")
    print(f"7. GPU Count: {run_cmd('python3 -c \"import torch; print(torch.cuda.device_count())\"')}")
    
    # Check GPUs
    gpu_info = run_cmd("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader")
    if gpu_info and gpu_info != "ERROR":
        print(f"\n8. GPUs:")
        for line in gpu_info.split('\n'):
            print(f"   {line}")
    
    print("\n" + "="*60)

def check_dataset():
    """Check dataset availability"""
    print("\nDATASET CHECK")
    print("="*60)
    
    # Check common dataset locations (prefer explicit env override)
    env_dataset = os.environ.get("DATA_DIR") or os.environ.get("PIPER_DATASET_DIR")
    dataset_paths = []
    if env_dataset:
        dataset_paths.append(env_dataset)
    dataset_paths.extend([
        "/workspace/datasets/felix_mirage",
        "/data/felix_mirage",
        "/data",
        str(Path.cwd() / "datasets" / "felix_mirage"),
    ])
    
    dataset_found = None
    for path in dataset_paths:
        if Path(path).exists():
            print(f"✓ Found dataset at: {path}")
            dataset_found = path
            
            # Check contents
            config = Path(path) / "config.json"
            metadata = Path(path) / "metadata_2col.csv"
            wavs = Path(path) / "wavs"
            
            print(f"  - config.json: {'✓' if config.exists() else '✗'}")
            print(f"  - metadata_2col.csv: {'✓' if metadata.exists() else '✗'}")
            print(f"  - wavs/: {'✓' if wavs.exists() else '✗'}")
            
            if wavs.exists():
                wav_count = len(list(wavs.glob("*.wav")))
                print(f"  - WAV files: {wav_count}")
            
            break
        else:
            print(f"✗ Not found: {path}")
    
    print("="*60)
    return dataset_found

def check_checkpoints():
    """Check available checkpoints"""
    print("\nCHECKPOINT CHECK")
    print("="*60)
    
    logs_dir = Path.cwd() / "lightning_logs"
    if not logs_dir.exists():
        print("✗ No lightning_logs directory")
        return None
    
    checkpoints = list(logs_dir.rglob("*.ckpt"))
    if not checkpoints:
        print("✗ No checkpoints found")
        return None
    
    print(f"✓ Found {len(checkpoints)} checkpoint(s):")
    for ckpt in sorted(checkpoints)[-5:]:  # Show last 5
        size_mb = ckpt.stat().st_size / (1024*1024)
        print(f"  - {ckpt.relative_to(logs_dir)} ({size_mb:.1f} MB)")
    
    latest = sorted(checkpoints)[-1]
    print(f"\n✓ Latest: {latest}")
    print("="*60)
    return str(latest)

def start_training(dataset_path, checkpoint=None):
    """Start training"""
    print("\nSTARTING TRAINING")
    print("="*60)
    
    if not dataset_path:
        print("✗ No dataset found - creating test dataset")
        # Use start_training.py to create test dataset
        subprocess.run([sys.executable, "start_training.py"], check=True)
        return
    
    # Build training command
    cmd = [
        sys.executable, "-m", "piper.train", "fit",
        f"--data.config_path={dataset_path}/config.json",
        "--data.voice_name=felix_mirage",
        f"--data.csv_path={dataset_path}/metadata_2col.csv",
        f"--data.audio_dir={dataset_path}/wavs",
        "--model.sample_rate=22050",
        "--data.espeak_voice=ru",
        f"--data.cache_dir={dataset_path}/.cache",
        "--data.batch_size=80",
        "--data.num_workers=4",
        "--trainer.precision=16-mixed",
        "--trainer.max_epochs=10000",
        "--trainer.devices=2",
        "--trainer.accelerator=gpu",
        "--trainer.strategy=ddp_find_unused_parameters_true",
        "--trainer.check_val_every_n_epoch=1",
        "--trainer.accumulate_grad_batches=1"
    ]
    
    if checkpoint:
        cmd.append(f"--ckpt_path={checkpoint}")
        print(f"✓ Resuming from: {checkpoint}")
    
    print("\nCommand:")
    print(" ".join(cmd))
    print("\n" + "="*60)
    print("\nPress Ctrl+C to stop training\n")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Training completed!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        sys.exit(130)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RunPod training launcher")
    parser.add_argument("--check-only", action="store_true", help="Only check environment")
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    
    args = parser.parse_args()
    
    # Change to project directory (portable)
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)
    
    # Activate venv if exists
    venv_activate = Path(".venv/bin/activate")
    if venv_activate.exists():
        # Source it by modifying PATH
        venv_bin = str(Path(".venv/bin").absolute())
        os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
        os.environ["VIRTUAL_ENV"] = str(Path(".venv").absolute())
    
    check_environment()
    dataset_path = args.dataset or check_dataset()
    checkpoint = args.checkpoint
    
    if args.resume:
        checkpoint = check_checkpoints()
    
    if args.check_only:
        print("\n✓ Environment check completed")
        sys.exit(0)
    
    start_training(dataset_path, checkpoint)
