#!/usr/bin/env python3
"""
Simple training launcher that works without Docker
Usage: python start_training.py
"""
import os
import sys
import subprocess
import json
from pathlib import Path

def create_test_dataset():
    """Create minimal test dataset"""
    test_dir = Path("/workspace/piper1-gpl/test_dataset")
    test_dir.mkdir(exist_ok=True)
    
    wavs_dir = test_dir / "wavs"
    wavs_dir.mkdir(exist_ok=True)
    
    cache_dir = test_dir / ".cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Create config.json
    config_path = test_dir / "config.json"
    if not config_path.exists():
        config = {
            "audio": {"sample_rate": 22050},
            "espeak": {"voice": "ru"},
            "inference": {
                "noise_scale": 0.667,
                "length_scale": 1.0,
                "noise_w": 0.8
            },
            "phoneme_type": "espeak",
            "phoneme_map": {},
            "phoneme_id_map": {}
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✓ Created {config_path}")
    
    # Create metadata.csv
    metadata_path = test_dir / "metadata.csv"
    if not metadata_path.exists():
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("test1.wav|Привет, это тестовое обучение.\n")
            f.write("test2.wav|Проверка работы системы обучения.\n")
            f.write("test3.wav|Нейросетевой синтез речи на русском языке.\n")
        print(f"✓ Created {metadata_path}")
    
    # Create dummy wav files (you need actual audio for real training)
    for i in range(1, 4):
        wav_path = wavs_dir / f"test{i}.wav"
        if not wav_path.exists():
            # Try to create silence with sox or ffmpeg
            try:
                subprocess.run([
                    "sox", "-n", "-r", "22050", "-c", "1", 
                    str(wav_path), "trim", "0.0", "1.0"
                ], check=True, capture_output=True)
                print(f"✓ Created {wav_path}")
            except:
                try:
                    subprocess.run([
                        "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
                        "-t", "1", "-y", str(wav_path)
                    ], check=True, capture_output=True, stderr=subprocess.DEVNULL)
                    print(f"✓ Created {wav_path}")
                except:
                    print(f"⚠ Could not create {wav_path} - install sox or ffmpeg")
    
    return test_dir

def start_training(checkpoint=None, batch_size=32, num_gpus=2, max_epochs=100):
    """Start training with specified parameters"""
    
    # Create test dataset
    test_dir = create_test_dataset()
    
    # Build command
    cmd = [
        sys.executable, "-m", "piper.train", "fit",
        f"--data.config_path={test_dir}/config.json",
        "--data.voice_name=test_voice",
        f"--data.csv_path={test_dir}/metadata.csv",
        f"--data.audio_dir={test_dir}/wavs",
        "--model.sample_rate=22050",
        "--data.espeak_voice=ru",
        f"--data.cache_dir={test_dir}/.cache",
        f"--data.batch_size={batch_size}",
        "--data.num_workers=4",
        "--trainer.precision=16-mixed",
        f"--trainer.max_epochs={max_epochs}",
        f"--trainer.devices={num_gpus}",
        "--trainer.accelerator=gpu",
        "--trainer.check_val_every_n_epoch=1",
        "--trainer.accumulate_grad_batches=1"
    ]
    
    if checkpoint and os.path.exists(checkpoint):
        cmd.append(f"--ckpt_path={checkpoint}")
        print(f"✓ Resuming from checkpoint: {checkpoint}")
    
    if num_gpus > 1:
        cmd.append("--trainer.strategy=ddp_find_unused_parameters_true")
    
    print("\n" + "="*60)
    print("TRAINING COMMAND:")
    print(" ".join(cmd))
    print("="*60 + "\n")
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        sys.exit(130)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Piper TTS training")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs (default: 2)")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs (default: 100)")
    
    args = parser.parse_args()
    
    start_training(
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        max_epochs=args.max_epochs
    )
