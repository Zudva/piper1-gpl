#!/usr/bin/env python3
"""
Optimal training configuration for 2x NVIDIA L40S
Based on VRAM: 2x46GB, RAM: 62GB
"""

TRAINING_CONFIG = {
    # Hardware
    "num_gpus": 2,
    "gpu_name": "NVIDIA L40S",
    "vram_per_gpu": "46GB",
    "total_ram": "62GB",
    
    # Optimal settings for dual L40S
    "batch_size": 80,  # Can go up to 96 for max speed
    "num_workers": 4,  # Safe for 62GB RAM
    "precision": "16-mixed",  # Use "bf16-mixed" if available
    "accumulate_grad_batches": 1,  # Increase to 2-4 if OOM
    
    # Multi-GPU strategy
    "strategy": "ddp_find_unused_parameters_true",  # For 2+ GPUs
    
    # Dataset
    "dataset_path": "/workspace/datasets/felix_mirage",
    "config_path": "/workspace/datasets/felix_mirage/config.json",
    "csv_path": "/workspace/datasets/felix_mirage/metadata_2col.csv",
    "audio_dir": "/workspace/datasets/felix_mirage/wavs",
    "cache_dir": "/workspace/datasets/felix_mirage/.cache",
    
    # Model
    "voice_name": "felix_mirage",
    "sample_rate": 22050,
    "espeak_voice": "ru",
    
    # Training
    "max_epochs": 10000,
    "check_val_every_n_epoch": 1,
    
    # Latest checkpoint
    "checkpoint": "lightning_logs/version_7/checkpoints/interrupt.ckpt",
}

# Performance notes
PERFORMANCE_NOTES = """
DUAL L40S OPTIMIZATION GUIDE
=============================

Current Settings (Balanced):
  - batch_size: 80
  - num_workers: 4
  - precision: 16-mixed
  
For Maximum Speed:
  - batch_size: 96
  - precision: bf16-mixed (if supported)
  - num_workers: 6
  
If Out of Memory:
  - batch_size: 64 or 48
  - accumulate_grad_batches: 2
  - num_workers: 2
  
Expected Performance:
  - ~2000-2500 steps/hour (batch_size=80)
  - ~2500-3000 steps/hour (batch_size=96)
  - Epoch time: ~30-40 minutes
"""

if __name__ == "__main__":
    import json
    print(json.dumps(TRAINING_CONFIG, indent=2, ensure_ascii=False))
    print("\n" + PERFORMANCE_NOTES)
