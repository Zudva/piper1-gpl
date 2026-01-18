#!/usr/bin/env python3
"""Найти все чекпоинты в lightning_logs"""
import os
from pathlib import Path

print("🔍 Поиск чекпоинтов...")
print()

lightning_logs = Path("/workspace/piper1-gpl/lightning_logs")

for version_dir in sorted(lightning_logs.glob("version_*")):
    ckpt_dir = version_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            print(f"📁 {version_dir.name}/checkpoints/")
            for ckpt in sorted(ckpts):
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                print(f"  • {ckpt.name}")
                print(f"    Размер: {size_mb:.1f} MB")
                
                # Ищем эпоху 749
                if "749" in ckpt.name or "epoch=749" in ckpt.name:
                    print(f"    ⭐ НАЙДЕН ЧЕКПОИНТ ЭПОХИ 749!")
                    print(f"    Путь: {ckpt}")
            print()
