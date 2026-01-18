#!/usr/bin/env python3
"""Проверка механизма сохранения чекпоинтов"""

import os
from pathlib import Path

print("🔍 ПРОВЕРКА АВТОСОХРАНЕНИЯ ЧЕКПОИНТОВ")
print("=" * 70)

print("\n📁 Текущие чекпоинты:")
lightning_logs = Path("/workspace/piper1-gpl/lightning_logs")

# Проверяем последнюю версию обучения
versions = sorted(lightning_logs.glob("version_*"), key=lambda x: int(x.name.split("_")[1]))
if versions:
    latest = versions[-1]
    print(f"\n  Последняя версия: {latest.name}")
    
    ckpt_dir = latest / "checkpoints"
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        print(f"  Найдено чекпоинтов: {len(ckpts)}")
        for ckpt in sorted(ckpts):
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"    • {ckpt.name} ({size_mb:.1f} MB)")

print("\n⚙️  ТЕКУЩАЯ КОНФИГУРАЦИЯ:")
print()
print("  ❌ ПРОБЛЕМА: callbacks: null в config.yaml")
print("     → ModelCheckpoint не настроен явно!")
print()
print("  По умолчанию Lightning сохраняет:")
print("    • Лучший чекпоинт (по val_loss)")
print("    • Последний чекпоинт (last.ckpt)")  
print("    • Interrupt чекпоинт (при Ctrl+C)")
print()
print("  НО: Нет периодического сохранения каждые N эпох!")

print("\n✅ РЕКОМЕНДУЕМЫЕ НАСТРОЙКИ:")
print()
print("  Добавить в команду запуска:")
print()
print("  --trainer.callbacks+=ModelCheckpoint")
print("  --trainer.callbacks.dirpath=lightning_logs/checkpoints")
print("  --trainer.callbacks.filename='epoch={epoch}-step={step}-val_loss={val_loss:.4f}'")
print("  --trainer.callbacks.monitor='val_loss'")
print("  --trainer.callbacks.mode='min'")
print("  --trainer.callbacks.save_top_k=3")
print("  --trainer.callbacks.every_n_epochs=25")
print("  --trainer.callbacks.save_last=True")

print("\n🚨 ЧТО СЕЙЧАС ПРОИСХОДИТ:")
print()
print("  ✅ interrupt.ckpt сохраняется при Ctrl+C или SIGTERM")
print("  ✅ Можно восстановить обучение после прерывания")
print()
print("  ⚠️  НО: Если RunPod pod умрет внезапно (out of credits, crash)")
print("      → Потеряете весь прогресс с последней валидации!")
print()
print("  ⚠️  Валидация каждые 0.5 эпохи (val_check_interval=0.5)")
print("      → Чекпоинт может сохраниться, но не гарантировано")

print("\n💡 РЕШЕНИЕ:")
print()
print("  1. Добавить явное сохранение каждые 25 эпох")
print("  2. Сохранять топ-3 лучших модели")
print("  3. Всегда сохранять last.ckpt")
print()
print("  Создать улучшенный скрипт train_from_749_safe.sh?")

print("\n" + "=" * 70)
