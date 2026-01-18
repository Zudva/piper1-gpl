#!/usr/bin/env python3
"""Мониторинг GPU во время обучения"""
import subprocess
import time
import os

print("📊 Мониторинг GPU во время обучения")
print("=" * 70)
print()
print("ℹ️  О БАЛАНСИРОВКЕ GPU:")
print()
print("Разница в памяти GPU0 (45201MB) vs GPU1 (42235MB) = ~3GB это нормально:")
print()
print("  • DDP (Distributed Data Parallel) распределяет батчи равномерно")
print("  • Но GPU 0 (rank 0) хранит дополнительные данные:")
print("    - Главный процесс логирования")
print("    - TensorBoard writer")
print("    - Валидационные метрики")
print("    - Чекпоинт менеджер")
print()
print("  • Если утилизация обоих GPU ~100%, всё работает оптимально")
print("  • Разница 3-5GB для rank 0 - это стандартное поведение")
print()
print("Если хотите максимальной точности - можно:")
print("  1. Использовать strategy='ddp' вместо 'ddp_find_unused_parameters_true'")
print("  2. Уменьшить batch_size до 76 (делится на 2 равномерно: 38 на GPU)")
print()
print("=" * 70)
print()

try:
    while True:
        os.system('clear')
        print("\n📊 GPU СТАТУС\n")
        subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', 
                       '--format=csv,noheader,nounits'])
        print("\n" + "=" * 70)
        print("Обновление каждые 5 секунд... (Ctrl+C для выхода)")
        time.sleep(5)
except KeyboardInterrupt:
    print("\n\n✅ Мониторинг остановлен")
