#!/usr/bin/env python3
"""Quick test of existing Piper model"""
import sys
import subprocess
from pathlib import Path

MODEL = "felix_mirage_epoch749.onnx"
CONFIG = "felix_mirage_epoch749.onnx.json"
OUTPUT = "test_output.wav"

if not Path(MODEL).exists():
    print(f"✗ Model not found: {MODEL}")
    sys.exit(1)

if not Path(CONFIG).exists():
    print(f"✗ Config not found: {CONFIG}")
    sys.exit(1)

text = "Привет! Это тест модели Феликс Мираж."

print(f"Synthesizing: {text}")
print(f"Model: {MODEL}")
print(f"Output: {OUTPUT}")

try:
    proc = subprocess.Popen(
        [sys.executable, "-m", "piper", 
         "--model", MODEL,
         "--config", CONFIG,
         "--output-file", OUTPUT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = proc.communicate(input=text, timeout=30)
    
    if proc.returncode == 0:
        if Path(OUTPUT).exists():
            size = Path(OUTPUT).stat().st_size
            print(f"✓ Success! Generated {OUTPUT} ({size} bytes)")
        else:
            print(f"⚠ Command succeeded but no output file")
    else:
        print(f"✗ Failed with exit code {proc.returncode}")
        if stderr:
            print("Error output:", stderr)
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    proc.kill()
    print("✗ Timeout - process killed")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
