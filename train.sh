#!/bin/bash
# Quick training start for RunPod
cd /workspace/piper1-gpl
source .venv/bin/activate
python runpod_launch.py "$@"
