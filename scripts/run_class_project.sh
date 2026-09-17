#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-scripts/configs/class_project/lfm25_450m_rmsmlp_last4.yaml}"
PYTHON="${PYTHON:-.venv-lfm/bin/python}"
ACCELERATE="${ACCELERATE:-.venv-lfm/bin/accelerate}"
exec "$ACCELERATE" launch \
  --config_file starVLA/config/deepseeds/deepspeed_single_gpu.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG"
