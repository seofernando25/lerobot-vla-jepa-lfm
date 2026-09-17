#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv .venv-lfm --python 3.10
uv pip install --python .venv-lfm/bin/python -r requirements-lfm.txt
uv pip install --python .venv-lfm/bin/python -e .
.venv-lfm/bin/python - <<'PY2'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
PY2
