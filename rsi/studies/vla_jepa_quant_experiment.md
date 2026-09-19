# Experiment: Quantized VLA-JEPA Benchmark & Initialization (`vrfai/vla-jepa-libero`)

## Overview

This experiment branch (`exp-vla-jepa-quant`) adapts the LeRobot VLA-JEPA + LFM2.5-VL research framework to support the quantized GGUF release of VLA-JEPA ([`vrfai/vla-jepa-libero`](https://huggingface.co/vrfai/vla-jepa-libero)) as:
1. An alternative reference baseline for closed-loop confirmation and evaluation on the LIBERO-Spatial benchmark.
2. A compatible initialization source for the DiT flow-matching action head via hybrid Safetensors/GGUF weight loading.

## Model Details

* **Base Model**: `lerobot/VLA-JEPA-LIBERO` converted to GGUF format (`vla-jepa.gguf`, 4.25 GiB).
* **Backbone**: Qwen3-VL-2B-Instruct vision-language transformer.
* **Action Head**: DiT-B flow-matching action head (denoising a 7-step action chunk over 4 flow-matching steps).
* **World Model**: Dropped in the GGUF conversion (as documented by upstream, the V-JEPA predictor is off the inference critical path).

## Changes Made

1. **`src/lerobot_policy_vla_jepa_lfm/modeling_vla_jepa_lfm.py`**:
   * Enhanced `_load_compatible_vla_jepa_weights` to detect and load `.gguf` format weights using `gguf.GGUFReader`.
   * Automatically resolves Hugging Face Hub repos providing `.gguf` bundles.
   * Handles dropped components (e.g. world model) gracefully when transferring compatible action-head weights.

2. **`src/lerobot_policy_vla_jepa_lfm/configuration_vla_jepa_lfm.py`**:
   * Updated `init_from_vla_jepa` default target to `vrfai/vla-jepa-libero`.

3. **`rsi/confirmation.json`**:
   * Updated `qwen_checkpoint` to `vrfai/vla-jepa-libero` to serve as the evaluation baseline template for edge/quantized benchmarks.

4. **`pyproject.toml`**:
   * Added `gguf>=0.10.0` to project dependencies.

## Hardware & Resource Considerations

* **Local Compute Constraint**: The local GPU on this host is an NVIDIA GeForce GTX 1650 with 4 GB VRAM.
* **Memory Footprint**: `vla-jepa.gguf` occupies 4.25 GiB in weights alone; running full policy inference + simulation environments exceeds 4 GB and triggers CUDA Out-Of-Memory (OOM).
* **Execution Strategy**:
  * Code changes are committed and pushed to `exp-vla-jepa-quant`.
  * Real benchmark rollouts and confirmation runs are deferred until execution on a workstation/cluster GPU (e.g., RTX 3090 / A100) or via CPU execution.
