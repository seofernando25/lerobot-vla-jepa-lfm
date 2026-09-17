# AGENTS.md

## Purpose

This repository is a computer-vision class-project derivative of `ginwind/VLA-JEPA`. The project compares the original Qwen3-VL backbone with LFM2.5-VL-450M on LIBERO while keeping the VLA-JEPA action/world-model stack as controlled as possible.

## Ground rules

- Preserve upstream VLA-JEPA attribution, git history, and existing file headers. Do not present upstream work as original work from this project.
- Do not add or change a license for upstream code unless the upstream licensing situation has been resolved by the maintainers.
- Never commit model weights, checkpoints, datasets, simulator recordings, virtual environments, caches, credentials, API tokens, or machine-specific absolute paths.
- Keep generated artifacts under ignored directories such as `models/`, `data/`, `runs/`, `.venv*`, or `.tools/`.
- Prefer small, reviewable commits. Do not mix experiment results with unrelated refactors.

## Environment

- Python: 3.10.
- Use `uv`, not Conda, for the class-project workflow.
- LFM2.5-VL requires the dependencies in `requirements-lfm.txt`; use `scripts/setup_uv_lfm.sh`.
- The intended reference hardware is a single NVIDIA GPU with BF16 support. The supplied DeepSpeed/Accelerate config is `starVLA/config/deepseeds/deepspeed_single_gpu.yaml`.
- Do not make FlashAttention a hard requirement. The project defaults to SDPA where supported.

## Experiment configs

The controlled class-project configs are under `scripts/configs/class_project/`:

- `qwen3_2b_baseline.yaml` — Qwen3-VL-2B reference.
- `lfm25_450m_frozen.yaml` — frozen LFM2.5-VL-450M with a 1024 -> 2048 bridge.
- `lfm25_450m_rmsmlp_last4.yaml` — residual RMSNorm/MLP bridge plus the final four LFM decoder blocks and multimodal projector trainable.

Use `scripts/run_class_project.sh <config>` to launch the intended single-GPU setup.

## Comparison discipline

When comparing backbones, keep the following matched unless the experiment explicitly studies one of them:

- LIBERO mixture and preprocessing.
- Seed.
- Number of optimizer steps.
- Micro-batch size / effective batch size.
- Warmup and scheduler.
- Action horizon and state representation.
- V-JEPA encoder and action/world-model initialization.
- Evaluation cadence.

If one of these changes, state it prominently in the result report.

## Metrics and evaluation

- The training script field named `mse_score` is **not true mean-squared error** in this codebase; it is a normalized Euclidean-distance-style score. Do not relabel it as MSE in reports.
- Periodic action metrics are computed on a training batch, not a held-out validation set. Treat them as optimization diagnostics, not generalization estimates.
- The 10-task, one-rollout-per-task LIBERO-spatial check used in this project is a smoke/pilot evaluation, **not** the official LIBERO benchmark protocol.
- Do not claim an improvement in LIBERO success rate unless a properly matched simulator evaluation was run.
- Record the exact config, checkpoint, seed, task suite, trial count, and relevant hardware for reported experiments.

## LFM2.5-VL integration

- The LFM implementation lives in `starVLA/model/modules/vlm/LFM2_5.py`.
- LFM2.5-VL native text hidden size is 1024; the pretrained VLA-JEPA action/world-model interface expects 2048-dimensional conditioning. Keep that dimensionality boundary explicit.
- The residual MLP adapter is initialized so its step-0 behavior matches the earlier linear bridge. Preserve this conservative initialization unless the experiment is specifically about adapter initialization.
- When partially fine-tuning LFM, keep the SigLIP vision tower frozen by default.

## Code changes

Before changing model code:

1. Identify whether the change affects only LFM or also the Qwen3 reference.
2. Avoid silently changing reference behavior.
3. Add a config switch when a behavior is experimental.
4. Run at least a forward/backward smoke test before a longer training run.
5. Check GPU memory and remove stale policy/training processes before interpreting OOMs.

## Publication hygiene

Before pushing:

- Run `git status` and stage files explicitly.
- Check staged files for `/home/`, tokens, keys, passwords, and large binaries.
- Do not commit `models/`, `data/`, `runs/`, `.tools/`, `.venv*`, `__pycache__/`, or `*.pt`/`*.safetensors`.
- Keep README results honest about sample size and protocol limitations.
- Link to the original VLA-JEPA repository and the pretrained model sources instead of redistributing their weights.

## Collaboration

Changes intended for the class report should be understandable without access to the original workstation. Prefer portable relative paths, documented commands, and reproducible configs.
