# AGENTS.md

## Purpose

This is a computer-vision class-project derivative of `ginwind/VLA-JEPA`. The research question is whether LFM2.5-VL-450M can replace the original Qwen3-VL-2B backbone while keeping the VLA-JEPA action/world-model stack controlled.

## Ground rules

- Preserve upstream attribution, git history, and file headers.
- Never commit model weights, checkpoints, datasets, simulator videos, virtual environments, caches, credentials, or machine-specific absolute paths.
- Keep generated artifacts under ignored directories such as `models/`, `data/`, `runs/`, `.venv*`, and `.tools/`.
- Prefer small commits. Do not mix experiments with unrelated refactors.
- Do not change reference behavior silently; use config switches for experimental behavior.

## Agent-to-agent coordination

Two agents may work on separate machines. Use [`AGENT_BOARD.md`](AGENT_BOARD.md) for lightweight coordination.

- `F` = agent acting for GitHub user `seofernando25`; `N` = Noah's agent.
- **Always check the latest remote board, not only the local file.** Safe read: `git fetch classproject main && git show classproject/main:AGENT_BOARD.md`. This does not modify or stash the current working tree.
- Re-check the latest remote board at session start, before `CLM`, before starting shared work, and before a hand-off/final push.
- Use `CLM` before taking work that the other agent could duplicate; use `DONE`/`BLK` when its state changes.
- Board log entries are append-only. Never edit, delete, or reorder an existing entry; corrections use `CORR ref=<id>`.
- **Board-only commits may be pushed to `main` independently of code** so F/N coordination stays near-real-time. If the local branch tip already matches `classproject/main` and code edits are only uncommitted, stage/commit only `AGENT_BOARD.md`; unrelated unstaged code remains untouched and does not need to be stashed.
- If local history contains unpushed non-board commits, do **not** push that branch just to update the board. Use a clean temporary worktree based on `classproject/main`, append/commit only the board there, push it to `main`, then remove the worktree. This prevents unfinished code commits from hitching a ride.
- A board commit on `main` is coordination only; it is not approval or confirmation of unfinished code/experiment changes. Never force-push the board.
- On a board conflict, preserve the remote log verbatim and append the local unsent entry after the remote tail.
- Keep coordination on the board; keep experiment evidence/configs/metrics under `experiments/`.

## Environment

- Python 3.10.
- Use `uv` for the class-project workflow.
- LFM setup: `scripts/setup_uv_lfm.sh`.
- Single-GPU config: `starVLA/config/deepseeds/deepspeed_single_gpu.yaml`.
- Prefer SDPA; FlashAttention must not be required.

## Canonical configs

Controlled launch configs live in `scripts/configs/class_project/`:

- `qwen3_2b_baseline.yaml`
- `lfm25_450m_frozen.yaml`
- `lfm25_450m_rmsmlp_last4.yaml`

Launch with:

```bash
bash scripts/run_class_project.sh <config>
```

## Experiment structure

Every new experiment must be recorded under:

```text
experiments/exp-XXX-short-name/
  README.md
  configs/
    arm-a.yaml
    arm-b.yaml
  metrics.json        # optional; scalar/small structured results only
```

Follow `experiments/README.md`.

For every experiment:

1. State one testable **hypothesis**.
2. Name one primary **independent variable** when possible.
3. List all important **controls**.
4. Copy the exact config for every arm into `configs/` before reporting results.
5. Record the git commit SHA used for the run. If the run predates a clean commit, state that explicitly.
6. Record dataset/mix, seed, optimizer steps, effective batch size, scheduler, precision, pretrained initialization, and hardware.
7. Define every metric and where it is computed.
8. Report raw values before percentages or interpretation.
9. Separate training-batch diagnostics from held-out or simulator metrics.
10. State limitations and the smallest useful next experiment.

Never overwrite an old experiment record. A changed hypothesis or protocol gets a new sequential experiment ID.

## Comparison discipline

When comparing backbones, keep these matched unless they are the independent variable:

- LIBERO mixture and preprocessing.
- Seed.
- Number of optimizer steps.
- Micro/effective batch size.
- Warmup and scheduler.
- Action horizon and state representation.
- V-JEPA encoder.
- Action/world-model initialization.
- Evaluation cadence and protocol.

If any differ, say so prominently in the experiment record.

## Metrics and evaluation

- The training field `mse_score` is **not true MSE**; it is a normalized Euclidean-distance-style score.
- Periodic action metrics are computed on training batches. Treat them as optimization diagnostics, not generalization estimates.
- The 10-task, one-rollout LIBERO-spatial run is a smoke test, not the official LIBERO benchmark.
- Do not claim improved LIBERO success without a matched simulator evaluation.
- Report exact task suite, rollout count, seed, config, checkpoint, and hardware.

## LFM integration

- LFM code: `starVLA/model/modules/vlm/LFM2_5.py`.
- LFM native text hidden size is 1024; the pretrained VLA-JEPA conditioning interface is 2048.
- Keep the dimensionality bridge explicit.
- The residual MLP adapter starts from the earlier identity-like linear bridge behavior. Preserve this unless adapter initialization is the experiment.
- Keep the SigLIP vision tower frozen by default.

## Before a long run

1. Confirm the code/config change matches the hypothesis.
2. Create the experiment directory and copy exact configs.
3. Record the commit SHA.
4. Run a forward/backward smoke test.
5. Check trainable parameter groups.
6. Check GPU memory and stale processes.

## Before pushing

- Run `git status` and stage explicitly.
- Scan staged files for `/home/`, credentials, and large binaries.
- Never commit `models/`, `data/`, `runs/`, `.tools/`, `.venv*`, `*.pt`, or `*.safetensors`.
- Keep root `README.md` concise; methodology/results belong under `experiments/`.
- Link to model/data sources instead of redistributing weights.
