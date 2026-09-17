# VLA-JEPA: Qwen3 vs LFM2.5-VL

Computer vision class project comparing the **original Qwen3-VL-2B backbone** in VLA-JEPA with **LFM2.5-VL-450M** on LIBERO.

This repository is derived from [ginwind/VLA-JEPA](https://github.com/ginwind/VLA-JEPA). The original upstream README is preserved as [`ORIGINAL_README.md`](ORIGINAL_README.md).

## Question

Can a much smaller pretrained VLM replace Qwen3-VL-2B in VLA-JEPA while preserving useful action conditioning and reducing compute/memory cost?

We compare:

- **Qwen3-VL-2B** — original VLA-JEPA reference.
- **LFM2.5-VL-450M, frozen** — 1024→2048 bridge into the pretrained VLA-JEPA action/world-model stack.
- **LFM2.5-VL-450M, adapted** — residual RMSNorm/MLP bridge plus the final four LFM decoder blocks and multimodal projector trainable.

## Preliminary results

Matched 1,000-step LIBERO runs, seed 42, batch size 2, same VLA-JEPA action/world-model initialization:

| Endpoint metric | Qwen3-VL-2B | LFM frozen | LFM adapted |
|---|---:|---:|---:|
| Action MAE ↓ | 0.2070 | 0.2231 | **0.1870** |
| Normalized distance ↓ | 0.03261 | 0.03336 | **0.03103** |
| JEPA loss ↓ | **0.13178** | 0.13192 | 0.13230 |
| Mean model step | 0.469 s | **0.431 s** | 0.469 s |
| Total parameters | 2.77B | 1.09B | 1.10B |
| LIBERO-spatial pilot | 0/10 | 0/10 | 0/10 |

These are **preliminary diagnostics**, not an official LIBERO benchmark. Periodic action metrics are computed on training batches, and the simulator pilot used only one rollout per task.

Full protocol and experiment records live in [`experiments/`](experiments/).

## Reproduce

```bash
bash scripts/setup_uv_lfm.sh

bash scripts/run_class_project.sh   scripts/configs/class_project/qwen3_2b_baseline.yaml

bash scripts/run_class_project.sh   scripts/configs/class_project/lfm25_450m_frozen.yaml

bash scripts/run_class_project.sh   scripts/configs/class_project/lfm25_450m_rmsmlp_last4.yaml
```

Pretrained models, LIBERO data, and VLA-JEPA checkpoints are intentionally not committed. See [`ORIGINAL_README.md`](ORIGINAL_README.md) for upstream asset/setup details.

## Project layout

- `starVLA/model/modules/vlm/LFM2_5.py` — LFM2.5-VL integration and adapters.
- `scripts/configs/class_project/` — controlled experiment configs.
- `experiments/` — hypotheses, frozen settings, metrics, and conclusions for each experiment.
- `AGENTS.md` — rules for running and recording future experiments.

## Attribution

Built on [ginwind/VLA-JEPA](https://github.com/ginwind/VLA-JEPA). Please preserve upstream attribution and cite the original projects in class reports or derivative work.
