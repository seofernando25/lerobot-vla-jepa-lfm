# EXP-001 — Qwen3 vs LFM2.5-VL (legacy)

> Historical pilot on the pre-LeRobot ginwind/starVLA-derived stack. The complete old tree is preserved at tag `legacy-ginwind-exp001`.

## Hypothesis

A 450M LFM2.5-VL backbone can provide competitive action conditioning for VLA-JEPA with substantially fewer total parameters than Qwen3-VL-2B.

## Change

Backbone / adaptation strategy:

- Qwen3-VL-2B reference.
- Frozen LFM2.5-VL-450M with 1024→2048 bridge.
- LFM2.5-VL-450M with residual RMSNorm/MLP bridge and final four decoder blocks + multimodal projector trainable.

## Controls

LIBERO mixture, seed 42, batch size 2, 1,000 optimizer steps, warmup/scheduler, action horizon, robot state input, V-JEPA encoder, and pretrained action/world-model initialization were matched.

## Protocol

- Dataset: LIBERO LeRobot mixture used by VLA-JEPA.
- GPU: RTX 3090 24 GB.
- Precision: BF16.
- Evaluation points: 250, 500, 750, 1000 steps.
- Closed-loop smoke test: LIBERO-spatial, 10 tasks, one rollout per task.
- Run code state: local pre-publication working tree; the relevant class-project code was later captured in commit `a9ef11b`.

Exact arm configs are preserved in `configs/`.

The training field `mse_score` is a normalized Euclidean-distance-style score, not true MSE. Periodic action metrics are computed on training batches.

## Results

| 1,000-step endpoint | Qwen3-VL-2B | LFM frozen | LFM adapted |
|---|---:|---:|---:|
| Action MAE ↓ | 0.2070 | 0.2231 | **0.1870** |
| Normalized distance ↓ | 0.03261 | 0.03336 | **0.03103** |
| Action loss ↓ | **0.0775** | 0.1426 | 0.1684 |
| JEPA loss ↓ | **0.13178** | 0.13192 | 0.13230 |
| Mean model step | 0.469 s | **0.431 s** | 0.469 s |
| Total parameters | 2.77B | 1.09B | 1.10B |
| Trainable parameters | 316.8M | 318.9M | 404.9M |
| LIBERO-spatial pilot | 0/10 | 0/10 | 0/10 |

## Conclusion

The adapted LFM endpoint produced lower diagnostic action MAE and normalized distance while using far fewer total parameters. JEPA loss stayed nearly unchanged across backbones.

This does **not** establish better generalization or LIBERO task success: all three closed-loop smoke tests were 0/10, and the action metrics were not measured on a held-out validation set.

## Superseded by

`EXP-002` verifies the untouched official LeRobot VLA-JEPA checkpoint before any new LFM comparison.
