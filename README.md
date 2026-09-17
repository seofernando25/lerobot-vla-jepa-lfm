# LeRobot VLA-JEPA: Qwen3 vs LFM2.5-VL

Computer vision class project testing whether **LFM2.5-VL-450M** can replace the original **Qwen3-VL-2B** backbone in VLA-JEPA.

The reference implementation and checkpoints now come directly from [Hugging Face LeRobot](https://github.com/huggingface/lerobot). This repo contains only our out-of-tree LeRobot policy plugin, experiment records, and coordination tooling.

## Design

- **Reference:** upstream LeRobot `vla_jepa` + `lerobot/VLA-JEPA-*` checkpoints.
- **Variant:** `vla_jepa_lfm`, using pretrained LFM2.5-VL-450M.
- **Control:** reuse LeRobot's VLA-JEPA action head, V-JEPA2 world model, preprocessing, training, and evaluation pipeline.
- **Bridge:** project LFM's 1024-D decoder state into VLA-JEPA's 2048-D conditioning interface.

## Setup

```bash
uv sync --extra dev
uv run python scripts/smoke_plugin.py
uv run pytest
```

Python 3.12+ is required by the pinned LeRobot version.

## Official Qwen baseline

Install simulator dependencies on Linux:

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --extra dev --extra eval
```

Then evaluate:

```bash
uv run lerobot-eval   --policy.path=lerobot/VLA-JEPA-LIBERO   --env.type=libero   --env.task=libero_spatial   --eval.n_episodes=10   --eval.batch_size=5
```

This evaluates the untouched upstream Qwen3 checkpoint. Larger, matched evaluations are recorded under `experiments/`.

## LFM training

```bash
uv run lerobot-train   --policy.type=vla_jepa_lfm   --policy.init_from_vla_jepa=lerobot/VLA-JEPA-Pretrain   --policy.adapter_type=linear   --dataset.repo_id=HuggingFaceVLA/libero   --steps=1000
```

`init_from_vla_jepa` transfers only architecture-compatible action/world-model tensors; Qwen weights are never loaded into LFM.

## Experiments

- `EXP-001` — legacy ginwind/starVLA pilot; preserved for history.
- `EXP-002` — LeRobot official baseline reproduction (planned/current).

See [`experiments/`](experiments/) for hypotheses, immutable settings, run records, metrics, and limitations.

## Legacy

The pre-pivot ginwind-derived tree is preserved at Git tag **`legacy-ginwind-exp001`**. It is intentionally not carried on `main`.

## Attribution

VLA-JEPA: Sun et al. (2026). LeRobot is maintained by Hugging Face. LFM2.5-VL is by Liquid AI. This project does not redistribute model weights or datasets.
