# VLA-JEPA + LFM2.5-VL on LIBERO

Computer vision course project exploring whether a much smaller pretrained vision-language model can replace the Qwen3-VL backbone in **VLA-JEPA** for robot action prediction.

This repository is a derivative of [ginwind/VLA-JEPA](https://github.com/ginwind/VLA-JEPA). The upstream git history is preserved. The project adds an LFM2.5-VL-450M backbone path, feature adapters, single-GPU/`uv` setup, and LIBERO experiments. **Model weights, datasets, checkpoints, and videos are intentionally not committed.**

## Research question

The original VLA-JEPA setup uses Qwen3-VL-2B as its VLM. We test whether [LiquidAI/LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) can provide useful vision-language conditioning with substantially fewer parameters and lower memory use.

LFM2.5-VL emits 1024-dimensional text states while the pretrained VLA-JEPA action/world-model stack expects 2048 dimensions. Two variants are included:

1. **Frozen LFM + linear bridge**: a trainable 1024→2048 projection while LFM stays frozen.
2. **Adapted LFM**: an RMSNorm + residual two-layer MLP bridge, with the LFM multimodal projector and final four language blocks unfrozen at small learning rates. The residual branch is zero-initialized so the adapter starts from the successful linear/identity-like mapping.

## Main code changes

- `starVLA/model/modules/vlm/LFM2_5.py` — LFM2.5-VL interface and adapters.
- `starVLA/model/modules/vlm/__init__.py` — VLM dispatch for LFM2.5-VL.
- `starVLA/dataloader/__init__.py` — configurable DataLoader workers/prefetch.
- `examples/LIBERO/model2libero_interface.py` — fixes singleton robot-state shape during policy serving.
- `starVLA/model/modules/vlm/QWen3.py` / `QWen2_5.py` — configurable SDPA attention fallback instead of requiring FlashAttention2.
- `scripts/configs/class_project/` — exact 1,000-step configs used for the comparison.

## Diagnostic results

All three runs below used the LIBERO mixture, seed 42, batch size 2, 1,000 optimizer steps, the same pretrained VLA-JEPA action/world-model initialization, and evaluations at steps 250/500/750/1000 on an RTX 3090 24 GB.

| 1,000-step endpoint | Qwen3-VL-2B | LFM2.5-450M frozen | LFM2.5-450M adapted |
|---|---:|---:|---:|
| Action MAE ↓ | 0.2070 | 0.2231 | **0.1870** |
| Normalized distance ↓ | 0.03261 | 0.03336 | **0.03103** |
| Action loss ↓ | **0.0775** | 0.1426 | 0.1684 |
| JEPA loss ↓ | **0.13178** | 0.13192 | 0.13230 |
| Mean model step | 0.469 s | **0.431 s** | 0.469 s |
| Checkpoint size | 5.74 GiB | 2.16 GiB | **2.17 GiB** |
| Total model parameters | 2.77B | 1.09B | 1.10B |
| Trainable parameters | 316.8M | 318.9M | 404.9M |
| LIBERO-spatial pilot | 0/10 | 0/10 | 0/10 |

The adapted LFM endpoint had 16.2% lower MAE than frozen LFM and 9.7% lower MAE than the Qwen3 endpoint in this diagnostic run. The world-model loss stayed nearly unchanged, suggesting most of the difference is in action conditioning rather than the JEPA objective.

### Important evaluation caveat

These are **course-project diagnostic results, not an official LIBERO benchmark reproduction**. The training script's periodic action metrics are computed on another training batch, not a held-out validation set. The closed-loop LIBERO result used only one rollout per each of the 10 spatial tasks; all three 1,000-step checkpoints scored 0/10. A proper report should use a held-out/fixed evaluation panel and the official multi-rollout LIBERO protocol.

## Hardware observations

On an RTX 3090 24 GB, frozen LFM reached batch 16 at ~19.6 GiB peak VRAM and 100% utilization. The adapted last-four-block version fit batch 8 at ~17.5 GiB and 100% utilization. For the accuracy comparison above, batch 2 was kept fixed to match Qwen3.

## Setup with `uv`

Python 3.10 was used.

```bash
git clone https://github.com/seofernando25/vla-jepa-lfm25-libero.git
cd vla-jepa-lfm25-libero

bash scripts/setup_uv_lfm.sh
```

The LFM path uses Transformers 5.x because LFM2.5's tokenizer/processor metadata requires the newer API. The original upstream `requirements.txt` is retained for the Qwen-oriented environment; `requirements-lfm.txt` is the tested LFM environment.

## Download pretrained assets

Create the expected local directories (all are ignored by git):

```bash
mkdir -p models data runs
```

Required assets:

- `models/LFM2.5-VL-450M` — `LiquidAI/LFM2.5-VL-450M`
- `models/Qwen3-VL-2B-Instruct` — `Qwen/Qwen3-VL-2B-Instruct` for the baseline
- `models/vjepa2-vitl-fpc64-256` — `facebook/vjepa2-vitl-fpc64-256`
- `models/VLA-JEPA-Pretrain/VLA-JEPA-pretrain.pt` — upstream VLA-JEPA pretraining checkpoint
- `data/LEROBOT_LIBERO_DATA` — LIBERO LeRobot datasets used by VLA-JEPA

For example, with the Hugging Face CLI:

```bash
hf download LiquidAI/LFM2.5-VL-450M --local-dir models/LFM2.5-VL-450M
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir models/Qwen3-VL-2B-Instruct
hf download facebook/vjepa2-vitl-fpc64-256 --local-dir models/vjepa2-vitl-fpc64-256
hf download ginwind/VLA-JEPA --include 'Pretrain/*' --local-dir models/VLA-JEPA-HF

mkdir -p models/VLA-JEPA-Pretrain
ln -sf ../VLA-JEPA-HF/Pretrain/checkpoints/VLA-JEPA-pretrain.pt models/VLA-JEPA-Pretrain/VLA-JEPA-pretrain.pt
```

Follow the upstream VLA-JEPA instructions for the LIBERO LeRobot datasets and their `modality.json` metadata.

## Run the class-project experiments

Adapted LFM experiment:

```bash
bash scripts/run_class_project.sh \
  scripts/configs/class_project/lfm25_450m_rmsmlp_last4.yaml
```

Frozen-LFM ablation:

```bash
bash scripts/run_class_project.sh \
  scripts/configs/class_project/lfm25_450m_frozen.yaml
```

Qwen3 baseline config is at `scripts/configs/class_project/qwen3_2b_baseline.yaml`. For the closest reproduction of the original Qwen setup, use the upstream dependency environment (`requirements.txt`).

## Attribution and licensing

This project modifies and extends **VLA-JEPA**, which itself is based on starVLA and V-JEPA2. Please cite and credit the upstream projects when using this code.

The upstream VLA-JEPA repository currently contains inconsistent root licensing metadata (its README badge, `pyproject.toml`, and individual source-file headers do not all state the same thing, and its referenced root `LICENSE` file is absent in the checkout used here). This derivative therefore **does not assert a new blanket license over upstream code**. Refer to the [upstream VLA-JEPA repository](https://github.com/ginwind/VLA-JEPA) and individual file headers for applicable terms. LFM2.5 model weights are not redistributed here and remain subject to Liquid AI's model license.

The original upstream README from the base checkout is preserved as [`UPSTREAM_README.md`](UPSTREAM_README.md).

## References

- VLA-JEPA: *Enhancing Vision-Language-Action Model with Latent World Model*, ECCV 2026 / arXiv:2602.10098.
- Liquid AI, LFM2.5-VL-450M.
- LIBERO benchmark.
- V-JEPA2.
