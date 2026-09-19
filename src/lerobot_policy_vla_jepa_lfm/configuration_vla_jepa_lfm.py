from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig


@PreTrainedConfig.register_subclass("vla_jepa_lfm")
@dataclass
class VLAJEPALFMConfig(VLAJEPAConfig):
    """VLA-JEPA with LFM2.5-VL replacing Qwen3-VL.

    The action head and JEPA predictor keep their 2048-D conditioning interface. LFM's native
    1024-D decoder state is projected through a small adapter.
    """

    lfm_model_name: str = "LiquidAI/LFM2.5-VL-450M"
    conditioning_dim: int = 2048
    adapter_type: str = "linear"  # linear | residual_mlp

    # LFM stays frozen by default. These switches exist for explicit later ablations.
    unfreeze_last_n: int = 0
    train_multimodal_projector: bool = False
    freeze_vision_tower: bool = True

    # Transfer only architecture-compatible VLA-JEPA modules from official checkpoints or quantized variants (e.g. vrfai/vla-jepa-libero).
    init_from_vla_jepa: str | None = "vrfai/vla-jepa-libero"
    init_prefixes: tuple[str, ...] = ("model.action_model.", "model.video_predictor.")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.adapter_type not in {"linear", "residual_mlp"}:
            raise ValueError("adapter_type must be 'linear' or 'residual_mlp'")
        if self.conditioning_dim <= 0:
            raise ValueError("conditioning_dim must be positive")
        if not 0 <= self.unfreeze_last_n <= 16:
            raise ValueError("unfreeze_last_n must be in [0, 16] for LFM2.5-VL-450M")
        if self.freeze_qwen:
            raise ValueError("freeze_qwen is a Qwen-only option; use the LFM trainability fields")
