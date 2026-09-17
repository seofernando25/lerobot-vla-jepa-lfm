from __future__ import annotations

import torch
from lerobot.policies.vla_jepa.processor_vla_jepa import make_vla_jepa_pre_post_processors

from .configuration_vla_jepa_lfm import VLAJEPALFMConfig


def make_vla_jepa_lfm_pre_post_processors(
    config: VLAJEPALFMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
):
    return make_vla_jepa_pre_post_processors(config, dataset_stats)
