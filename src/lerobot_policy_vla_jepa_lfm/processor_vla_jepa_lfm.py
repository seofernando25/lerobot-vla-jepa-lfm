from __future__ import annotations

import torch
from lerobot.policies.vla_jepa.processor_vla_jepa import make_vla_jepa_pre_post_processors
from lerobot.processor import ProcessorStep, ProcessorStepRegistry, TransitionKey

from .configuration_vla_jepa_lfm import VLAJEPALFMConfig


@ProcessorStepRegistry.register(name="vla_jepa_lfm_drop_image_pad_masks")
class DropImagePadMasksProcessorStep(ProcessorStep):
    """Drop temporal camera padding masks before upstream VLA-JEPA image preparation.

    LeRobot datasets with temporal image deltas add keys such as
    observation.images.image_is_pad. Upstream VLA-JEPA currently identifies image
    observations by the substring image and therefore tries to resize those boolean
    masks as if they were CxHxW tensors. VLA-JEPA does not consume image padding masks,
    so removing only camera *_is_pad keys preserves model semantics.
    """

    def __call__(self, transition):
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return transition
        filtered = {
            key: value
            for key, value in observation.items()
            if not (key.endswith("_is_pad") and "image" in key)
        }
        if len(filtered) == len(observation):
            return transition
        result = dict(transition)
        result[TransitionKey.OBSERVATION] = filtered
        return result

    def transform_features(self, features):
        return features


def make_vla_jepa_lfm_pre_post_processors(
    config: VLAJEPALFMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
):
    preprocessor, postprocessor = make_vla_jepa_pre_post_processors(config, dataset_stats)
    # Upstream order: rename -> batch -> device -> image prep -> relative -> normalize.
    preprocessor.steps.insert(3, DropImagePadMasksProcessorStep())
    return preprocessor, postprocessor
