"""LFM2.5-VL variant of LeRobot's VLA-JEPA policy."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError("Install LeRobot before using this plugin.") from exc

from .configuration_vla_jepa_lfm import VLAJEPALFMConfig
from .modeling_vla_jepa_lfm import VLAJEPALFMPolicy
from .processor_vla_jepa_lfm import make_vla_jepa_lfm_pre_post_processors

__all__ = [
    "VLAJEPALFMConfig",
    "VLAJEPALFMPolicy",
    "make_vla_jepa_lfm_pre_post_processors",
]
