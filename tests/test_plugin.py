import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class

import lerobot_policy_vla_jepa_lfm  # noqa: F401
from lerobot_policy_vla_jepa_lfm.lfm_interface import ResidualRMSMLPAdapter, _LinearAdapter


def test_plugin_registration():
    assert "vla_jepa_lfm" in PreTrainedConfig.get_known_choices()
    assert get_policy_class("vla_jepa_lfm").name == "vla_jepa_lfm"


def test_residual_adapter_matches_linear_at_initialization():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    linear = _LinearAdapter(4, 8)
    residual = ResidualRMSMLPAdapter(4, 8)
    assert torch.allclose(linear(x), residual(x), atol=0, rtol=0)
