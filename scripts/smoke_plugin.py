#!/usr/bin/env python3
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class

import lerobot_policy_vla_jepa_lfm  # noqa: F401


def main() -> None:
    assert "vla_jepa_lfm" in PreTrainedConfig.get_known_choices()
    cls = get_policy_class("vla_jepa_lfm")
    print(f"registered: vla_jepa_lfm -> {cls.__module__}.{cls.__name__}")


if __name__ == "__main__":
    main()
