from __future__ import annotations

import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vla_jepa.action_head import VLAJEPAActionHead
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAModel, VLAJEPAPolicy
from lerobot.policies.vla_jepa.world_model import ActionConditionedVideoPredictor
from safetensors import safe_open
from torch import nn
from transformers import AutoModel, AutoVideoProcessor

from .configuration_vla_jepa_lfm import VLAJEPALFMConfig
from .lfm_interface import LFM25VLInterface

logger = logging.getLogger(__name__)


class VLAJEPALFMModel(VLAJEPAModel):
    """Reuse LeRobot's VLA-JEPA losses/action plumbing with an LFM vision-language interface."""

    def __init__(self, config: VLAJEPALFMConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        # Keep the attribute name `qwen` because upstream VLAJEPAModel helper methods use it.
        self.qwen = LFM25VLInterface(config)

        self.action_tokens, self.action_token_ids, self.embodied_action_token_id = self.qwen.expand_tokenizer()
        self.register_buffer(
            "_action_token_ids_t",
            torch.tensor(self.action_token_ids, dtype=torch.long),
            persistent=False,
        )
        self.action_model = VLAJEPAActionHead(config, cross_attention_dim=config.conditioning_dim)

        if config.enable_world_model:
            dtype = self.qwen._get_torch_dtype(config.torch_dtype)
            self.video_encoder = AutoModel.from_pretrained(config.jepa_encoder_name, dtype=dtype)
            self.video_processor = AutoVideoProcessor.from_pretrained(config.jepa_encoder_name)
            num_views = config.num_world_model_views
            tubelet_size = self.video_encoder.config.tubelet_size
            image_size = getattr(self.video_encoder.config, "image_size", None)
            if image_size is None:
                image_size = next(iter(config.image_features.values())).shape[-1]
            self.video_predictor = ActionConditionedVideoPredictor(
                num_frames=config.num_video_frames // tubelet_size,
                img_size=(image_size, image_size),
                patch_size=16,
                tubelet_size=1,
                embed_dim=self.video_encoder.config.hidden_size * num_views,
                action_embed_dim=config.conditioning_dim,
                predictor_embed_dim=self.video_encoder.config.hidden_size,
                depth=config.predictor_depth,
                num_heads=config.predictor_num_heads,
                mlp_ratio=config.predictor_mlp_ratio,
                num_action_tokens_per_step=config.num_action_tokens_per_timestep,
                dropout=config.predictor_dropout,
            )
        else:
            self.video_encoder = None
            self.video_processor = None
            self.video_predictor = None

        tubelet = self.video_encoder.config.tubelet_size if self.video_encoder is not None else config.jepa_tubelet_size
        prompt_steps = config.num_video_frames // tubelet - 1
        self.replace_prompt = "".join(
            token * config.num_action_tokens_per_timestep for token in self.action_tokens[:prompt_steps]
        )
        self.embodied_replace_prompt = config.embodied_action_token * config.num_embodied_action_tokens_per_instruction

    def _qwen_last_decoder_hidden(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        captured: list[torch.Tensor] = []

        def hook(_module, _inputs, output):
            captured.append(output[0] if isinstance(output, tuple) else output)

        last_layer = self.qwen.model.model.language_model.layers[-1]
        handle = last_layer.register_forward_hook(hook)
        try:
            self.qwen.model.model(**inputs)
        finally:
            handle.remove()
        return self.qwen.project_hidden(captured[0])


class VLAJEPALFMPolicy(VLAJEPAPolicy):
    config_class = VLAJEPALFMConfig
    name = "vla_jepa_lfm"

    def __init__(self, config: VLAJEPALFMConfig, **kwargs) -> None:
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.model = VLAJEPALFMModel(config)
        if config.init_from_vla_jepa:
            self._load_compatible_vla_jepa_weights(config.init_from_vla_jepa, config.init_prefixes)
        self.reset()

    def get_optim_params(self):
        return [param for param in self.model.parameters() if param.requires_grad]

    def _load_compatible_vla_jepa_weights(self, source: str, prefixes: tuple[str, ...]) -> None:
        path = Path(source)
        is_gguf = False
        if path.is_dir():
            if (path / "model.safetensors").is_file():
                model_file = path / "model.safetensors"
            else:
                ggufs = list(path.glob("*.gguf"))
                if ggufs:
                    model_file = ggufs[0]
                    is_gguf = True
                else:
                    model_file = path / "model.safetensors"
        elif path.is_file():
            model_file = path
            is_gguf = model_file.suffix == ".gguf"
        else:
            try:
                model_file = Path(hf_hub_download(repo_id=source, filename="model.safetensors"))
            except Exception:
                from huggingface_hub import HfApi
                api = HfApi()
                repo_files = api.list_repo_files(repo_id=source)
                ggufs = [f for f in repo_files if f.endswith(".gguf")]
                if ggufs:
                    gguf_name = "vla-jepa.gguf" if "vla-jepa.gguf" in ggufs else ggufs[0]
                    model_file = Path(hf_hub_download(repo_id=source, filename=gguf_name))
                    is_gguf = True
                else:
                    raise

        current = self.state_dict()
        selected: dict[str, torch.Tensor] = {}
        mismatched: list[str] = []

        if is_gguf:
            import gguf
            reader = gguf.GGUFReader(str(model_file))
            for tensor in reader.tensors:
                key = tensor.name
                candidate_keys = [key]
                if not key.startswith("model."):
                    candidate_keys.append("model." + key)
                for k in candidate_keys:
                    if not any(k.startswith(p) for p in prefixes):
                        continue
                    if k not in current:
                        continue
                    t = torch.from_numpy(tensor.data.copy())
                    if t.shape != current[k].shape:
                        mismatched.append(f"{k}: {tuple(t.shape)} != {tuple(current[k].shape)}")
                        continue
                    selected[k] = t
        else:
            with safe_open(model_file, framework="pt", device="cpu") as handle:
                for key in handle.keys():  # noqa: SIM118 - safetensors.safe_open is not iterable
                    if not any(key.startswith(p) for p in prefixes):
                        continue
                    if key not in current:
                        continue
                    tensor = handle.get_tensor(key)
                    if tensor.shape != current[key].shape:
                        mismatched.append(f"{key}: {tuple(tensor.shape)} != {tuple(current[key].shape)}")
                        continue
                    selected[key] = tensor

        if mismatched:
            raise ValueError("Incompatible VLA-JEPA initialization tensors:\n" + "\n".join(mismatched))
        if not selected:
            logger.warning(
                "No compatible tensors found in %s for prefixes %s; keeping default initialization",
                model_file,
                prefixes,
            )
            return
        _missing, unexpected = self.load_state_dict(selected, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected initialization keys: {unexpected}")
        logger.info("Loaded %d compatible tensors from %s", len(selected), source)
