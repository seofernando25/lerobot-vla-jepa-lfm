from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration

from .configuration_vla_jepa_lfm import VLAJEPALFMConfig


class _LinearAdapter(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(in_dim, out_dim, bias=False)
        with torch.no_grad():
            self.weight.zero_()
            if out_dim == 2 * in_dim:
                eye = torch.eye(in_dim)
                self.weight[:in_dim].copy_(eye / math.sqrt(2.0))
                self.weight[in_dim:].copy_(eye / math.sqrt(2.0))
            else:
                nn.init.xavier_uniform_(self.weight)


class ResidualRMSMLPAdapter(nn.Module):
    """1024->2048 adapter whose step-0 output matches the linear bridge."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim, bias=False)
        self.skip = _LinearAdapter(in_dim, out_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.fc2(self.act(self.fc1(self.norm(x))))


class LFM25VLInterface(nn.Module):
    def __init__(self, config: VLAJEPALFMConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Lfm2VlForConditionalGeneration.from_pretrained(
            config.lfm_model_name,
            dtype=self._get_torch_dtype(config.torch_dtype),
        )
        self.processor = AutoProcessor.from_pretrained(config.lfm_model_name)
        self.processor.tokenizer.padding_side = config.tokenizer_padding_side

        self.native_hidden_size = int(self.model.config.text_config.hidden_size)
        adapter_cls = ResidualRMSMLPAdapter if config.adapter_type == "residual_mlp" else _LinearAdapter
        self.hidden_adapter = adapter_cls(self.native_hidden_size, config.conditioning_dim)
        self.hidden_adapter.to(device=self.model.device, dtype=self._get_torch_dtype(config.torch_dtype))
        self._configure_trainability()

    @staticmethod
    def _get_torch_dtype(dtype_name: str) -> torch.dtype:
        if dtype_name == "float32":
            return torch.float32
        if dtype_name == "float16":
            return torch.float16
        return torch.bfloat16

    def _configure_trainability(self) -> None:
        self.model.requires_grad_(False)
        if not self.config.freeze_vision_tower:
            self.model.model.vision_tower.requires_grad_(True)
        if self.config.train_multimodal_projector:
            self.model.model.multi_modal_projector.requires_grad_(True)
        if self.config.unfreeze_last_n:
            layers = self.model.model.language_model.layers[-self.config.unfreeze_last_n :]
            for layer in layers:
                layer.requires_grad_(True)
            self.model.model.language_model.embedding_norm.requires_grad_(True)
        self.hidden_adapter.requires_grad_(True)

    def project_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.hidden_adapter(hidden)

    def expand_tokenizer(self) -> tuple[list[str], list[int], int]:
        max_action_tokens = self.config.chunk_size * 4
        tokenizer = self.processor.tokenizer
        action_tokens: list[str] = []
        action_token_ids: list[int] = []
        for idx in range(max_action_tokens):
            token = self.config.special_action_token.format(idx)
            action_tokens.append(token)
            if token not in tokenizer.get_vocab():
                tokenizer.add_tokens([token], special_tokens=True)
            action_token_ids.append(tokenizer.convert_tokens_to_ids(token))

        embodied = self.config.embodied_action_token
        if embodied not in tokenizer.get_vocab():
            tokenizer.add_tokens([embodied], special_tokens=True)
        embodied_id = tokenizer.convert_tokens_to_ids(embodied)

        rows = self.model.get_input_embeddings().weight.size(0)
        if rows < len(tokenizer):
            self.model.resize_token_embeddings(len(tokenizer))
        return action_tokens, action_token_ids, embodied_id

    def build_inputs(
        self,
        images: Sequence[Sequence[torch.Tensor]],
        instructions: Sequence[str],
        action_prompt: str,
        embodied_prompt: str,
    ) -> dict[str, torch.Tensor]:
        messages = []
        for sample_images, instruction in zip(images, instructions, strict=True):
            prompt = self.config.prompt_template.format(
                instruction=instruction,
                actions=action_prompt,
                e_actions=embodied_prompt,
            )
            content = [{"type": "image", "image": image} for image in sample_images]
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "padding": True,
                "return_tensors": "pt",
                "device": self.model.device,
                "do_rescale": False,
            },
        )
        return inputs.to(self.model.device)

    @staticmethod
    def to_pixel_values(image_tensor: torch.Tensor) -> torch.Tensor:
        image = image_tensor.detach().float()
        if image.shape[-3] == 1:
            repeats = [1] * image.ndim
            repeats[-3] = 3
            image = image.repeat(*repeats)
        return image
