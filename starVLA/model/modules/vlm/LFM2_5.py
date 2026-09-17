import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration


class _ResidualRMSMLPAdapter(nn.Module):
    """Residual 1024->2048 adapter initialized to the successful linear bridge."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.RMSNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim, bias=False)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.fc1.weight)
            self.fc2.weight.zero_()
            self.skip.weight.zero_()
            if out_dim == 2 * in_dim:
                eye = torch.eye(in_dim)
                self.skip.weight[:in_dim].copy_(eye / math.sqrt(2.0))
                self.skip.weight[in_dim:].copy_(eye / math.sqrt(2.0))
            else:
                nn.init.xavier_uniform_(self.skip.weight)

    def forward(self, x):
        return self.skip(x) + self.fc2(self.act(self.fc1(self.norm(x))))


class _LFM2_5_VL_Interface(nn.Module):
    """LFM2.5-VL wrapper with a trainable projection into VLA-JEPA's pretrained 2048-D interface."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        vl_cfg = config.framework.get("qwenvl", {})
        model_id = vl_cfg.get("base_vlm", "LiquidAI/LFM2.5-VL-450M")
        self.model = Lfm2VlForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"
        self.config = config

        self.native_hidden_size = int(self.model.config.text_config.hidden_size)
        self.output_hidden_size = int(vl_cfg.get("vl_hidden_dim", 2048))
        adapter_type = str(vl_cfg.get("adapter_type", "linear")).lower()
        if adapter_type == "residual_mlp":
            self.hidden_adapter = _ResidualRMSMLPAdapter(self.native_hidden_size, self.output_hidden_size)
        else:
            self.hidden_adapter = nn.Linear(self.native_hidden_size, self.output_hidden_size, bias=False)
            with torch.no_grad():
                self.hidden_adapter.weight.zero_()
                if self.output_hidden_size == 2 * self.native_hidden_size:
                    eye = torch.eye(self.native_hidden_size)
                    self.hidden_adapter.weight[: self.native_hidden_size].copy_(eye / math.sqrt(2.0))
                    self.hidden_adapter.weight[self.native_hidden_size :].copy_(eye / math.sqrt(2.0))
                else:
                    nn.init.xavier_uniform_(self.hidden_adapter.weight)
        self.hidden_adapter.to(device=self.model.device, dtype=torch.bfloat16)

        # VLA_JEPA currently reads this compatibility alias when constructing its pretrained heads.
        self.model.config.hidden_size = self.output_hidden_size

    def forward(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**kwargs)
            if getattr(outputs, "hidden_states", None) is not None:
                hs = list(outputs.hidden_states)
                hs[-1] = self.hidden_adapter(hs[-1])
                outputs.hidden_states = tuple(hs)
        return outputs

    def generate(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.generate(**kwargs)

    def build_qwenvl_inputs(
        self,
        images,
        instructions,
        solutions=None,
        prompt_replace_dict=None,
        prompt_template=None,
        **kwargs,
    ):
        messages = []
        assert len(images) == len(instructions)
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]
            if prompt_template is None:
                if "CoT_prompt" in self.config.datasets.vla_data:
                    prompt = self.config.datasets.vla_data.get("CoT_prompt", "").replace("{instruction}", instruction)
                else:
                    prompt = instruction
            else:
                prompt = prompt_template.replace("{instruction}", instruction)
            if prompt_replace_dict is not None:
                for key, value in prompt_replace_dict.items():
                    prompt = prompt.replace(key, value)
            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]
            if solutions is not None:
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solutions[len(messages)]}]})
            messages.append(msg)

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return batch_inputs.to(self.model.device)
