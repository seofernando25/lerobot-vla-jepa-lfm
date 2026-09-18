"""Fixed evaluator command construction and strict metric parsing."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path


def train_args(config, output="/tmp/output/train", steps=None):
    steps = config["probe_steps"] if steps is None else steps
    return [
        "--policy.type=vla_jepa_lfm",
        "--policy.init_from_vla_jepa=lerobot/VLA-JEPA-Pretrain",
        "--policy.push_to_hub=false",
        "--policy.device=cuda",
        "--policy.torch_dtype=bfloat16",
        "--policy.lfm_model_name=LiquidAI/LFM2.5-VL-450M",
        "--policy.conditioning_dim=2048",
        "--policy.enable_world_model=true",
        "--policy.jepa_encoder_name=facebook/vjepa2-vitl-fpc64-256",
        "--policy.chunk_size=7",
        "--policy.n_action_steps=7",
        "--policy.num_action_tokens_per_timestep=8",
        "--policy.num_embodied_action_tokens_per_instruction=32",
        "--policy.num_inference_timesteps=4",
        "--policy.action_hidden_size=1024",
        "--policy.action_model_type=DiT-B",
        "--policy.action_num_layers=16",
        "--policy.action_dropout=0.2",
        "--policy.num_video_frames=8",
        "--policy.predictor_depth=12",
        "--policy.predictor_num_heads=8",
        "--policy.predictor_mlp_ratio=4.0",
        "--policy.predictor_dropout=0.0",
        "--policy.world_model_loss_weight=0.1",
        "--policy.jepa_tubelet_size=2",
        "--policy.repeated_diffusion_steps=8",
        "--policy.causal_world_model_context=false",
        "--policy.use_relative_actions=false",
        "--policy.binarize_gripper_action=false",
        "--policy.pre_snap_gripper_action=false",
        "--policy.clip_normalized_actions=true",
        f"--dataset.repo_id={config['dataset']['repo_id']}",
        f"--dataset.root={config['dataset']['sandbox_root']}",
        "--dataset.video_backend=pyav",
        "--dataset.use_imagenet_stats=true",
        f"--dataset.eval_split={config['dataset']['eval_split']}",
        "--dataset.streaming=false",
        f"--steps={steps}",
        f"--eval_steps={steps}",
        "--env_eval_freq=0",
        f"--max_eval_samples={config['max_eval_samples']}",
        f"--seed={config['seed']}",
        f"--batch_size={config['batch_size']}",
        f"--num_workers={config['num_workers']}",
        "--cudnn_deterministic=true",
        f"--accelerator.mixed_precision={config['precision']}",
        "--accelerator.gradient_accumulation.steps=1",
        "--use_policy_training_preset=false",
        "--optimizer.type=adamw",
        "--optimizer.lr=0.0001",
        "--optimizer.weight_decay=0.0000000001",
        "--optimizer.betas=[0.9,0.95]",
        "--optimizer.eps=0.00000001",
        "--scheduler.type=cosine_decay_with_warmup",
        "--scheduler.num_warmup_steps=1000",
        "--scheduler.num_decay_steps=30000",
        "--scheduler.peak_lr=0.0001",
        "--scheduler.decay_lr=0.0000025",
        "--optimizer.grad_clip_norm=10.0",
        "--wandb.enable=false",
        "--save_checkpoint=true",
        "--save_freq=0",
        f"--output_dir={output}",
        "--job_name=rsi_fixed_probe",
    ]


def parse_metric(text, steps):
    matches = re.findall(r"step (\d+): eval_loss=([-+0-9.eE]+)", text)
    values = [float(value) for step, value in matches if int(step) == steps]
    if len(values) != 1 or not math.isfinite(values[0]):
        raise ValueError("expected exactly one finite final held-out eval_loss")
    return {
        "eval_loss": values[0],
        "score": -values[0],
        "optimizer_steps": steps,
        "metric_provenance": "pinned LeRobot held-out eval_loss log (4 decimal places)",
    }


def runtime_guard():
    # Capture upstream method objects BEFORE importing any candidate module.
    from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAModel, VLAJEPAPolicy

    protected = {
        VLAJEPAModel: ("_action_loss", "_world_model_loss", "forward", "predict_action"),
        VLAJEPAPolicy: ("forward", "predict_action_chunk", "select_action"),
    }
    original = {
        (base, name): getattr(base, name) for base, names in protected.items() for name in names
    }
    from lerobot_policy_vla_jepa_lfm.modeling_vla_jepa_lfm import (
        VLAJEPALFMModel,
        VLAJEPALFMPolicy,
    )

    def check():
        for base, cls in ((VLAJEPAModel, VLAJEPALFMModel), (VLAJEPAPolicy, VLAJEPALFMPolicy)):
            for name in protected[base]:
                if getattr(cls, name) is not original[base, name]:
                    raise ValueError(f"protected runtime override: {cls.__name__}.{name}")
                if getattr(base, name) is not original[base, name]:
                    raise ValueError("candidate monkeypatched upstream")

    check()
    return check


def main():
    """Trusted external entry point. No candidate-selected CLI arguments."""
    from importlib.metadata import distribution

    direct = json.loads(distribution("lerobot").read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != "30074f7f1358b3c015ae1750017200e86e9c4eb6":
        raise RuntimeError("installed LeRobot revision does not match fixed protocol")
    output_arg = next(
        arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--output_dir=")
    )
    metadata_dir = Path(output_arg).parent
    metadata_dir.mkdir(parents=True, exist_ok=True)
    import torch
    from lerobot.scripts import lerobot_train

    # Snapshot trusted framework callables before candidate modules are imported.
    original_factory = lerobot_train.make_policy
    original_loaders = lerobot_train.make_dataloaders
    check = runtime_guard()

    def make_policy(*args, **kwargs):
        cfg = args[0] if args else kwargs["cfg"]
        fixed = {
            "type": "vla_jepa_lfm",
            "lfm_model_name": "LiquidAI/LFM2.5-VL-450M",
            "conditioning_dim": 2048,
            "enable_world_model": True,
            "jepa_encoder_name": "facebook/vjepa2-vitl-fpc64-256",
            "chunk_size": 7,
            "n_action_steps": 7,
            "num_action_tokens_per_timestep": 8,
            "num_embodied_action_tokens_per_instruction": 32,
            "num_inference_timesteps": 4,
            "action_hidden_size": 1024,
            "action_model_type": "DiT-B",
            "action_num_layers": 16,
            "action_dropout": 0.2,
            "num_video_frames": 8,
            "predictor_depth": 12,
            "predictor_num_heads": 8,
            "predictor_mlp_ratio": 4.0,
            "predictor_dropout": 0.0,
            "world_model_loss_weight": 0.1,
            "jepa_tubelet_size": 2,
            "repeated_diffusion_steps": 8,
            "causal_world_model_context": False,
            "use_relative_actions": False,
            "binarize_gripper_action": False,
            "pre_snap_gripper_action": False,
            "clip_normalized_actions": True,
            "resize_images_to": None,
            "init_from_vla_jepa": "lerobot/VLA-JEPA-Pretrain",
        }
        for name, expected in fixed.items():
            if getattr(cfg, name, object()) != expected:
                raise ValueError(f"candidate changed fixed protocol field {name}")
        normalization = {
            str(k): getattr(v, "value", str(v)) for k, v in cfg.normalization_mapping.items()
        }
        if normalization != {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MIN_MAX"}:
            raise ValueError("candidate changed fixed normalization protocol")
        policy = original_factory(*args, **kwargs)
        check()  # Also reject constructor-installed overrides.
        from lerobot.policies.vla_jepa.action_head import VLAJEPAActionHead
        from lerobot.policies.vla_jepa.world_model import ActionConditionedVideoPredictor

        if type(policy.model.action_model) is not VLAJEPAActionHead:
            raise ValueError("candidate changed upstream action-head architecture")
        if type(policy.model.video_predictor) is not ActionConditionedVideoPredictor:
            raise ValueError("candidate changed upstream world-model predictor architecture")
        if not all(p.requires_grad for p in policy.model.action_model.parameters()):
            raise ValueError("candidate changed action-head trainability")
        if not all(p.requires_grad for p in policy.model.video_predictor.parameters()):
            raise ValueError("candidate changed world-predictor trainability")
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        checkpoint = hf_hub_download(
            repo_id="lerobot/VLA-JEPA-Pretrain", filename="model.safetensors"
        )
        state = policy.state_dict()
        prefixes = ("model.action_model.", "model.video_predictor.")
        current_keys = {k for k in state if k.startswith(prefixes)}
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            expected_keys = {k for k in list(handle.keys()) if k.startswith(prefixes)}
            unexpected_current = current_keys - expected_keys
            if unexpected_current:
                raise ValueError(
                    f"candidate added non-checkpoint action/world tensors: {sorted(unexpected_current)[:20]}"
                )
            for key in sorted(current_keys):
                if not torch.equal(state[key].detach().cpu(), handle.get_tensor(key)):
                    raise ValueError(f"candidate changed fixed pretrained initialization: {key}")
        for name in ("forward", "predict_action_chunk", "select_action"):
            if name in policy.__dict__:
                raise ValueError("instance override of policy semantics")
        for name in ("forward", "_action_loss", "_world_model_loss", "predict_action"):
            if name in policy.model.__dict__:
                raise ValueError("instance override of model semantics")
        groups = [
            {"name": n, "parameters": p.numel()}
            for n, p in policy.named_parameters()
            if p.requires_grad
        ]
        (metadata_dir / "trainability.json").write_text(json.dumps(groups))
        return policy

    lerobot_train.make_policy = make_policy

    def loaders(cfg, *args, **kwargs):
        train, heldout = original_loaders(cfg, *args, **kwargs)
        if heldout is None or not len(heldout.dataset):
            raise ValueError("empty held-out dataset")
        # Upstream per-task minimum can exceed max_eval_samples when tasks > cap.
        # Enforce a strict, deterministic global cap in this external evaluator.
        if len(heldout.dataset) > cfg.max_eval_samples:
            indices = [
                i * len(heldout.dataset) // cfg.max_eval_samples
                for i in range(cfg.max_eval_samples)
            ]
            heldout = torch.utils.data.DataLoader(
                torch.utils.data.Subset(heldout.dataset, indices),
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                collate_fn=heldout.collate_fn,
            )
        (metadata_dir / "eval_samples.json").write_text(json.dumps({"count": len(heldout.dataset)}))
        return train, heldout

    lerobot_train.make_dataloaders = loaders
    lerobot_train.main()
    check()
    (metadata_dir / "hardware.json").write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "python": sys.version,
            }
        )
    )


if __name__ == "__main__":
    main()
