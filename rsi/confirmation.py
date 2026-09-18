"""Emit a concrete matched retraining/evaluation plan; never launch training."""

import json
import sys

from rsi.core import digest, manifest
from rsi.evaluator import train_args


def plan(runner):
    runner.verify()
    template = json.loads((runner.repo / "rsi/confirmation.json").read_text())
    candidates = [e for e in runner.events("outcome") if e["node"]["status"] == "ok"]
    if not candidates or runner.synthetic:
        raise ValueError("confirmation requires measured real candidates")
    best = max(candidates, key=lambda e: e["node"]["score"])
    arms = {
        "base_lfm": runner.state / "base",
        "best_rsi": runner.state / "attempts" / best["attempt"] / "workspace",
    }
    if manifest(arms["best_rsi"]) != best["workspace_hash"]:
        raise ValueError("selected architecture source changed")
    config = dict(runner.config, seed=template["seed"])
    evaluation = [
        f"--env.task={template['suite']}",
        "--env.type=libero",
        f"--eval.n_episodes={template['n_episodes']}",
        f"--eval.batch_size={template['eval_batch_size']}",
        f"--seed={template['seed']}",
    ]
    result = {
        "status": "plan_only",
        "selected_node": best["attempt"],
        "template": template,
        "config_hash": digest(config),
        "arms": {},
    }
    for arm, workspace in arms.items():
        output = runner.state / "confirmation" / arm
        checkpoint = output / "train/checkpoints/last/pretrained_model"
        result["arms"][arm] = {
            "cwd": str(workspace),
            "env": {"PYTHONPATH": str(workspace / "src")},
            "source_hash": digest(manifest(workspace)),
            "train_argv": [
                sys.executable,
                str(runner.repo / "rsi/evaluator.py"),
                *train_args(
                    config, str(output / "train"), template["training_steps"], save_checkpoint=True
                ),
            ],
            "eval_argv": [
                sys.executable,
                "-m",
                "lerobot.scripts.lerobot_eval",
                f"--policy.path={checkpoint}",
                *evaluation,
                f"--output_dir={output / 'libero'}",
            ],
        }
    result["qwen_reference_eval_argv"] = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_eval",
        f"--policy.path={template['qwen_checkpoint']}",
        *evaluation,
        f"--output_dir={runner.state / 'confirmation/qwen_reference'}",
    ]
    result["instructions"] = (
        "Run each arm with its recorded PYTHONPATH and cwd in a separate process; "
        "fresh training, no probe checkpoints. Same steps/seed/data/precision/eval protocol. "
        "Qwen is evaluation-only. Record results separately; probe loss is not LIBERO success."
    )
    return result
