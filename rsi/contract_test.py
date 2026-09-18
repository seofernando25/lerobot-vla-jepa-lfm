"""Minimal candidate contract; no weights or GPU required."""

import ast
from pathlib import Path


def test_inherited_loss_and_action_methods():
    protected = {
        "VLAJEPALFMModel": {"_action_loss", "_world_model_loss", "forward", "predict_action"},
        "VLAJEPALFMPolicy": {"forward", "predict_action_chunk", "select_action"},
    }
    for path in Path("src/lerobot_policy_vla_jepa_lfm").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef) and node.name in protected:
                assert not (
                    {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
                    & protected[node.name]
                )
