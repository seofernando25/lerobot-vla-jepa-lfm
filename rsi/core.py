"""Pure state, tree, guard and replay contracts (stdlib only)."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

PLUGIN = Path("src/lerobot_policy_vla_jepa_lfm")
FROZEN_PLUGIN_FILES = {str(PLUGIN / "__init__.py"), str(PLUGIN / "processor_vla_jepa_lfm.py")}

PROTECTED = {
    "VLAJEPALFMModel": {"_action_loss", "_world_model_loss", "forward", "predict_action"},
    "VLAJEPALFMPolicy": {"forward", "predict_action_chunk", "select_action"},
}


def digest(data):
    if not isinstance(data, bytes):
        data = json.dumps(data, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Journal:
    """Immutable numbered JSON events; JSONL view is derived, never authoritative.

    Atomic rename avoids torn JSONL tails. Caller holds the single-writer flock.
    """

    def __init__(self, root):
        self.root = Path(root) / "events"
        self.root.mkdir(parents=True, exist_ok=True)

    def read(self):
        paths = sorted(self.root.glob("*.json"))
        result = []
        for i, path in enumerate(paths):
            event = json.loads(path.read_text())
            if path.name != f"{i:08d}.json" or event["seq"] != i:
                raise ValueError("journal sequence gap/corruption")
            if event["previous"] != (digest(result[-1]) if result else None):
                raise ValueError("journal hash chain broken")
            result.append(event)
        return result

    def append(self, kind, **fields):
        events = self.read()
        record = dict(
            fields, kind=kind, seq=len(events), previous=digest(events[-1]) if events else None
        )
        atomic(self.root / f"{len(events):08d}.json", record)
        return record


def manifest(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink forbidden: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = digest(path.read_bytes())
    return result


def candidate_guard(workspace, before):
    after = manifest(workspace)
    for name in FROZEN_PLUGIN_FILES:
        if before.get(name) != after.get(name):
            raise ValueError(f"candidate changed frozen protocol file: {name}")
    for name in before.keys() | after.keys():
        if not name.startswith(str(PLUGIN) + "/") and before.get(name) != after.get(name):
            raise ValueError(f"candidate changed forbidden file: {name}")
    for name in after:
        if name.startswith(str(PLUGIN) + "/"):
            if not name.endswith(".py"):
                raise ValueError("plugin candidates may contain Python source only")
            tree = ast.parse((Path(workspace) / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name in PROTECTED:
                    for item in node.body:
                        names = []
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            names = [item.name]
                        elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                            targets = (
                                item.targets if isinstance(item, ast.Assign) else [item.target]
                            )
                            names = [t.id for t in targets if isinstance(t, ast.Name)]
                        if set(names) & PROTECTED[node.name]:
                            raise ValueError(f"protected override in {node.name}: {names}")
    return after


def observation(nodes, limit):
    parents = {n["parent"] for n in nodes[1:]}
    return {
        "nodes": json.loads(json.dumps(nodes)),
        "leaves": [n["id"] for n in nodes[1:] if n["id"] not in parents],
        "round": len(nodes) - 1,
        "limit": limit,
    }


def validate_action(action, obs):
    if action is not None and action != "root" and action not in obs["leaves"]:
        raise ValueError(f"invalid parent selection: {action!r}")


def validate_tree(nodes):
    if not nodes or nodes[0]["id"] != "root" or nodes[0]["parent"] is not None:
        raise ValueError("tree must start with root")
    seen, parents = {"root"}, set()
    for node in nodes[1:]:
        if node["id"] in seen or node["parent"] not in seen:
            raise ValueError("duplicate or noncausal node")
        if node["parent"] != "root" and node["parent"] in parents:
            raise ValueError("non-root branch must be a chain")
        if not math.isfinite(node["score"]):
            raise ValueError("nonfinite quality")
        seen.add(node["id"])
        parents.add(node["parent"])


def replay(nodes, select, seed, limit, beta1, failure_score):
    validate_tree(nodes)
    revealed = [nodes[0]]
    trace = []
    reason = "K2"
    for turn in range(limit):
        if len(revealed) == len(nodes):
            reason = "full_tree"
            break
        obs = observation(revealed, limit)
        action = select(obs, seed + turn)
        validate_action(action, obs)
        if action is None:
            reason = "STOP"
            break
        seen = {n["id"] for n in revealed}
        child = next((n for n in nodes[1:] if n["parent"] == action and n["id"] not in seen), None)
        if child is None:
            reason = "empty_action"
            break
        trace.append(
            {
                "visible_before": [n["id"] for n in revealed],
                "action": action,
                "revealed": child["id"],
            }
        )
        revealed.append(child)
    quality = max(
        (n["score"] for n in revealed if n.get("score") is not None), default=failure_score
    )
    return {
        "objective": quality - beta1 * (len(revealed) - 1),
        "revealed": len(revealed) - 1,
        "trace": trace,
        "reason": reason,
    }
