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


def eligible_nodes(nodes):
    return [n for n in nodes[1:] if n.get("eligible", n.get("status") == "ok")]


def mechanism_families(nodes):
    return {
        n.get("family_id")
        or n.get("mechanism_family")
        or (n.get("proposal") or {}).get("mechanism_family")
        for n in eligible_nodes(nodes)
    } - {None, ""}


def observation(nodes, limit, protocol=None):
    protocol = protocol or {}
    parents = {n["parent"] for n in nodes[1:]}
    minima = {
        k: protocol.get(k, 0)
        for k in (
            "min_nodes_before_stop",
            "min_root_branches_before_stop",
            "min_distinct_families_before_stop",
        )
    }
    eligible = eligible_nodes(nodes)
    roots = sum(n["parent"] == "root" for n in eligible)
    families = len(mechanism_families(nodes))
    measured = len(eligible)
    return {
        "nodes": json.loads(json.dumps(nodes)),
        "leaves": [
            n["id"]
            for n in nodes[1:]
            if n["id"] not in parents and n.get("eligible", n.get("status") == "ok")
        ],
        "round": len(nodes) - 1,
        "limit": limit,
        "root_branches": roots,
        "distinct_families": families,
        "measured_nodes": measured,
        "protocol_minima": minima,
        "root_open_target": protocol.get("root_open_target", 3),
        "stop_allowed": (
            measured >= minima["min_nodes_before_stop"]
            and roots >= minima["min_root_branches_before_stop"]
            and families >= minima["min_distinct_families_before_stop"]
        ),
    }


def constrain_action(action, obs):
    validate_action(action, obs)
    if action is not None or obs["stop_allowed"]:
        return action, False
    minima = obs["protocol_minima"]
    if (
        obs["root_branches"] < minima["min_root_branches_before_stop"]
        or obs["distinct_families"] < minima["min_distinct_families_before_stop"]
    ):
        return "root", True
    leaves = [n for n in obs["nodes"] if n["id"] in obs["leaves"]]
    best = max(leaves, key=lambda n: (n["score"], n["id"]), default=None)
    return best["id"] if best else "root", True


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


def plan_batch_actions(select, frozen_observation, seed, batch_size, protocol):
    """Plan against one immutable prefix; expose choices, never sibling outcomes."""
    actions, substitutions = [], []
    for slot in range(batch_size):
        obs = json.loads(json.dumps(frozen_observation))
        obs.update(batch_slot=slot, batch_size=batch_size, planned_actions=list(actions))
        requested = select(obs, seed + slot)
        # Constrain against the trusted prefix, not policy-mutated input.
        action, constrained = constrain_action(requested, frozen_observation)
        if action is None:
            break
        duplicate = action != "root" and action in actions
        if duplicate:
            leaves = [
                n
                for n in frozen_observation["nodes"]
                if n["id"] in frozen_observation["leaves"] and n["id"] not in actions
            ]
            best = max(leaves, key=lambda n: (n["score"], n["id"]), default=None)
            action = best["id"] if best else "root"
        if constrained or duplicate:
            substitutions.append(
                {
                    "batch_slot": slot,
                    "requested": requested,
                    "action": action,
                    "reason": "duplicate_parent" if duplicate else "coverage",
                }
            )
        actions.append(action)
    return {
        "planned_actions": actions,
        "substitutions": substitutions,
        "constrained_actions": len(substitutions),
    }


def replay(nodes, select, seed, limit, beta1, failure_score, beta_diversity=0.0, protocol=None):
    validate_tree(nodes)
    revealed = [nodes[0]]
    trace = []
    reason = "K2"
    constrained_actions = 0
    protocol = protocol or {}
    while len(revealed) - 1 < limit:
        if len(revealed) == len(nodes):
            reason = "full_tree"
            break
        obs = observation(revealed, limit, protocol)
        plan = plan_batch_actions(
            select,
            obs,
            seed + len(trace) * max(1, limit),
            min(protocol.get("proposal_batch_size", 1), limit - len(revealed) + 1),
            protocol,
        )
        constrained_actions += plan["constrained_actions"]
        seen = {n["id"] for n in revealed}
        children = []
        for action in plan["planned_actions"]:
            child = next(
                (n for n in nodes[1:] if n["parent"] == action and n["id"] not in seen), None
            )
            if child is not None:
                children.append(child)
                seen.add(child["id"])
        trace.append(
            dict(
                visible_before=[n["id"] for n in revealed],
                **plan,
                constrained=bool(plan["constrained_actions"]),
                revealed=[n["id"] for n in children],
            )
        )
        if not plan["planned_actions"]:
            reason = "STOP"
            break
        if not children:
            reason = "empty_action"
            break
        revealed.extend(children)
    quality = max(
        (n["score"] for n in revealed if n.get("score") is not None), default=failure_score
    )
    return {
        "objective": quality
        - beta1 * (len(revealed) - 1)
        + beta_diversity * len(mechanism_families(revealed)),
        "unique_families": len(mechanism_families(revealed)),
        "constrained_actions": constrained_actions,
        "revealed": len(revealed) - 1,
        "trace": trace,
        "reason": reason,
    }
