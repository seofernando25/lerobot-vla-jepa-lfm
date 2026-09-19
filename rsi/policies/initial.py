"""V4: exploit strong structural leaves while preserving mixed-batch diversity."""

import random


def evidence_score(node):
    score = node["score"]
    promotion = node.get("promotion") or {}
    if promotion.get("status") == "ok":
        score += promotion.get("delta_vs_baseline", 0.0)
    return score


def select(observation, seed):
    rng = random.Random(seed)
    nodes = observation["nodes"]
    if observation["round"] >= observation["limit"]:
        return None
    successful = [n for n in nodes[1:] if n["status"] == "ok"]
    if not successful:
        return "root"
    leaves = [n for n in successful if n["id"] in observation["leaves"]]
    if not leaves:
        return "root"
    best = max(evidence_score(n) for n in leaves)
    candidates = [n["id"] for n in leaves if evidence_score(n) >= best - 0.01]
    if observation["stop_allowed"] and len(nodes) >= 7:
        recent = [evidence_score(n) for n in successful[-3:]]
        earlier = [evidence_score(n) for n in successful[:-3]]
        if earlier and max(recent) <= max(earlier):
            return None
    return rng.choice(candidates)
