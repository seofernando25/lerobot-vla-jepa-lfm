"""W=1 parallel-refining approximation. Only revealed observations are available."""

import random


def select(observation, seed):
    rng = random.Random(seed)
    nodes = observation["nodes"]
    if observation["round"] >= observation["limit"]:
        return None
    if len(nodes) < 4:  # Three independent initial branches.
        return "root"
    successful = [n for n in nodes[1:] if n["status"] == "ok"]
    if not successful:
        return None
    # Stop after three successive attempts without a measured improvement.
    if len(nodes) >= 7:
        earlier = max(n["score"] for n in nodes[1:-3])
        if max(n["score"] for n in nodes[-3:]) <= earlier:
            return None
    leaves = [n for n in successful if n["id"] in observation["leaves"]]
    if not leaves:
        return "root"
    best = max(n["score"] for n in leaves)
    return rng.choice([n["id"] for n in leaves if abs(n["score"] - best) <= 0.01])
