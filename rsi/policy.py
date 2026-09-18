"""Restricted policy language and isolated, timeout-bounded execution."""

from __future__ import annotations

import ast
import json
import math
import random
import subprocess
import sys
from pathlib import Path

from rsi.core import observation, validate_action

SAFE_BUILTINS = {
    k: getattr(__import__("builtins"), k)
    for k in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    )
}


def validate_source(source):
    tree = ast.parse(source)
    if not any(isinstance(n, ast.FunctionDef) and n.name == "select" for n in tree.body):
        raise ValueError("policy requires select(observation, seed)")
    forbidden = (ast.ImportFrom, ast.ClassDef, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError("unsupported policy syntax")  # noqa: TRY004
        if isinstance(node, ast.Import) and any(
            n.name not in {"random", "math"} or n.asname for n in node.names
        ):
            raise ValueError("only unaliased random/math imports allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("private attribute access forbidden")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder access forbidden")
    return tree


def execute(source, obs, seed):
    tree = validate_source(source)
    tree.body = [n for n in tree.body if not isinstance(n, ast.Import)]
    namespace = {"__builtins__": SAFE_BUILTINS, "random": random, "math": math}
    random.seed(seed)
    exec(compile(tree, "policy.py", "exec"), namespace)  # noqa: S102 - restricted AST/child process
    action = namespace["select"](obs, seed)
    validate_action(action, obs)
    return action


class Policy:
    def __init__(self, path, timeout=5):
        self.source = Path(path).read_text()
        validate_source(self.source)
        self.timeout = timeout
        # Validate real interface before any replay, including a non-root world.
        root = {"id": "root", "parent": None, "score": None, "status": "root", "summary": ""}
        self(observation([root], 6), 42)
        leaf = {"id": "n1", "parent": "root", "score": -1.0, "status": "ok", "summary": ""}
        self(observation([root, leaf], 6), 42)

    def __call__(self, obs, seed):
        process = subprocess.run(
            [sys.executable, "-m", "rsi.policy"],
            input=json.dumps({"source": self.source, "obs": obs, "seed": seed}),
            text=True,
            capture_output=True,
            timeout=self.timeout,
            cwd=Path(__file__).resolve().parents[1],
            check=True,
        )
        action = json.loads(process.stdout)
        validate_action(action, obs)
        return action


if __name__ == "__main__":
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    payload = json.load(sys.stdin)
    print(json.dumps(execute(payload["source"], payload["obs"], payload["seed"])))
