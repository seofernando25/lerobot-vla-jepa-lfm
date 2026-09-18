"""Trusted, deterministic structural screening; never imports candidate code.

Fingerprints are a conservative syntax proxy, not proof of semantic novelty.
Constants, repetition counts and activation choices are deliberately invisible.
"""

import ast
import json
import re
from pathlib import Path

from rsi.core import PLUGIN

AXES = (
    "representation_source",
    "information_flow",
    "adaptation_location",
    "conditioning_target",
    "trainability",
)
IGNORED = {
    "super",
    "len",
    "range",
    "enumerate",
    "zip",
    "list",
    "tuple",
    "dict",
    "set",
    "int",
    "float",
    "bool",
    "str",
    "getattr",
    "setattr",
    "hasattr",
    "isinstance",
    "issubclass",
    "print",
    "ValueError",
    "RuntimeError",
    "TypeError",
    "assert",
    "warn",
    "warning",
    "info",
    "debug",
    "to",
    "type",
    "half",
    "bfloat16",
    "device",
    "dtype",
    "contiguous",
    "view",
    "reshape",
    "flatten",
    "unflatten",
    "transpose",
    "permute",
    "unsqueeze",
    "squeeze",
    "expand",
    "expand_as",
    "repeat",
    "repeat_interleave",
    "clone",
    "detach",
    "size",
    "dim",
    "numel",
    "item",
    "cpu",
    "cuda",
    "new_zeros",
    "zeros",
    "ones",
    "empty",
    "zeros_like",
    "ones_like",
    "arange",
    "tensor",
    "as_tensor",
    "register_buffer",
    "normal_",
    "uniform_",
    "zeros_",
    "ones_",
    "constant_",
    "xavier_uniform_",
    "kaiming_uniform_",
    "relu",
    "gelu",
    "silu",
    "sigmoid",
    "tanh",
    "softplus",
    "ReLU",
    "GELU",
    "SiLU",
    "Sigmoid",
    "Tanh",
    "Dropout",
    "Identity",
}


def normalize(value):
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def axes_signature(axes):
    return "|".join(normalize(axes[k]) for k in AXES)


def parse_proposal(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=pairs)
    schema = json.loads(Path(__file__).with_name("proposal_schema.json").read_text())

    def validate(value, spec):
        kind = spec["type"]
        if kind == "object":
            if not isinstance(value, dict) or set(value) != set(spec["required"]):
                raise ValueError("proposal object must contain exactly the schema keys")
            for key, child in value.items():
                validate(child, spec["properties"][key])
        elif kind == "array":
            if (
                not isinstance(value, list)
                or not spec["minItems"] <= len(value) <= spec["maxItems"]
            ):
                raise ValueError("invalid proposal checks")
            for child in value:
                validate(child, spec["items"])
        elif not isinstance(value, str) or not value.strip():
            raise ValueError("proposal values must be nonempty strings")
        elif len(value) > spec.get("maxLength", 1600):
            raise ValueError("proposal value too long")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError("invalid proposal relation")

    validate(value, schema)
    return value


def architecture_fingerprint(workspace):
    features = set()
    for path in sorted((Path(workspace) / PLUGIN).rglob("*.py")):
        filename = str(path.relative_to(Path(workspace) / PLUGIN))

        def name(node):
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                return name(node.value) + "." + node.attr
            return ""

        def visit(node, scope="module", filename=filename):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                scope += "." + node.name
                features.add(f"{filename}:{kind}:{scope}")
            if isinstance(node, ast.Call):
                call = name(node.func)
                if call and call.split(".")[-1] not in IGNORED:
                    features.add(f"{filename}:call:{scope}:{call}")
                    # Call dependencies capture topology changes with the same call vocabulary.
                    for arg in [*node.args, *(k.value for k in node.keywords)]:
                        if isinstance(arg, ast.Call):
                            child = name(arg.func)
                            if child and child.split(".")[-1] not in IGNORED:
                                features.add(f"{filename}:edge:{scope}:{child}->{call}")
            for child in ast.iter_child_nodes(node):
                visit(child, scope)

        visit(ast.parse(path.read_text()))
    return features


def architecture_delta(base, candidate):
    return architecture_fingerprint(base) ^ architecture_fingerprint(candidate)


def architecture_similarity(delta_a, delta_b):
    a, b = set(delta_a), set(delta_b)
    return len(a & b) / len(a | b) if a or b else 1.0


def check_novelty(proposal, delta, parent, references, config):
    if len(delta) < config["min_refinement_arch_delta"]:
        raise ValueError("insufficient AST structural delta; parameter-only changes are forbidden")
    if parent != "root":
        if proposal["relation_to_parent"] != "structural_refinement":
            raise ValueError("non-root proposal must declare structural_refinement")
        return
    if proposal["relation_to_parent"] != "new_family":
        raise ValueError("root proposal must declare new_family")
    signature = axes_signature(proposal["architecture_axes"])
    for ref in references:
        if signature == ref["axes_signature"]:
            raise ValueError("duplicate root architecture_axes signature")
        if (
            architecture_similarity(delta, ref["architecture_delta"])
            > config["max_root_arch_similarity"]
        ):
            # Do not expose reference features, labels, IDs or v1 results to workers.
            raise ValueError("root AST architecture overlaps a previously accepted mechanism")


def compact_node(node):
    proposal = node.get("proposal") or {}
    return {
        **{
            k: node.get(k)
            for k in ("id", "outer", "parent", "score", "status", "batch_id", "batch_slot")
        },
        **{
            k: proposal.get(k, node.get(k))
            for k in ("mechanism_family", "architecture_axes", "structural_change")
        },
        "family_id": node.get("family_id"),
        "eligible": node.get("eligible", node.get("status") == "ok"),
        "failure_reason": (
            str(node.get("summary", ""))[:800] if node.get("status") != "ok" else None
        ),
    }
