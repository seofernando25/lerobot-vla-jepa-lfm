"""Batched logical exploration with one physical evaluator and event sourcing."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from rsi.core import (
    PLUGIN,
    Journal,
    atomic,
    candidate_guard,
    digest,
    manifest,
    mechanism_families,
    observation,
    plan_batch_actions,
    replay,
)
from rsi.evaluator import parse_metric, parse_training_telemetry, train_args
from rsi.novelty import (
    AXES,
    architecture_delta,
    axes_signature,
    check_novelty,
    compact_node,
    normalize,
    parse_proposal,
)
from rsi.policy import Policy
from rsi.process import OperatorStop, ProcessTimeout, codex_argv, isolated, run_logged

ROOT = {
    "id": "root",
    "parent": None,
    "score": None,
    "status": "root",
    "summary": "Clean base LFM source; unmeasured, not a zero-quality baseline.",
}


def assert_no_model_artifacts(output):
    for path in Path(output).rglob("*"):
        if path.is_file() and (
            path.suffix.lower() in {".safetensors", ".pt", ".pth", ".distcp"}
            or path.name == ".metadata"
        ):
            raise ValueError(f"search probe retained heavyweight model artifact: {path}")


def now():
    return datetime.now(UTC).isoformat()


GPU_RUNTIME_SIGNATURES = (
    "cuda error: unspecified launch failure",
    "cudaerrorlaunchfailure",
    "cuda initialization: cuda unknown error",
    "cuda driver initialization failed",
    "driver shutting down",
    "no cuda-capable device",
    "gpu runtime health check failed",
)


def classify_external_failure(exc, logs):
    if isinstance(exc, OperatorStop):
        return "interrupted"
    text = str(exc).casefold()
    logs = Path(logs)
    for name in ("stderr.log", "stdout.log"):
        path = logs / name
        if path.is_file():
            text += "\n" + path.read_text(errors="replace").casefold()
    if isinstance(exc, ProcessTimeout) or any(
        signature in text for signature in GPU_RUNTIME_SIGNATURES
    ):
        return "runtime_failure"
    return "implementation_failure"


def load_config(path):
    c = json.loads(Path(path).read_text())
    if c.get("protocol_version") != "dream-rsi-v4-continuation":
        raise ValueError(
            "v4 requires a fresh dream-rsi-v4-continuation study; old state cannot be reused"
        )
    for key in (
        "proposal_batch_size",
        "proposal_parallel_workers",
        "min_free_disk_gib",
        "max_proposal_retries",
        "root_open_target",
        "min_nodes_before_stop",
        "min_root_branches_before_stop",
        "min_distinct_families_before_stop",
        "min_outer_iterations_before_stop",
        "min_total_measured_nodes_before_stop",
        "min_refinement_arch_delta",
        "probe_steps",
        "promotion_steps",
        "max_promotions",
        "K1",
        "K2",
        "max_outer_iterations",
        "max_real_attempts",
        "max_total_reservations",
        "max_runtime_failures",
        "max_eval_samples",
        "batch_size",
        "agent_timeout_seconds",
        "probe_timeout_seconds",
        "promotion_timeout_seconds",
        "policy_timeout_seconds",
        "refinement_slots_per_batch",
        "novel_slots_per_batch",
    ):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if c["physical_evaluator_workers"] != 1:
        raise ValueError("physical_evaluator_workers must equal 1")
    if c["W"] != 1 or c["policy_versions"] != 4 or c["replay_trajectories"] != 100:
        raise ValueError("fixed contract requires W=1, four version slots, 100 trajectories")
    if c["precision"] != "bf16" or not 0 < c["dataset"]["eval_split"] < 1:
        raise ValueError("invalid precision or held-out split")
    if c["probe_steps"] != 500 or c["promotion_steps"] <= c["probe_steps"]:
        raise ValueError("screening must be 500 steps and promotion must be longer")
    if not isinstance(c["promotion_margin"], (int, float)) or not 0 < c["promotion_margin"] < 0.1:
        raise ValueError("invalid promotion_margin")
    if not 0 <= c["max_root_arch_similarity"] <= 1:
        raise ValueError("invalid root similarity threshold")
    if c["refinement_slots_per_batch"] + c["novel_slots_per_batch"] != c["proposal_batch_size"]:
        raise ValueError("mixed batch slots must equal proposal_batch_size")
    anchors = c.get("continuation_anchor_ids")
    if not isinstance(anchors, list) or not anchors or len(set(anchors)) != len(anchors):
        raise ValueError("continuation_anchor_ids must be a nonempty unique list")
    if not all(isinstance(x, str) and x.startswith("n") for x in anchors):
        raise ValueError("invalid continuation anchor id")
    if c["max_total_reservations"] < c["max_real_attempts"]:
        raise ValueError("max_total_reservations must cover max_real_attempts")
    if (
        c["min_nodes_before_stop"] > min(c["K1"], c["K2"])
        or c["min_root_branches_before_stop"] > c["root_open_target"]
        or c["root_open_target"] > c["K1"]
        or c["min_outer_iterations_before_stop"] > c["max_outer_iterations"]
        or c["min_total_measured_nodes_before_stop"] > c["max_real_attempts"]
    ):
        raise ValueError("inconsistent coverage floors")
    if c["novelty_reference"] != "rsi/studies/v1_novelty_reference.json":
        raise ValueError("novelty reference must be the tracked trusted reference")
    if c["beta1"] < 0 or c["beta_diversity"] < 0 or c["failure_score"] >= 0:
        raise ValueError("invalid objective constants")
    return c


@contextlib.contextmanager
def lock(state):
    state.mkdir(parents=True, exist_ok=True)
    with (state / "runner.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another runner holds this state") from exc
        yield


def source_contract(repo, target):
    target.mkdir(parents=True)
    for source in sorted((repo / PLUGIN).rglob("*.py")):
        if source.is_symlink():
            raise ValueError("plugin source symlink forbidden")
        destination = target / source.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    text = (repo / "pyproject.toml").read_text()
    text = "\n".join(line for line in text.splitlines() if not line.startswith("readme ="))
    (target / "pyproject.toml").write_text(text + "\n")
    if (repo / "uv.lock").exists():
        shutil.copyfile(repo / "uv.lock", target / "uv.lock")
    (target / "tests").mkdir()
    shutil.copyfile(repo / "rsi/contract_test.py", target / "tests/test_contract.py")
    shutil.copyfile(repo / "rsi/proposal_schema.json", target / "proposal_schema.json")
    candidate_guard(target, manifest(target))


def harness_manifest(repo):
    return {
        k: v
        for k, v in manifest(repo / "rsi").items()
        if "__pycache__" not in k and not k.endswith(".pyc")
    }


class Runner:
    def __init__(self, repo, state, synthetic=False):
        self.repo, self.state = Path(repo).resolve(), Path(state).resolve()
        self.synthetic = synthetic
        self.journal = Journal(self.state)
        self.stop = self.state / "STOP"

    def events(self, kind=None):
        events = self.journal.read()
        return [e for e in events if kind is None or e["kind"] == kind]

    def initialize(self, config_path, prior_study=None):
        if self.events():
            raise ValueError("already initialized; use status or run --resume")
        config = load_config(config_path)
        if prior_study is None and not self.synthetic:
            raise ValueError(
                "v4 continuation requires --prior-study pointing at the sealed V3 state"
            )

        prior_history, prior_roots, anchors = [], [], []
        prior_path = Path(prior_study).resolve() if prior_study is not None else None
        prior_status = "synthetic"
        prior = []
        if prior_path is not None:
            if not (prior_path / "events").is_dir():
                raise ValueError("prior study journal missing")
            prior = Journal(prior_path).read()
            prior_init = next(e for e in prior if e["kind"] == "initialized")
            version = prior_init["config"]["protocol_version"]
            if version not in {"dream-rsi-v2", "dream-rsi-v3-batched", "dream-rsi-v4-continuation"}:
                raise ValueError("only v2/v3/v4 mechanism history may be imported")
            attempts = [e for e in prior if e["kind"] == "attempt"]
            outcomes = [e for e in prior if e["kind"] == "outcome"]
            finished = any(e["kind"] == "finished" for e in prior)
            exhausted = len(attempts) >= prior_init["config"]["max_real_attempts"] and len(
                outcomes
            ) == len(attempts)
            if not finished and not exhausted:
                raise ValueError(
                    "prior study must be finished or exhausted with every reservation closed"
                )
            prior_status = "finished" if finished else "exhausted"
            prior_history = list(prior_init.get("prior_history", [])) + [
                compact_node(e["node"])
                for e in prior
                if e["kind"] == "outcome" and e["node"].get("proposal")
            ]
            prior_roots = list(prior_init.get("prior_roots", [])) + [
                {k: e[k] for k in ("architecture_delta", "axes_signature")}
                for e in prior
                if e["kind"] == "proposal_accepted"
                and e["parent"] == "root"
                and not e.get("source_anchor")
            ]

        source_contract(self.repo, self.state / "base")
        anchor_root = self.state / "anchors"
        anchor_root.mkdir()
        if prior_path is None:
            for index, identifier in enumerate(config["continuation_anchor_ids"]):
                destination = anchor_root / identifier
                shutil.copytree(self.state / "base", destination)
                anchors.append(
                    {
                        "id": identifier,
                        "score": -0.70 - index * 0.01,
                        "status": "ok",
                        "family_id": "synthetic anchor " + identifier,
                        "mechanism_family": "synthetic anchor " + identifier,
                        "architecture_axes": dict.fromkeys(AXES, identifier),
                        "structural_change": "Synthetic continuation anchor",
                        "workspace_hash": manifest(destination),
                    }
                )
        else:
            outcome_by_id = {e["attempt"]: e for e in prior if e["kind"] == "outcome"}
            for identifier in config["continuation_anchor_ids"]:
                if identifier not in outcome_by_id:
                    raise ValueError(f"continuation anchor missing from prior study: {identifier}")
                event = outcome_by_id[identifier]
                node = event["node"]
                if (
                    node.get("status") != "ok"
                    or not node.get("eligible")
                    or not event.get("workspace_hash")
                ):
                    raise ValueError(
                        f"continuation anchor is not an intact measured candidate: {identifier}"
                    )
                source = prior_path / "attempts" / identifier / "workspace"
                if manifest(source) != event["workspace_hash"]:
                    raise ValueError(f"continuation anchor workspace changed: {identifier}")
                destination = anchor_root / identifier
                shutil.copytree(source, destination)
                anchor = compact_node(node)
                anchor["workspace_hash"] = manifest(destination)
                anchors.append(anchor)

        (self.state / "policies").mkdir()
        shutil.copyfile(self.repo / "rsi/policies/initial.py", self.state / "policies/p0.py")
        atomic(self.state / "config.json", config)
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        self.journal.append(
            "initialized",
            config=config,
            prior_history=prior_history,
            prior_roots=prior_roots,
            prior_study=str(prior_path) if prior_path else None,
            prior_status=prior_status,
            continuation_anchors=anchors,
            config_hash=digest(config),
            harness=harness_manifest(self.repo),
            base=manifest(self.state / "base"),
            git_sha=revision,
            lerobot_sha=config["lerobot_sha"],
            synthetic=self.synthetic,
            at=now(),
            policy="p0",
            policy_hash=digest((self.state / "policies/p0.py").read_bytes()),
        )

    def verify(self):
        init = self.events("initialized")[0]
        c = load_config(self.state / "config.json")
        if digest(c) != init["config_hash"] or harness_manifest(self.repo) != init["harness"]:
            raise ValueError("frozen config/harness changed; initialize a separate study")
        if manifest(self.state / "base") != init["base"]:
            raise ValueError("base workspace changed")
        for anchor in init["continuation_anchors"]:
            if manifest(self.state / "anchors" / anchor["id"]) != anchor["workspace_hash"]:
                raise ValueError(f"continuation anchor changed: {anchor['id']}")
        if self.synthetic != init["synthetic"]:
            raise ValueError("synthetic/real state cannot be mixed")
        self.config = c
        return init

    def anchor_metadata(self, identifier=None):
        anchors = self.events("initialized")[0].get("continuation_anchors", [])
        if identifier is None:
            return list(anchors)
        return next(a for a in anchors if a["id"] == identifier)

    def anchor_workspace(self, identifier):
        return self.state / "anchors" / identifier

    def policy(self, name):
        path = self.state / "policies" / f"{name}.py"
        expected = (
            self.events("initialized")[0]["policy_hash"]
            if name == "p0"
            else next(e["policy_hash"] for e in self.events("version") if e["policy"] == name)
        )
        if digest(path.read_bytes()) != expected:
            raise ValueError("saved policy was modified")
        return Policy(path, self.config["policy_timeout_seconds"])

    def promotion_for(self, attempt):
        events = [e for e in self.events("promotion") if e["attempt"] == attempt]
        if not events:
            return None
        event = events[-1]
        return {
            "status": event["status"],
            "score": (event.get("metrics") or {}).get("score"),
            "eval_loss": (event.get("metrics") or {}).get("eval_loss"),
            "delta_vs_baseline": event.get("delta_vs_baseline"),
            "failure_class": event.get("failure_class"),
        }

    def root_node(self):
        root = dict(ROOT)
        baseline = self.events("baseline")
        if baseline:
            event = baseline[-1]
            root.update(
                score=event["metrics"]["score"],
                status="ok",
                summary="Measured clean base LFM under the fixed screening evaluator.",
                metrics=event["metrics"],
            )
        promoted = self.events("promotion_baseline")
        if promoted:
            root["promotion"] = {
                "status": "ok",
                "score": promoted[-1]["metrics"]["score"],
                "eval_loss": promoted[-1]["metrics"]["eval_loss"],
                "delta_vs_baseline": 0.0,
            }
        return root

    def tree(self, outer):
        nodes = [self.root_node()]
        for event in self.events("outcome"):
            if event["outer"] != outer or not event["node"].get("proposal"):
                continue
            node = json.loads(json.dumps(event["node"]))
            promotion = self.promotion_for(event["attempt"])
            if promotion:
                node["promotion"] = promotion
            nodes.append(node)
        return nodes

    def ensure_baseline(self):
        if self.events("baseline"):
            return
        if self.events("baseline_started"):
            raise InterruptedError("baseline may have run; never relaunch in this study")
        self.check_disk()
        self.journal.append("baseline_started", at=now())
        if self.synthetic:
            metrics = {
                "score": -0.75,
                "eval_loss": 0.75,
                "optimizer_steps": self.config["probe_steps"],
                "wall_seconds": 0.0,
                "metric_provenance": "synthetic base; no training",
            }
        else:
            output = self.state / "baseline" / "evaluation"
            output.mkdir(parents=True, exist_ok=True)
            metrics = self.evaluate_workspace(
                self.state / "base",
                output,
                self.state / "baseline" / "probe",
                steps=self.config["probe_steps"],
            )
        self.journal.append(
            "baseline",
            metrics=metrics,
            workspace_hash=manifest(self.state / "base"),
            ended_at=now(),
        )

    def check_disk(self):
        if shutil.disk_usage(self.state).free < self.config["min_free_disk_gib"] * 1024**3:
            self.journal.append("disk_stop", at=now())
            self.stop.touch()
            raise OperatorStop("free disk below min_free_disk_gib")

    def gpu_healthcheck(self):
        if self.synthetic:
            return
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                cwd=self.repo,
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            subprocess.run(
                [
                    str(self.repo / ".venv/bin/python"),
                    "-c",
                    (
                        "import torch; "
                        "assert torch.cuda.is_available(), 'cuda unavailable'; "
                        "x=torch.ones(1, device='cuda'); "
                        "torch.cuda.synchronize(); "
                        "print(torch.cuda.get_device_name(0))"
                    ),
                ],
                cwd=self.repo,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        except Exception as exc:
            raise RuntimeError(f"GPU runtime health check failed: {exc}") from exc

    def evaluate_workspace(self, workspace, output, logs, steps=None, timeout=None):
        self.check_disk()
        self.gpu_healthcheck()
        steps = self.config["probe_steps"] if steps is None else steps
        timeout = (
            self.config["probe_timeout_seconds"]
            if timeout is None and steps == self.config["probe_steps"]
            else self.config["promotion_timeout_seconds"]
            if timeout is None
            else timeout
        )
        evaluator = self.repo / "rsi/evaluator.py"
        argv = isolated(
            [
                str(self.repo / ".venv/bin/python"),
                "/tmp/evaluator.py",
                *train_args(self.config, steps=steps),
            ],
            self.repo,
            workspace,
            output,
            evaluator=True,
        )
        boundary = argv.index("--")
        argv[boundary:boundary] = ["--ro-bind", str(evaluator), "/tmp/evaluator.py"]
        dataset_root = Path(os.environ.get("RSI_DATASET_ROOT", "")).expanduser().resolve()
        if not dataset_root.is_dir():
            raise ValueError(
                "RSI_DATASET_ROOT must point to the fixed local LIBERO-Spatial LeRobot dataset"
            )
        argv[boundary:boundary] = ["--ro-bind", str(dataset_root), "/tmp/dataset"]
        try:
            elapsed = run_logged(argv, self.repo, logs, timeout, self.stop)
        finally:
            assert_no_model_artifacts(output)
        text = (logs / "stderr.log").read_text()
        text += (logs / "stdout.log").read_text()
        metrics = parse_metric(text, steps)
        hardware = json.loads((output / "hardware.json").read_text())
        eval_samples = json.loads((output / "eval_samples.json").read_text())
        trainability = json.loads((output / "trainability.json").read_text())
        metrics.update(
            wall_seconds=elapsed,
            hardware=hardware,
            eval_samples=eval_samples,
            trainability=trainability,
        )
        telemetry = parse_training_telemetry(text)
        telemetry.update(
            final_eval_loss=metrics["eval_loss"],
            wall_seconds=elapsed,
            hardware=hardware,
            eval_samples=eval_samples,
            trainable_parameter_count=sum(item.get("parameters", 0) for item in trainability),
        )
        atomic(logs.parent / "telemetry.json", telemetry)
        return metrics

    def workspace(self, outer, node):
        if node.startswith("anchor:"):
            return self.anchor_workspace(node.split(":", 1)[1])
        return (
            self.state / "base" if node == "root" else self.state / "attempts" / node / "workspace"
        )

    def recover(self):
        self._frozen_history = None
        finished = {e["attempt"] for e in self.events("outcome")}
        for event in self.events("attempt"):
            if event["attempt"] not in finished:
                accepted = next(
                    (
                        e
                        for e in self.events("proposal_accepted")
                        if e["attempt"] == event["attempt"]
                    ),
                    {},
                )
                self.finish(
                    event,
                    "interrupted",
                    "Reserved attempt interrupted; never relaunched.",
                    proposal=accepted.get("proposal"),
                    delta=accepted.get("architecture_delta"),
                    family_id=accepted.get("family_id"),
                    source_anchor=event.get("source_anchor"),
                )

        promoted = {e["attempt"] for e in self.events("promotion")}
        for event in self.events("promotion_started"):
            if event["attempt"] not in promoted:
                self.journal.append(
                    "promotion",
                    attempt=event["attempt"],
                    outer=event["outer"],
                    status="interrupted",
                    metrics={},
                    baseline_score=(
                        self.events("promotion_baseline")[-1]["metrics"]["score"]
                        if self.events("promotion_baseline")
                        else None
                    ),
                    delta_vs_baseline=None,
                    failure_class="interrupted",
                    summary="Reserved promotion interrupted; never relaunched.",
                    ended_at=now(),
                )

        baseline_failures = {
            e.get("reservation", 1) for e in self.events("promotion_baseline_failure")
        }
        if not self.events("promotion_baseline"):
            for event in self.events("promotion_baseline_started"):
                reservation = event.get("reservation", 1)
                if reservation not in baseline_failures:
                    self.journal.append(
                        "promotion_baseline_failure",
                        reservation=reservation,
                        status="interrupted",
                        summary="Reserved promotion baseline interrupted; never relaunched.",
                        ended_at=now(),
                    )

        closed = {e["batch_id"] for e in self.events("batch_done")}
        for batch in self.events("batch_started"):
            if batch["batch_id"] not in closed:
                self.journal.append(
                    "batch_done",
                    batch_id=batch["batch_id"],
                    outer=batch["outer"],
                    reason="interrupted",
                    at=now(),
                )

    def finish(
        self,
        event,
        status,
        summary,
        metrics=None,
        workspace_hash=None,
        proposal=None,
        delta=None,
        family_id=None,
        source_anchor=None,
    ):
        metrics = metrics or {}
        node = {
            "batch_id": event.get("batch_id"),
            "batch_slot": event.get("batch_slot"),
            "id": event["attempt"],
            "parent": event["parent"],
            "source_anchor": source_anchor
            if source_anchor is not None
            else event.get("source_anchor"),
            "status": status,
            "score": metrics.get("score", self.config["failure_score"]),
            "summary": summary,
            "metrics": metrics,
            "outer": event["outer"],
            "proposal": proposal,
            "architecture_delta": sorted(delta or []),
            "axes_signature": axes_signature(proposal["architecture_axes"]) if proposal else None,
            "family_id": family_id,
            "eligible": status == "ok" and workspace_hash is not None,
        }
        self.journal.append(
            "outcome",
            outer=event["outer"],
            attempt=event["attempt"],
            node=node,
            batch_id=event.get("batch_id"),
            batch_slot=event.get("batch_slot"),
            source_anchor=source_anchor
            if source_anchor is not None
            else event.get("source_anchor"),
            workspace_hash=workspace_hash,
            ended_at=now(),
        )

    def ledger(self):
        prior = self.events("initialized")[0].get("prior_history", [])
        current = []
        for event in self.events("outcome"):
            if not event["node"].get("proposal"):
                continue
            node = json.loads(json.dumps(event["node"]))
            promotion = self.promotion_for(event["attempt"])
            if promotion:
                node["promotion"] = promotion
            current.append(compact_node(node))
        return prior + current

    def root_references(self):
        reference = json.loads((self.repo / self.config["novelty_reference"]).read_text())
        if reference["fingerprint_version"] != 1:
            raise ValueError("unsupported novelty fingerprint version")
        return (
            reference["roots"]
            + self.events("initialized")[0].get("prior_roots", [])
            + [
                e
                for e in self.events("proposal_accepted")
                if e["parent"] == "root" and not e.get("source_anchor")
            ]
        )

    def discovery_prompt(self, outer, parent, feedback):
        fixed = {
            k: self.config[k]
            for k in (
                "probe_steps",
                "promotion_steps",
                "promotion_margin",
                "seed",
                "batch_size",
                "precision",
                "max_eval_samples",
            )
        }
        anchor = None
        if parent.startswith("anchor:"):
            anchor = self.anchor_metadata(parent.split(":", 1)[1])
        return (self.repo / "rsi/prompts/discovery.txt").read_text() + json.dumps(
            {
                "selected_parent": parent,
                "continuation_mode": "novel" if parent == "root" else "structural_refinement",
                "continuation_anchor": anchor,
                "priority_anchor_ids": self.config["continuation_anchor_ids"],
                "global_history": self._frozen_history[0]
                if getattr(self, "_frozen_history", None)
                else self.ledger(),
                "current_tree": self._frozen_history[1]
                if getattr(self, "_frozen_history", None)
                else [compact_node(n) for n in self.tree(outer)],
                "fixed_probe": fixed,
                "rejection_feedback": feedback,
            },
            separators=(",", ":"),
        )

    def discover(self, outer, parent, identifier, retry, workspace, directory, feedback):
        if self.synthetic:
            # Exercise real schema, guard and novelty paths without a model or subprocess.
            label = identifier + "_" + str(retry)
            (workspace / PLUGIN / (label + ".py")).write_text(
                f"class Synthetic_{label}:\n"
                "    def encode(self, x): return self.route(self.project(x))\n"
                "    def project(self, x): return x\n"
                "    def route(self, x): return x\n"
            )
            return parse_proposal(
                json.dumps(
                    {
                        "hypothesis": "Synthetic hypothesis " + label,
                        "mechanism_family": "synthetic " + identifier,
                        "architecture_axes": dict.fromkeys(AXES, label),
                        "relation_to_parent": "new_family"
                        if parent == "root"
                        else "structural_refinement",
                        "structural_change": "Synthetic module " + label,
                        "why_not_parameter_only": "Adds structural definitions and call edges",
                        "expected_effect": "No scientific interpretation",
                        "implementation_checks": ["Synthetic syntax only"],
                        "limitations": "No training or discovery performed",
                    }
                )
            )
        prompt = self.discovery_prompt(outer, parent, feedback)
        (directory / "prompt.txt").write_text(prompt)
        args = codex_argv()
        args[-1:-1] = [
            "--output-schema",
            "/tmp/work/proposal_schema.json",
            "-o",
            "/tmp/work/.proposal.json",
        ]
        run_logged(
            isolated(args, self.repo, workspace),
            self.repo,
            directory / "discovery",
            self.config["agent_timeout_seconds"],
            self.stop,
            prompt,
        )
        output = workspace / ".proposal.json"
        if not output.is_file() or output.is_symlink():
            raise ValueError("discovery did not emit a regular proposal JSON file")
        text = output.read_text()
        output.unlink()
        return parse_proposal(text)

    def proposal_worker(self, event, retry, feedback):
        """No journal writes or novelty decisions on worker threads."""
        parent = event["parent"]
        source_anchor = event.get("source_anchor")
        effective_parent = f"anchor:{source_anchor}" if source_anchor else parent
        directory = self.state / "attempts" / event["attempt"]
        workspace = directory / "workspace"
        proposed = None
        try:
            if self.stop.exists():
                raise OperatorStop("stop requested")
            parent_workspace = self.workspace(event["outer"], effective_parent)
            if source_anchor:
                anchor = self.anchor_metadata(source_anchor)
                if manifest(parent_workspace) != anchor["workspace_hash"]:
                    raise ValueError("frozen continuation anchor changed")
            elif parent != "root":
                parent_event = next(e for e in self.events("outcome") if e["attempt"] == parent)
                if (
                    not parent_event["node"]["eligible"]
                    or manifest(parent_workspace) != parent_event["workspace_hash"]
                ):
                    raise ValueError("parent has no intact eligible workspace")
            if workspace.exists():
                shutil.rmtree(workspace)
            shutil.copytree(parent_workspace, workspace)
            before = manifest(workspace)
            session = directory / f"proposal_{retry}"
            session.mkdir()
            proposed = self.discover(
                event["outer"],
                effective_parent,
                event["attempt"],
                retry,
                workspace,
                session,
                feedback,
            )
            proposed = parse_proposal(json.dumps(proposed))
            accepted = candidate_guard(workspace, before)
            delta = architecture_delta(parent_workspace, workspace)
            return {
                "proposal": proposed,
                "workspace_hash": accepted,
                "delta": delta,
                "effective_parent": effective_parent,
            }
        except Exception as exc:  # noqa: BLE001 - untrusted implementation boundary
            return {
                "error": str(exc),
                "proposal": proposed,
                "interrupted": isinstance(exc, OperatorStop),
            }

    def attempt(self, outer, parent):
        # Convenience for synthetic contract callers; production plans a full mixed batch.
        return self.execute_batch(
            outer,
            {
                "planned_actions": [parent],
                "source_anchors": [None],
                "modes": ["policy"],
                "substitutions": [],
                "constrained_actions": 0,
            },
        )

    def research_attempt_count(self, outer=None):
        outcomes = self.events("outcome")
        if outer is not None:
            outcomes = [e for e in outcomes if e["outer"] == outer]
        return sum(e["node"]["status"] in {"ok", "implementation_failure"} for e in outcomes)

    def runtime_failure_count(self):
        outcome_failures = sum(
            e["node"]["status"] == "runtime_failure" for e in self.events("outcome")
        )
        promotion_failures = sum(
            e.get("status") == "runtime_failure" for e in self.events("promotion")
        )
        baseline_failures = sum(
            e.get("status") == "runtime_failure" for e in self.events("promotion_baseline_failure")
        )
        return outcome_failures + promotion_failures + baseline_failures

    def promotion_research_count(self):
        return sum(
            e.get("status") in {"ok", "implementation_failure"} for e in self.events("promotion")
        )

    def plan_online_batch(self, outer, policy, obs, size):
        plan = plan_batch_actions(
            policy,
            obs,
            self.config["seed"] + outer * 1009 + obs["round"],
            size,
            self.config,
        )
        plan["source_anchors"] = [None] * len(plan["planned_actions"])
        usage = Counter(
            e.get("source_anchor") for e in self.events("attempt") if e.get("source_anchor")
        )
        unused = [a for a in self.config["continuation_anchor_ids"] if usage[a] == 0]
        selected_anchors = set()
        for slot, mode in enumerate(plan.get("modes", [])):
            if mode != "refinement":
                continue
            anchor = None
            if unused:
                anchor = unused.pop(0)
            elif plan["planned_actions"][slot] == "root":
                candidates = [
                    a for a in self.config["continuation_anchor_ids"] if a not in selected_anchors
                ]
                if candidates:
                    anchor = min(
                        candidates,
                        key=lambda a: (
                            usage[a],
                            self.config["continuation_anchor_ids"].index(a),
                        ),
                    )
            if anchor is not None:
                requested = plan["planned_actions"][slot]
                plan["planned_actions"][slot] = "root"
                plan["source_anchors"][slot] = anchor
                selected_anchors.add(anchor)
                plan["substitutions"].append(
                    {
                        "batch_slot": slot,
                        "requested": requested,
                        "action": "root",
                        "source_anchor": anchor,
                        "reason": "continuation_anchor",
                    }
                )
        plan["constrained_actions"] = len(plan["substitutions"])
        return plan

    def promotion_qualifies(self, metrics):
        if self.promotion_research_count() >= self.config["max_promotions"]:
            return False
        baseline = self.events("baseline")
        if not baseline:
            return False
        root_score = baseline[-1]["metrics"]["score"]
        return metrics["score"] >= root_score - self.config["promotion_margin"]

    def ensure_promotion_baseline(self):
        completed = self.events("promotion_baseline")
        if completed:
            return completed[-1]["metrics"]

        starts = self.events("promotion_baseline_started")
        failures = self.events("promotion_baseline_failure")
        resolved = {e.get("reservation", 1) for e in failures}
        pending = [e for e in starts if e.get("reservation", 1) not in resolved]
        if pending:
            return None
        if failures and failures[-1]["status"] not in {"runtime_failure", "interrupted"}:
            return None
        if self.runtime_failure_count() >= self.config["max_runtime_failures"]:
            return None

        reservation = len(starts) + 1
        self.journal.append(
            "promotion_baseline_started",
            reservation=reservation,
            steps=self.config["promotion_steps"],
            at=now(),
        )
        base_dir = self.state / "promotion" / "baseline" / f"r{reservation:04d}"
        logs = base_dir / "probe"
        try:
            if self.synthetic:
                metrics = {
                    "score": -0.70,
                    "eval_loss": 0.70,
                    "optimizer_steps": self.config["promotion_steps"],
                    "wall_seconds": 0.0,
                    "metric_provenance": "synthetic promotion base; no training",
                }
            else:
                output = base_dir / "evaluation"
                output.mkdir(parents=True, exist_ok=True)
                metrics = self.evaluate_workspace(
                    self.state / "base",
                    output,
                    logs,
                    steps=self.config["promotion_steps"],
                    timeout=self.config["promotion_timeout_seconds"],
                )
            self.journal.append(
                "promotion_baseline",
                reservation=reservation,
                metrics=metrics,
                steps=self.config["promotion_steps"],
                ended_at=now(),
            )
            return metrics
        except Exception as exc:  # noqa: BLE001 - external evaluator boundary
            failure_class = classify_external_failure(exc, logs)
            self.journal.append(
                "promotion_baseline_failure",
                reservation=reservation,
                status=failure_class,
                summary=str(exc),
                ended_at=now(),
            )
            if failure_class in {"runtime_failure", "interrupted"}:
                self.stop.touch()
            return None

    def maybe_promote(self, event, workspace, metrics):
        if not self.promotion_qualifies(metrics) or self.stop.exists():
            return
        baseline = self.ensure_promotion_baseline()
        if baseline is None or self.stop.exists():
            return
        if any(e["attempt"] == event["attempt"] for e in self.events("promotion_started")):
            return
        self.journal.append(
            "promotion_started",
            attempt=event["attempt"],
            outer=event["outer"],
            short_score=metrics["score"],
            steps=self.config["promotion_steps"],
            at=now(),
        )
        directory = self.state / "attempts" / event["attempt"] / "promotion"
        logs = directory / "probe"
        try:
            if self.synthetic:
                promoted = {
                    "score": metrics["score"] + 0.01,
                    "eval_loss": -(metrics["score"] + 0.01),
                    "optimizer_steps": self.config["promotion_steps"],
                    "wall_seconds": 0.0,
                    "metric_provenance": "synthetic promotion; no training",
                }
            else:
                output = directory / "evaluation"
                output.mkdir(parents=True, exist_ok=True)
                promoted = self.evaluate_workspace(
                    workspace,
                    output,
                    logs,
                    steps=self.config["promotion_steps"],
                    timeout=self.config["promotion_timeout_seconds"],
                )
            delta = promoted["score"] - baseline["score"]
            atomic(directory / "metrics.json", promoted)
            self.journal.append(
                "promotion",
                attempt=event["attempt"],
                outer=event["outer"],
                status="ok",
                metrics=promoted,
                baseline_score=baseline["score"],
                delta_vs_baseline=delta,
                ended_at=now(),
            )
        except Exception as exc:  # noqa: BLE001 - external evaluator boundary
            failure_class = classify_external_failure(exc, logs)
            self.journal.append(
                "promotion",
                attempt=event["attempt"],
                outer=event["outer"],
                status=failure_class,
                metrics={},
                baseline_score=baseline["score"],
                delta_vs_baseline=None,
                failure_class=failure_class,
                summary=str(exc),
                ended_at=now(),
            )
            if failure_class in {"runtime_failure", "interrupted"}:
                self.stop.touch()

    def execute_batch(self, outer, plan):
        self.check_disk()
        if self.config["physical_evaluator_workers"] != 1:
            raise ValueError("physical_evaluator_workers must equal 1")
        if (
            len(self.events("attempt")) + len(plan["planned_actions"])
            > self.config["max_total_reservations"]
        ):
            raise ValueError("max_total_reservations would be exceeded")
        plan = dict(plan)
        plan.setdefault("source_anchors", [None] * len(plan["planned_actions"]))
        plan.setdefault("modes", ["policy"] * len(plan["planned_actions"]))
        if len(plan["source_anchors"]) != len(plan["planned_actions"]):
            raise ValueError("source_anchors must align with planned_actions")

        batch_id = f"b{len(self.events('batch_started')) + 1:04d}"
        history = (self.ledger(), [compact_node(n) for n in self.tree(outer)])
        self._frozen_history = history
        references = self.root_references()
        self.journal.append(
            "batch_started",
            batch_id=batch_id,
            outer=outer,
            prefix_hash=digest(history),
            **plan,
            at=now(),
        )
        for substitution in plan["substitutions"]:
            self.journal.append(
                "action_constrained", outer=outer, batch_id=batch_id, **substitution, at=now()
            )

        events = []
        for slot, parent in enumerate(plan["planned_actions"]):
            events.append(
                self.journal.append(
                    "attempt",
                    attempt=f"n{len(self.events('attempt')) + 1:04d}",
                    outer=outer,
                    parent=parent,
                    source_anchor=plan["source_anchors"][slot],
                    mode=plan["modes"][slot] if slot < len(plan["modes"]) else "policy",
                    batch_id=batch_id,
                    batch_slot=slot,
                    started_at=now(),
                    config_hash=digest(self.config),
                    seed=self.config["seed"],
                    steps=self.config["probe_steps"],
                )
            )

        accepted, feedback, sibling_roots = {}, {e["attempt"]: [] for e in events}, []
        for retry in range(1, self.config["max_proposal_retries"] + 1):
            pending = [e for e in events if e["attempt"] not in accepted]
            if not pending or self.stop.exists():
                break
            with ThreadPoolExecutor(max_workers=self.config["proposal_parallel_workers"]) as pool:
                futures = [
                    pool.submit(self.proposal_worker, e, retry, feedback[e["attempt"]])
                    for e in pending
                ]
                results = [f.result() for f in futures]
            for event, result in zip(pending, results, strict=True):
                fields = {
                    k: event[k]
                    for k in (
                        "attempt",
                        "outer",
                        "parent",
                        "source_anchor",
                        "mode",
                        "batch_id",
                        "batch_slot",
                    )
                }
                self.journal.append("proposal_session", **fields, retry=retry, at=now())
                proposed = result.get("proposal")
                reason = result.get("error")
                if result.get("interrupted"):
                    self.stop.touch()
                effective_parent = result.get(
                    "effective_parent",
                    f"anchor:{event['source_anchor']}"
                    if event.get("source_anchor")
                    else event["parent"],
                )
                if reason is None:
                    try:
                        check_novelty(
                            proposed, result["delta"], effective_parent, references, self.config
                        )
                    except ValueError as exc:
                        reason = str(exc)
                    if reason is None and effective_parent == "root":
                        try:
                            check_novelty(
                                proposed, result["delta"], "root", sibling_roots, self.config
                            )
                        except ValueError:
                            reason = (
                                "batch-internal structural collision; propose a "
                                "substantially different root mechanism"
                            )
                if reason is not None:
                    metadata = {
                        k: proposed[k]
                        for k in ("mechanism_family", "architecture_axes", "structural_change")
                        if proposed and k in proposed
                    }
                    self.journal.append(
                        "proposal_rejected",
                        **fields,
                        retry=retry,
                        reason=reason,
                        mechanism=metadata,
                        at=now(),
                    )
                    feedback[event["attempt"]].append(
                        {"retry": retry, "reason": reason, "rejected_mechanism": metadata}
                    )
                    continue

                if event.get("source_anchor"):
                    family = self.anchor_metadata(event["source_anchor"])["family_id"]
                elif event["parent"] == "root":
                    family = normalize(proposed["mechanism_family"])
                else:
                    family = next(n["family_id"] for n in history[1] if n["id"] == event["parent"])
                result["family_id"] = family
                accepted[event["attempt"]] = result
                record = self.journal.append(
                    "proposal_accepted",
                    **fields,
                    proposal=proposed,
                    architecture_delta=sorted(result["delta"]),
                    family_id=family,
                    axes_signature=axes_signature(proposed["architecture_axes"]),
                    at=now(),
                )
                if effective_parent == "root":
                    sibling_roots.append(record)

        self.journal.append("batch_proposals_done", batch_id=batch_id, outer=outer, at=now())
        for event in events:
            result = accepted.get(event["attempt"])
            if result is None:
                self.finish(
                    event,
                    "interrupted" if self.stop.exists() else "proposal_failure",
                    "Proposal sessions stopped or rejected; no probe.",
                    source_anchor=event.get("source_anchor"),
                )
                continue
            directory = self.state / "attempts" / event["attempt"]
            start = time.monotonic()
            proposal = result["proposal"]
            try:
                self.check_disk()
                if self.stop.exists():
                    raise OperatorStop("stop requested")
                if self.synthetic:
                    score = -0.50 + int(event["attempt"][1:]) / 10000
                    metrics = {
                        "score": score,
                        "eval_loss": -score,
                        "optimizer_steps": self.config["probe_steps"],
                        "wall_seconds": 0.0,
                        "metric_provenance": "synthetic; no training",
                    }
                else:
                    output = directory / "evaluation"
                    output.mkdir()
                    metrics = self.evaluate_workspace(
                        directory / "workspace",
                        output,
                        directory / "probe",
                        steps=self.config["probe_steps"],
                    )
                    assert_no_model_artifacts(output)
                metrics["attempt_wall_seconds"] = time.monotonic() - start
                atomic(directory / "metrics.json", metrics)
                self.finish(
                    event,
                    "ok",
                    proposal["mechanism_family"] + ": " + proposal["structural_change"],
                    metrics,
                    result["workspace_hash"],
                    proposal,
                    result["delta"],
                    result["family_id"],
                    source_anchor=event.get("source_anchor"),
                )
                self.maybe_promote(event, directory / "workspace", metrics)
            except Exception as exc:  # noqa: BLE001 - external evaluator boundary
                failure_class = classify_external_failure(exc, directory / "probe")
                self.finish(
                    event,
                    failure_class,
                    str(exc),
                    proposal=proposal,
                    delta=result["delta"],
                    family_id=result["family_id"],
                    source_anchor=event.get("source_anchor"),
                )
                if failure_class in {"runtime_failure", "interrupted"}:
                    self.stop.touch()

        self.journal.append(
            "batch_done",
            batch_id=batch_id,
            outer=outer,
            reason="interrupted" if self.stop.exists() else "complete",
            at=now(),
        )
        self._frozen_history = None

    def improve(self, outer, incumbent):
        cycles = [e for e in self.events("cycle") if e["outer"] == outer]
        if not cycles:
            history = [
                [compact_node(n) for n in self.tree(e["outer"])] for e in self.events("online_done")
            ]
            self.journal.append(
                "cycle",
                outer=outer,
                incumbent=incumbent,
                history=history,
                history_hash=digest(history),
                at=now(),
            )
        cycle = next(e for e in self.events("cycle") if e["outer"] == outer)
        history = cycle["history"]
        slots = [e for e in self.events("version") if e["outer"] == outer]
        for slot in range(len(slots), 4):
            if self.stop.exists():
                return None
            previous = slots[-1]["policy"] if slots else incumbent
            name = incumbent if slot == 0 else f"p{outer}_{slot}"
            reason = None
            if slot:
                reservations = [
                    e for e in self.events("revision") if e["outer"] == outer and e["slot"] == slot
                ]
                if reservations:
                    # A developer invocation may have happened. Never repeat it on restart.
                    reason = "interrupted revision; replay previous valid policy in this slot"
                    name = previous
                else:
                    self.journal.append("revision", outer=outer, slot=slot, at=now())
                    work = self.state / "development" / f"{outer}_{slot}"
                    work.mkdir(parents=True)
                    path = work / "policy.py"
                    shutil.copyfile(self.state / "policies" / f"{previous}.py", path)
                    try:
                        if not self.synthetic:
                            prompt = (self.repo / "rsi/prompts/policy.txt").read_text()
                            prompt += json.dumps(
                                {
                                    "history": history,
                                    "feedback": [policy_feedback(v) for v in slots],
                                    "protocol": self.config,
                                    "beta1": self.config["beta1"],
                                }
                            )
                            run_logged(
                                isolated(codex_argv(), self.repo, work),
                                self.repo,
                                self.state / "development_logs" / f"{outer}_{slot}",
                                self.config["agent_timeout_seconds"],
                                self.stop,
                                prompt,
                            )
                        else:
                            # Different root-opening schedules on the same synthetic worlds.
                            text = path.read_text().replace(
                                'observation["root_open_target"]', str(slot + 3)
                            )
                            path.write_text(text)
                        if set(manifest(work)) != {"policy.py"}:
                            raise ValueError("developer modified files outside policy.py")
                        Policy(path, self.config["policy_timeout_seconds"])
                        shutil.copyfile(path, self.state / "policies" / f"{name}.py")
                    except Exception as exc:  # noqa: BLE001 - external candidate failure boundary
                        name, reason = previous, str(exc)
            path = self.state / "policies" / f"{name}.py"
            policy = Policy(path, self.config["policy_timeout_seconds"])
            trajectories = []
            # Same 25 history/seed worlds for EVERY slot: paired, fair selection.
            for world in range(25):
                tree_index = world % len(history)
                seed = self.config["seed"] + world * 1009
                try:
                    result = replay(
                        history[tree_index],
                        policy,
                        seed,
                        self.config["K2"],
                        self.config["beta1"],
                        self.config["failure_score"],
                        self.config["beta_diversity"],
                        self.config,
                    )
                except Exception as exc:  # noqa: BLE001 - external candidate failure boundary
                    result = {
                        "objective": self.config["failure_score"],
                        "error": str(exc),
                        "trace": [],
                        "revealed": 0,
                        "unique_families": 0,
                        "constrained_actions": 0,
                        "reason": "invalid_policy",
                    }
                trajectories.append(dict(result, history=tree_index, seed=seed))
            event = self.journal.append(
                "version",
                outer=outer,
                slot=slot,
                policy=name,
                policy_hash=digest(path.read_bytes()),
                fallback_reason=reason,
                trajectories=trajectories,
                mean_score=sum(t["objective"] for t in trajectories) / 25,
                eligible=all(t["reason"] != "invalid_policy" for t in trajectories),
            )
            slots.append(event)
        best = max(
            (e for e in slots if e["eligible"]),
            key=lambda e: (e["mean_score"], -e["slot"]),
            default=slots[0],
        )
        self.journal.append(
            "cycle_done",
            outer=outer,
            selected=best["policy"],
            trajectories=sum(len(e["trajectories"]) for e in slots),
            mean_score=best["mean_score"],
        )
        return best["policy"]

    def accepted_measured_count(self):
        return sum(e["node"]["status"] == "ok" for e in self.events("outcome"))

    def global_stop_allowed(self, outer):
        return (
            outer + 1 >= self.config["min_outer_iterations_before_stop"]
            and self.accepted_measured_count()
            >= self.config["min_total_measured_nodes_before_stop"]
        )

    def run(self, resume=False):
        self.verify()
        if self.events("finished"):
            return
        if self.events("started") and not resume:
            raise ValueError("existing run requires explicit --resume")
        if self.stop.exists():
            if not resume:
                raise ValueError("stop requested; explicit --resume required")
            self.stop.unlink()
        self.recover()
        self.ensure_baseline()
        self.journal.append("started", at=now(), resumed=resume)

        for outer in range(self.config["max_outer_iterations"]):
            if self.stop.exists():
                return
            if self.runtime_failure_count() >= self.config["max_runtime_failures"]:
                self.journal.append("finished", reason="runtime_failure_cap", at=now())
                return
            if any(e["outer"] == outer for e in self.events("cycle_done")):
                done = next(e for e in self.events("online_done") if e["outer"] == outer)
                if self.research_attempt_count() >= self.config["max_real_attempts"]:
                    self.journal.append("finished", reason="global_attempt_cap", at=now())
                    return
                if done["reason"] == "policy_STOP" and self.global_stop_allowed(outer):
                    self.journal.append("finished", reason="policy_STOP", at=now())
                    return
                continue

            previous = self.events("cycle_done")
            current = previous[-1]["selected"] if previous else "p0"
            opened = [e for e in self.events("online") if e["outer"] == outer]
            if not opened:
                self.journal.append("online", outer=outer, policy=current, at=now())
            else:
                current = opened[0]["policy"]
            policy = self.policy(current)
            completed = [e for e in self.events("online_done") if e["outer"] == outer]
            reason = completed[0]["reason"] if completed else "K1"

            if not completed:
                while self.research_attempt_count(outer) < self.config["K1"]:
                    if self.stop.exists():
                        return
                    if self.runtime_failure_count() >= self.config["max_runtime_failures"]:
                        reason = "runtime_failure_cap"
                        break
                    if self.research_attempt_count() >= self.config["max_real_attempts"]:
                        reason = "global_attempt_cap"
                        break
                    if len(self.events("attempt")) >= self.config["max_total_reservations"]:
                        reason = "reservation_cap"
                        break

                    nodes = self.tree(outer)
                    obs = observation(
                        [compact_node(n) for n in nodes], self.config["K1"], self.config
                    )
                    self.check_disk()
                    size = min(
                        self.config["proposal_batch_size"],
                        self.config["K1"] - self.research_attempt_count(outer),
                        self.config["max_real_attempts"] - self.research_attempt_count(),
                        self.config["max_total_reservations"] - len(self.events("attempt")),
                    )
                    if size <= 0:
                        reason = "reservation_cap"
                        break
                    plan = self.plan_online_batch(outer, policy, obs, size)
                    if not plan["planned_actions"]:
                        reason = "policy_STOP"
                        break
                    self.execute_batch(outer, plan)

                self.journal.append("online_done", outer=outer, reason=reason, at=now())

            if self.stop.exists():
                return
            selected = self.improve(outer, current)
            if selected is None:
                return
            if self.research_attempt_count() >= self.config["max_real_attempts"]:
                self.journal.append("finished", reason="global_attempt_cap", at=now())
                return
            if reason in {"runtime_failure_cap", "reservation_cap"}:
                self.journal.append("finished", reason=reason, at=now())
                return
            if reason == "policy_STOP":
                if self.global_stop_allowed(outer):
                    self.journal.append("finished", reason="policy_STOP", at=now())
                    return
                self.journal.append(
                    "stop_deferred",
                    outer=outer,
                    measured=self.accepted_measured_count(),
                    required_outer_iterations=self.config["min_outer_iterations_before_stop"],
                    required_measured=self.config["min_total_measured_nodes_before_stop"],
                    at=now(),
                )
        self.journal.append("finished", reason="global_outer_cap", at=now())

    def status(self):
        events = self.events()
        promotions = self.events("promotion")
        return {
            "initialized": bool(events),
            "synthetic": self.synthetic,
            "attempts_reserved": len(self.events("attempt")),
            "research_attempts": self.research_attempt_count() if events else 0,
            "outcomes": len(self.events("outcome")),
            "runtime_failures": self.runtime_failure_count() if events else 0,
            "proposal_sessions": len(self.events("proposal_session")),
            "proposal_rejections": len(self.events("proposal_rejected")),
            "accepted_measured_nodes": self.accepted_measured_count(),
            "distinct_families": len(
                mechanism_families(
                    [
                        ROOT,
                        *[e["node"] for e in self.events("outcome") if e["node"].get("proposal")],
                    ]
                )
            ),
            "continuation_anchors": [
                a["id"] for a in self.events("initialized")[0].get("continuation_anchors", [])
            ]
            if self.events("initialized")
            else [],
            "promotions_reserved": len(self.events("promotion_started")),
            "promotion_research_attempts": self.promotion_research_count() if events else 0,
            "promotions_completed": sum(e["status"] == "ok" for e in promotions),
            "promotion_baseline_ready": bool(self.events("promotion_baseline")),
            "completed_cycles": len(self.events("cycle_done")),
            "replay_trajectories": sum(e["trajectories"] for e in self.events("cycle_done")),
            "stop_requested": self.stop.exists(),
            "finished": self.events("finished")[-1]["reason"] if self.events("finished") else None,
        }


def policy_feedback(version):
    trajectories = version["trajectories"]
    count = len(trajectories)
    return {
        "policy": version["policy"],
        "slot": version["slot"],
        "mean_objective": version["mean_score"],
        "mean_batches": sum(len(t["trace"]) for t in trajectories) / count,
        "mean_planner_substitutions": sum(t["constrained_actions"] for t in trajectories) / count,
        "mean_revealed": sum(t["revealed"] for t in trajectories) / count,
        "mean_unique_families": sum(t["unique_families"] for t in trajectories) / count,
        "reason_counts": dict(Counter(t["reason"] for t in trajectories)),
        "constrained_actions": sum(t["constrained_actions"] for t in trajectories),
        "representative_traces": [t["trace"] for t in trajectories[:2]],
    }


def dry_run(repo):
    with tempfile.TemporaryDirectory(prefix="rsi-synthetic-") as tmp:
        runner = Runner(repo, Path(tmp), synthetic=True)
        config = json.loads((Path(repo) / "rsi/config.json").read_text())
        # Keep dry-run intentionally one-cycle while exercising V4 mixed batches and promotions.
        config["max_outer_iterations"] = 1
        config["min_outer_iterations_before_stop"] = 1
        config["min_total_measured_nodes_before_stop"] = 1
        path = Path(tmp) / "input.json"
        atomic(path, config)
        with lock(runner.state):
            runner.initialize(path)
            runner.run()
            before = runner.journal.read()
            runner.run(resume=True)
            assert before == runner.journal.read(), "restart changed completed state"
            assert runner.status()["replay_trajectories"] == 100
            for event in runner.events("version"):
                for trajectory in event["trajectories"]:
                    visible = ["root"]
                    for step in trajectory["trace"]:
                        assert step["visible_before"] == visible
                        assert all(a in visible for a in step["planned_actions"])
                        assert not set(step["revealed"]) & set(visible)
                        visible.extend(step["revealed"])
            return runner.status()
