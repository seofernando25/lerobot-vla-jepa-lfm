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
from rsi.process import codex_argv, isolated, run_logged

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


def load_config(path):
    c = json.loads(Path(path).read_text())
    if c.get("protocol_version") != "dream-rsi-v3-batched":
        raise ValueError(
            "v3 requires a fresh dream-rsi-v3-batched study; old state cannot be reused"
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
        "K1",
        "K2",
        "max_outer_iterations",
        "max_real_attempts",
        "max_eval_samples",
        "batch_size",
        "agent_timeout_seconds",
        "probe_timeout_seconds",
        "policy_timeout_seconds",
    ):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if c["physical_evaluator_workers"] != 1:
        raise ValueError("physical_evaluator_workers must equal 1")
    if c["W"] != 1 or c["policy_versions"] != 4 or c["replay_trajectories"] != 100:
        raise ValueError("fixed contract requires W=1, four version slots, 100 trajectories")
    if c["precision"] != "bf16" or not 0 < c["dataset"]["eval_split"] < 1:
        raise ValueError("invalid precision or held-out split")
    if c["probe_steps"] != 500:
        raise ValueError("fixed probe budget is 500 steps")
    if not 0 <= c["max_root_arch_similarity"] <= 1:
        raise ValueError("invalid root similarity threshold")
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
        prior_history, prior_roots = [], []
        if prior_study is not None:
            prior_path = Path(prior_study).resolve()
            if not (prior_path / "events").is_dir():
                raise ValueError("prior study journal missing")
            prior = Journal(prior_path).read()
            if not any(e["kind"] == "finished" for e in prior):
                raise ValueError("prior study must be completed; live studies cannot be imported")
            version = next(
                e["config"]["protocol_version"] for e in prior if e["kind"] == "initialized"
            )
            if version not in {"dream-rsi-v2", "dream-rsi-v3-batched"}:
                raise ValueError("only v2/v3 mechanism history may be imported")
            prior_init = next(e for e in prior if e["kind"] == "initialized")
            prior_history = list(prior_init.get("prior_history", [])) + [
                compact_node(e["node"])
                for e in prior
                if e["kind"] == "outcome" and e["node"].get("proposal")
            ]
            prior_roots = list(prior_init.get("prior_roots", [])) + [
                {k: e[k] for k in ("architecture_delta", "axes_signature")}
                for e in prior
                if e["kind"] == "proposal_accepted" and e["parent"] == "root"
            ]
        source_contract(self.repo, self.state / "base")
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
        if self.synthetic != init["synthetic"]:
            raise ValueError("synthetic/real state cannot be mixed")
        self.config = c
        return init

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

    def root_node(self):
        root = dict(ROOT)
        baseline = self.events("baseline")
        if baseline:
            event = baseline[-1]
            root.update(
                score=event["metrics"]["score"],
                status="ok",
                summary="Measured clean base LFM under the fixed probe evaluator.",
                metrics=event["metrics"],
            )
        return root

    def tree(self, outer):
        return [self.root_node()] + [
            e["node"]
            for e in self.events("outcome")
            if e["outer"] == outer and e["node"].get("proposal")
        ]

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
            raise InterruptedError("free disk below min_free_disk_gib")

    def evaluate_workspace(self, workspace, output, logs):
        self.check_disk()
        evaluator = self.repo / "rsi/evaluator.py"
        argv = isolated(
            [
                str(self.repo / ".venv/bin/python"),
                "/tmp/evaluator.py",
                *train_args(self.config),
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
            elapsed = run_logged(
                argv,
                self.repo,
                logs,
                self.config["probe_timeout_seconds"],
                self.stop,
            )
        finally:
            assert_no_model_artifacts(output)
        text = (logs / "stderr.log").read_text()
        text += (logs / "stdout.log").read_text()
        metrics = parse_metric(text, self.config["probe_steps"])
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
    ):
        metrics = metrics or {}
        node = {
            "batch_id": event.get("batch_id"),
            "batch_slot": event.get("batch_slot"),
            "id": event["attempt"],
            "parent": event["parent"],
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
            workspace_hash=workspace_hash,
            ended_at=now(),
        )

    def ledger(self):
        return self.events("initialized")[0].get("prior_history", []) + [
            compact_node(e["node"]) for e in self.events("outcome") if e["node"].get("proposal")
        ]

    def root_references(self):
        reference = json.loads((self.repo / self.config["novelty_reference"]).read_text())
        if reference["fingerprint_version"] != 1:
            raise ValueError("unsupported novelty fingerprint version")
        return (
            reference["roots"]
            + self.events("initialized")[0].get("prior_roots", [])
            + [e for e in self.events("proposal_accepted") if e["parent"] == "root"]
        )

    def discovery_prompt(self, outer, parent, feedback):
        # Never serialize config wholesale: it includes the private novelty-reference path.
        fixed = {
            k: self.config[k]
            for k in ("probe_steps", "seed", "batch_size", "precision", "max_eval_samples")
        }
        return (self.repo / "rsi/prompts/discovery.txt").read_text() + json.dumps(
            {
                "selected_parent": parent,
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
        directory = self.state / "attempts" / event["attempt"]
        workspace = directory / "workspace"
        proposed = None
        try:
            if self.stop.exists():
                raise InterruptedError("stop requested")
            parent_workspace = self.workspace(event["outer"], parent)
            if parent != "root":
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
                event["outer"], parent, event["attempt"], retry, workspace, session, feedback
            )
            proposed = parse_proposal(json.dumps(proposed))
            accepted = candidate_guard(workspace, before)
            delta = architecture_delta(parent_workspace, workspace)
            return {"proposal": proposed, "workspace_hash": accepted, "delta": delta}
        except Exception as exc:  # noqa: BLE001 - untrusted implementation boundary
            return {
                "error": str(exc),
                "proposal": proposed,
                "interrupted": isinstance(exc, InterruptedError),
            }

    def attempt(self, outer, parent):
        # Convenience for synthetic contract callers; production always plans a full batch.
        return self.execute_batch(outer, {"planned_actions": [parent], "substitutions": []})

    def execute_batch(self, outer, plan):
        self.check_disk()
        if self.config["physical_evaluator_workers"] != 1:
            raise ValueError("physical_evaluator_workers must equal 1")
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
                    k: event[k] for k in ("attempt", "outer", "parent", "batch_id", "batch_slot")
                }
                self.journal.append("proposal_session", **fields, retry=retry, at=now())
                proposed = result.get("proposal")
                reason = result.get("error")
                if result.get("interrupted"):
                    self.stop.touch()
                if reason is None:
                    try:
                        check_novelty(
                            proposed, result["delta"], event["parent"], references, self.config
                        )
                    except ValueError as exc:
                        reason = str(exc)
                    if reason is None and event["parent"] == "root":
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
                family = (
                    normalize(proposed["mechanism_family"])
                    if event["parent"] == "root"
                    else next(n["family_id"] for n in history[1] if n["id"] == event["parent"])
                )
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
                if event["parent"] == "root":
                    sibling_roots.append(record)
        self.journal.append("batch_proposals_done", batch_id=batch_id, outer=outer, at=now())
        for event in events:
            result = accepted.get(event["attempt"])
            if result is None:
                self.finish(
                    event,
                    "interrupted" if self.stop.exists() else "proposal_failure",
                    "Proposal sessions stopped or rejected; no probe.",
                )
                continue
            directory = self.state / "attempts" / event["attempt"]
            start = time.monotonic()
            proposal = result["proposal"]
            try:
                self.check_disk()
                if self.stop.exists():
                    raise InterruptedError("stop requested")
                if self.synthetic:
                    score = -1.0 / (1 + int(event["attempt"][1:]))
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
                        directory / "workspace", output, directory / "probe"
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
                )
            except Exception as exc:  # noqa: BLE001 - external evaluator boundary
                self.finish(
                    event,
                    "interrupted"
                    if isinstance(exc, InterruptedError)
                    else "implementation_failure",
                    str(exc),
                    proposal=proposal,
                    delta=result["delta"],
                    family_id=result["family_id"],
                )
                if isinstance(exc, InterruptedError):
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
            if any(e["outer"] == outer for e in self.events("cycle_done")):
                done = next(e for e in self.events("online_done") if e["outer"] == outer)
                if len(self.events("attempt")) >= self.config["max_real_attempts"]:
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
                while sum(e["outer"] == outer for e in self.events("attempt")) < self.config["K1"]:
                    if self.stop.exists():
                        return
                    if len(self.events("attempt")) >= self.config["max_real_attempts"]:
                        reason = "global_attempt_cap"
                        break
                    nodes = self.tree(outer)
                    obs = observation(
                        [compact_node(n) for n in nodes], self.config["K1"], self.config
                    )
                    self.check_disk()
                    size = min(
                        self.config["proposal_batch_size"],
                        self.config["K1"]
                        - sum(e["outer"] == outer for e in self.events("attempt")),
                        self.config["max_real_attempts"] - len(self.events("attempt")),
                    )
                    plan = plan_batch_actions(
                        policy,
                        obs,
                        self.config["seed"] + outer * 1009 + obs["round"],
                        size,
                        self.config,
                    )
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
            if len(self.events("attempt")) >= self.config["max_real_attempts"]:
                self.journal.append("finished", reason="global_attempt_cap", at=now())
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
        return {
            "initialized": bool(events),
            "synthetic": self.synthetic,
            "attempts_reserved": len(self.events("attempt")),
            "outcomes": len(self.events("outcome")),
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
        # Keep dry-run intentionally one-cycle while preserving real v3 coverage floors.
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
