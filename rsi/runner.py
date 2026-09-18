"""Single-worker, event-sourced online/offline controller."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from rsi.core import (
    PLUGIN,
    Journal,
    atomic,
    candidate_guard,
    digest,
    manifest,
    observation,
    replay,
    validate_action,
)
from rsi.evaluator import parse_metric, train_args
from rsi.policy import Policy
from rsi.process import codex_argv, isolated, run_logged

ROOT = {
    "id": "root",
    "parent": None,
    "score": None,
    "status": "root",
    "summary": "Clean base LFM source; unmeasured, not a zero-quality baseline.",
}


def now():
    return datetime.now(UTC).isoformat()


def load_config(path):
    c = json.loads(Path(path).read_text())
    for key in (
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
    if c["W"] != 1 or c["policy_versions"] != 4 or c["replay_trajectories"] != 100:
        raise ValueError("fixed contract requires W=1, four version slots, 100 trajectories")
    if c["precision"] != "bf16" or not 0 < c["dataset"]["eval_split"] < 1:
        raise ValueError("invalid precision or held-out split")
    if c["beta1"] < 0 or c["failure_score"] >= 0:
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
    shutil.copytree(
        repo / PLUGIN, target / PLUGIN, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    text = (repo / "pyproject.toml").read_text()
    text = "\n".join(line for line in text.splitlines() if not line.startswith("readme ="))
    (target / "pyproject.toml").write_text(text + "\n")
    if (repo / "uv.lock").exists():
        shutil.copyfile(repo / "uv.lock", target / "uv.lock")
    (target / "tests").mkdir()
    shutil.copyfile(repo / "rsi/contract_test.py", target / "tests/test_contract.py")
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

    def initialize(self, config_path):
        if self.events():
            raise ValueError("already initialized; use status or run --resume")
        config = load_config(config_path)
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
            e["node"] for e in self.events("outcome") if e["outer"] == outer
        ]

    def ensure_baseline(self):
        if self.events("baseline"):
            return
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

    def evaluate_workspace(self, workspace, output, logs):
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
        elapsed = run_logged(
            argv,
            self.repo,
            logs,
            self.config["probe_timeout_seconds"],
            self.stop,
        )
        text = (logs / "stderr.log").read_text()
        text += (logs / "stdout.log").read_text()
        metrics = parse_metric(text, self.config["probe_steps"])
        metrics.update(
            wall_seconds=elapsed,
            hardware=json.loads((output / "hardware.json").read_text()),
            eval_samples=json.loads((output / "eval_samples.json").read_text()),
            trainability=json.loads((output / "trainability.json").read_text()),
        )
        return metrics

    def workspace(self, outer, node):
        return (
            self.state / "base" if node == "root" else self.state / "attempts" / node / "workspace"
        )

    def recover(self):
        finished = {e["attempt"] for e in self.events("outcome")}
        for event in self.events("attempt"):
            if event["attempt"] not in finished:
                self.finish(event, "interrupted", "Reserved attempt interrupted; never relaunched.")

    def finish(self, event, status, summary, metrics=None, workspace_hash=None):
        metrics = metrics or {}
        node = {
            "id": event["attempt"],
            "parent": event["parent"],
            "status": status,
            "score": metrics.get("score", self.config["failure_score"]),
            "summary": summary,
            "metrics": metrics,
        }
        self.journal.append(
            "outcome",
            outer=event["outer"],
            attempt=event["attempt"],
            node=node,
            workspace_hash=workspace_hash,
            ended_at=now(),
        )

    def attempt(self, outer, parent):
        identifier = f"n{len(self.events('attempt')) + 1:04d}"
        event = self.journal.append(
            "attempt",
            attempt=identifier,
            outer=outer,
            parent=parent,
            started_at=now(),
            config_hash=digest(self.config),
            seed=self.config["seed"],
            steps=self.config["probe_steps"],
        )
        directory = self.state / "attempts" / identifier
        directory.mkdir(parents=True)
        workspace = directory / "workspace"
        start = time.monotonic()
        try:
            if parent != "root":
                parent_event = next(e for e in self.events("outcome") if e["attempt"] == parent)
                if parent_event["workspace_hash"] is None:
                    raise ValueError("parent has no accepted saved workspace; select root instead")
                if manifest(self.workspace(outer, parent)) != parent_event["workspace_hash"]:
                    raise ValueError("parent workspace modified")
            shutil.copytree(self.workspace(outer, parent), workspace)
            before = manifest(workspace)
            if self.synthetic:
                metrics = {
                    "score": -1.0 / (1 + int(identifier[1:])),
                    "eval_loss": 1.0 / (1 + int(identifier[1:])),
                    "optimizer_steps": self.config["probe_steps"],
                    "wall_seconds": 0.0,
                    "metric_provenance": "synthetic; no training",
                }
                summary = "Synthetic architecture result, never a scientific measurement."
            else:
                prompt = (self.repo / "rsi/prompts/discovery.txt").read_text()
                prompt += json.dumps(
                    {
                        "selected_parent": parent,
                        "observed_nodes": self.tree(outer),
                        "fixed_probe": self.config,
                    },
                    indent=2,
                )
                (directory / "prompt.txt").write_text(prompt)
                discovery_args = codex_argv()
                discovery_args[-1:-1] = ["-o", "/tmp/work/.discovery_summary"]
                argv = isolated(discovery_args, self.repo, workspace)
                run_logged(
                    argv,
                    self.repo,
                    directory / "discovery",
                    self.config["agent_timeout_seconds"],
                    self.stop,
                    prompt,
                )
                summary_path = workspace / ".discovery_summary"
                if not summary_path.is_file():
                    raise ValueError("discovery agent did not emit a final summary")
                summary = summary_path.read_text()[-8000:]
                summary_path.unlink()
                candidate_guard(workspace, before)
                output = directory / "evaluation"
                output.mkdir()
                metrics = self.evaluate_workspace(
                    workspace,
                    output,
                    directory / "probe",
                )
            accepted = candidate_guard(workspace, before)
            metrics["attempt_wall_seconds"] = time.monotonic() - start
            atomic(directory / "metrics.json", metrics)
            self.finish(event, "ok", summary, metrics, accepted)
        except Exception as exc:  # noqa: BLE001 - external candidate failure boundary
            self.finish(
                event,
                "implementation_failure",
                str(exc),
                {"attempt_wall_seconds": time.monotonic() - start},
            )
            if isinstance(exc, InterruptedError):
                self.stop.touch()

    def improve(self, outer, incumbent):
        cycles = [e for e in self.events("cycle") if e["outer"] == outer]
        if not cycles:
            history = [self.tree(e["outer"]) for e in self.events("online_done")]
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
                                    "feedback": slots,
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
                            # Different valid branch-opening schedules in fake worlds.
                            text = path.read_text().replace(
                                "len(nodes) < 4", f"len(nodes) < {slot + 1}"
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
                    )
                except Exception as exc:  # noqa: BLE001 - external candidate failure boundary
                    result = {
                        "objective": self.config["failure_score"],
                        "error": str(exc),
                        "trace": [],
                        "revealed": 0,
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
                if done["reason"] == "policy_STOP" or (
                    len(self.events("attempt")) >= self.config["max_real_attempts"]
                ):
                    self.journal.append("finished", reason=done["reason"], at=now())
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
                while len(self.tree(outer)) - 1 < self.config["K1"]:
                    if self.stop.exists():
                        return
                    if len(self.events("attempt")) >= self.config["max_real_attempts"]:
                        reason = "global_attempt_cap"
                        break
                    nodes = self.tree(outer)
                    obs = observation(nodes, self.config["K1"])
                    action = policy(obs, self.config["seed"] + outer * 1009 + obs["round"])
                    validate_action(action, obs)
                    if action is None:
                        reason = "policy_STOP"
                        break
                    self.attempt(outer, action)
                self.journal.append("online_done", outer=outer, reason=reason, at=now())
            if self.stop.exists():
                return
            selected = self.improve(outer, current)
            if selected is None:
                return
            if (
                reason == "policy_STOP"
                or len(self.events("attempt")) >= self.config["max_real_attempts"]
            ):
                reason = "policy_STOP" if reason == "policy_STOP" else "global_attempt_cap"
                self.journal.append("finished", reason=reason, at=now())
                return
        self.journal.append("finished", reason="global_outer_cap", at=now())

    def status(self):
        events = self.events()
        return {
            "initialized": bool(events),
            "synthetic": self.synthetic,
            "attempts_reserved": len(self.events("attempt")),
            "outcomes": len(self.events("outcome")),
            "completed_cycles": len(self.events("cycle_done")),
            "replay_trajectories": sum(e["trajectories"] for e in self.events("cycle_done")),
            "stop_requested": self.stop.exists(),
            "finished": self.events("finished")[-1]["reason"] if self.events("finished") else None,
        }


def dry_run(repo):
    with tempfile.TemporaryDirectory(prefix="rsi-synthetic-") as tmp:
        runner = Runner(repo, Path(tmp), synthetic=True)
        config = json.loads((Path(repo) / "rsi/config.json").read_text())
        config["max_outer_iterations"] = 1
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
                        assert step["action"] in visible
                        assert step["revealed"] not in visible
                        visible.append(step["revealed"])
            return runner.status()
