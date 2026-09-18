"""Fast synthetic scientific-contract tests: never call Codex or load a GPU model."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rsi.core import Journal, atomic, candidate_guard, manifest, observation, replay, validate_tree
from rsi.evaluator import parse_metric, train_args
from rsi.policy import Policy, execute, validate_source
from rsi.process import codex_argv, isolated, run_logged
from rsi.runner import ROOT, Runner, dry_run, lock, source_contract

REPO = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPO / "rsi/config.json").read_text())


def node(name, parent, score):
    return {"id": name, "parent": parent, "score": score, "status": "ok", "summary": name}


TREE = [ROOT, node("a", "root", -4), node("b", "root", -3), node("c", "a", -1), node("d", "b", -2)]


def test_prefix_causality_root_order_and_chain():
    seen = []
    actions = iter(["root", "a", "root", "b"])

    def select(obs, seed):
        seen.append([n["id"] for n in obs["nodes"]])
        assert "children" not in obs
        return next(actions)

    result = replay(TREE, select, 42, 6, 0.1, -1000)
    assert seen == [["root"], ["root", "a"], ["root", "a", "c"], ["root", "a", "c", "b"]]
    assert result["objective"] == -1.4
    assert result["reason"] == "full_tree"


def test_unselected_future_cannot_affect_policy():
    def select(obs, seed):
        assert [n["id"] for n in obs["nodes"]] in (["root"], ["root", "a"])
        return "root" if len(obs["nodes"]) == 1 else None

    modified = TREE[:2] + [node("b", "root", 999), node("c", "a", 999)]
    assert replay(TREE, select, 2, 6, 0.1, -1000) == replay(modified, select, 2, 6, 0.1, -1000)


@pytest.mark.parametrize(
    "actions,reason,count",
    [
        ([None], "STOP", 0),
        (["root", "a", "c"], "empty_action", 2),
        (["root"] * 4, "empty_action", 2),
    ],
)
def test_replay_termination(actions, reason, count):
    actions = iter(actions)
    result = replay(TREE, lambda obs, seed: next(actions), 42, 6, 0.01, -1000)
    assert result["reason"] == reason
    assert result["revealed"] == count


def test_k2_and_invalid_action():
    assert replay(TREE, lambda obs, seed: "root", 0, 1, 0.1, -1000)["revealed"] == 1
    with pytest.raises(ValueError, match="invalid parent"):
        replay(TREE, lambda obs, seed: "c", 0, 6, 0.1, -1000)
    with pytest.raises(ValueError, match="chain"):
        validate_tree(TREE + [node("e", "a", 0)])
    with pytest.raises(ValueError, match="noncausal"):
        validate_tree([ROOT, node("a", "future", 0)])


def test_policy_seeded_determinism_and_interface(tmp_path):
    source = (REPO / "rsi/policies/initial.py").read_text()
    obs = observation(
        [ROOT, node("a", "root", -1), node("b", "root", -1), node("c", "root", -1)], 6
    )
    assert execute(source, obs, 42) == execute(source, obs, 42)
    assert len({execute(source, obs, seed) for seed in range(20)}) > 1
    path = tmp_path / "policy.py"
    path.write_text("def select(observation, seed):\n    return 'future'\n")
    with pytest.raises(subprocess.CalledProcessError):
        Policy(path)
    path.write_text("def select(observation, seed):\n    while True: pass\n")
    with pytest.raises(subprocess.TimeoutExpired):
        Policy(path, timeout=0.1)
    for source in (
        "import os\ndef select(o,s): return None",
        "def select(o,s): return o.__class__",
    ):
        with pytest.raises(ValueError):
            validate_source(source)


def test_journal_atomic_hash_chain_and_torn_temp(tmp_path):
    journal = Journal(tmp_path)
    journal.append("a", value=1)
    journal.append("b", value=2)
    (journal.root / ".pending-crash").write_text('{"partial":')
    assert len(journal.read()) == 2
    path = journal.root / "00000000.json"
    record = json.loads(path.read_text())
    record["value"] = 99
    atomic(path, record)
    with pytest.raises(ValueError, match="hash chain"):
        journal.read()


def test_workspace_allowlist_and_guards(tmp_path):
    workspace = tmp_path / "work"
    source_contract(REPO, workspace)
    before = manifest(workspace)
    assert not any(Path(n).name in {"README.md", "AGENTS.md", ".git"} for n in before)
    assert all(
        n.startswith("src/")
        or n in {"pyproject.toml", "uv.lock", "tests/test_contract.py", "proposal_schema.json"}
        for n in before
    )
    new = workspace / "src/lerobot_policy_vla_jepa_lfm/new_arch.py"
    new.write_text("class Fusion: pass\n")
    candidate_guard(workspace, before)
    for cls, method in [
        ("VLAJEPALFMModel", "forward"),
        ("VLAJEPALFMModel", "_action_loss"),
        ("VLAJEPALFMModel", "_world_model_loss"),
        ("VLAJEPALFMModel", "predict_action"),
        ("VLAJEPALFMPolicy", "forward"),
        ("VLAJEPALFMPolicy", "predict_action_chunk"),
        ("VLAJEPALFMPolicy", "select_action"),
    ]:
        new.write_text(f"class {cls}:\n    def {method}(self): pass\n")
        with pytest.raises(ValueError, match="protected override"):
            candidate_guard(workspace, before)
    new.unlink()
    (workspace / "README.md").write_text("leak")
    with pytest.raises(ValueError, match="forbidden file"):
        candidate_guard(workspace, before)
    (workspace / "README.md").unlink()
    new.symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlink"):
        candidate_guard(workspace, before)


def test_fixed_codex_and_evaluator_contract():
    argv = codex_argv()
    assert "--ephemeral" in argv and "gpt-6-astra" in argv
    assert 'model_reasoning_effort="low"' in argv
    assert "resume" not in argv and "fork" not in argv
    args = train_args(CONFIG)
    for flag in (
        "--steps=500",
        "--eval_steps=500",
        "--dataset.eval_split=0.1",
        "--max_eval_samples=64",
        "--batch_size=2",
        "--seed=42",
        "--use_policy_training_preset=false",
        "--scheduler.type=cosine_decay_with_warmup",
    ):
        assert flag in args
    assert not any("policy.path" in a for a in args)
    assert parse_metric("step 500: eval_loss=1.2500", 500)["score"] == -1.25
    for text in (
        "step 499: eval_loss=1.0",
        "step 500: eval_loss=nan",
        "step 500: eval_loss=1.0\nstep 500: eval_loss=2.0",
    ):
        with pytest.raises(ValueError):
            parse_metric(text, 500)


def make_runner(tmp_path, **overrides):
    config = dict(
        CONFIG,
        max_outer_iterations=1,
        min_outer_iterations_before_stop=1,
        min_total_measured_nodes_before_stop=1,
        **overrides,
    )
    path = tmp_path / "input.json"
    atomic(path, config)
    runner = Runner(REPO, tmp_path / "state", synthetic=True)
    runner.initialize(path)
    return runner


def test_restart_pending_consumes_budget_once(tmp_path):
    runner = make_runner(tmp_path, max_real_attempts=1)
    runner.verify()
    runner.journal.append("attempt", attempt="n0001", outer=0, parent="root", started_at="fake")
    runner.journal.append("started")
    with pytest.raises(ValueError, match="resume"):
        runner.run()
    runner.run(resume=True)
    assert len(runner.events("attempt")) == 1
    assert runner.events("outcome")[0]["node"]["status"] == "interrupted"
    assert runner.status()["replay_trajectories"] == 100
    assert runner.status()["finished"] == "global_attempt_cap"


def test_offline_resume_interrupted_revision_and_selection(tmp_path):
    runner = make_runner(tmp_path)
    runner.verify()
    runner.journal.append("online_done", outer=0, reason="K1")
    runner.journal.append("revision", outer=0, slot=1)
    selected = runner.improve(0, "p0")
    versions = runner.events("version")
    assert len(versions) == 4
    assert sum(len(v["trajectories"]) for v in versions) == 100
    assert versions[1]["fallback_reason"].startswith("interrupted")
    assert selected == "p0"  # All root-only scores tied: incumbent wins.
    worlds = [[(t["history"], t["seed"]) for t in v["trajectories"]] for v in versions]
    assert all(w == worlds[0] for w in worlds)


def test_frozen_state_and_exclusive_runner(tmp_path):
    runner = make_runner(tmp_path)
    with (
        lock(runner.state),
        pytest.raises(RuntimeError, match="another runner"),
        lock(runner.state),
    ):
        pass
    runner.stop.touch()
    with pytest.raises(ValueError, match="resume"):
        runner.run()
    config = json.loads((runner.state / "config.json").read_text())
    config["K1"] += 1
    atomic(runner.state / "config.json", config)
    with pytest.raises(ValueError, match="frozen"):
        runner.verify()


def test_safe_process_stop_and_logs(tmp_path):
    stop = tmp_path / "STOP"
    stop.touch()
    with pytest.raises(InterruptedError):
        run_logged(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            tmp_path,
            tmp_path / "logs",
            20,
            stop,
        )
    assert (tmp_path / "logs/command.json").exists()
    assert (tmp_path / "logs/stdout.log").exists()
    assert (tmp_path / "logs/stderr.log").exists()


def test_synthetic_end_to_end_never_invokes_codex_or_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic mode attempted real work")

    monkeypatch.setattr("rsi.runner.run_logged", forbidden)
    result = dry_run(REPO)
    assert result["replay_trajectories"] == 100
    assert result["attempts_reserved"] == 8


def test_isolation_command_hides_repo(tmp_path):
    command = isolated([sys.executable, "-c", "pass"], REPO, tmp_path, evaluator=True)
    assert command[0] == "bwrap"
    assert ["--tmpfs", str(REPO.parent)] == command[
        command.index(str(REPO.parent)) - 1 : command.index(str(REPO.parent)) + 1
    ]
    assert "/tmp/work" in command
    assert str(REPO) not in command
    assert "--ro-bind" in command
