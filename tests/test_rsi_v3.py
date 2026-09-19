"""Batched orchestration contracts: synthetic only, no external services."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi.core import PLUGIN, observation, plan_batch_actions, replay
from rsi.evaluator import train_args
from rsi.policy import execute
from rsi.process import OperatorStop
from rsi.runner import ROOT, Runner, assert_no_model_artifacts, load_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO / "rsi/config.json")


def runner_at(tmp_path):
    runner = Runner(REPO, tmp_path / "state", synthetic=True)
    runner.initialize(REPO / "rsi/config.json")
    runner.verify()
    return runner


def nodes():
    return [ROOT] + [
        {"id": f"n{i}", "parent": "root", "status": "ok", "score": -i, "family_id": f"f{i}"}
        for i in range(1, 5)
    ]


def test_planner_prefix_root_target_unique_and_mutation():
    obs = observation(nodes(), 8, CONFIG)
    source = (REPO / "rsi/policies/initial.py").read_text()
    seen = []

    def select(o, seed):
        seen.append(json.loads(json.dumps(o)))
        result = execute(source, o, seed)
        o["nodes"].clear()
        return result

    plan = plan_batch_actions(select, obs, 42, 4, CONFIG)
    assert plan["modes"] == ["refinement", "refinement", "novel", "novel"]
    assert plan["planned_actions"][:2] == ["n1", "n2"]
    assert plan["planned_actions"][2:] == ["root", "root"]
    assert all(o["nodes"] == obs["nodes"] for o in seen)
    assert [o["planned_actions"] for o in seen] == [plan["planned_actions"][:i] for i in range(4)]
    duplicate = plan_batch_actions(lambda o, s: "n1", obs, 0, 4, CONFIG)
    assert duplicate["planned_actions"][0] == "n1"
    assert duplicate["planned_actions"][1] != "n1"
    assert duplicate["planned_actions"][2:] == ["root", "root"]
    assert duplicate["constrained_actions"] >= 3
    stopped = plan_batch_actions(lambda o, s: None, obs, 0, 4, CONFIG)
    assert stopped["planned_actions"][2:] == ["root", "root"]


def test_parallel_proposals_collision_reset_histories_and_eval_barrier(tmp_path, monkeypatch):
    runner = runner_at(tmp_path)
    original = runner.discover
    barrier = threading.Barrier(4)
    prompts, finished, evals, active = {}, [], [], []
    lock = threading.Lock()

    def discover(outer, parent, identifier, retry, workspace, directory, feedback):
        assert len(runner.events("attempt")) == 4
        assert not (workspace / PLUGIN / "retry_marker.py").exists()
        prompts[identifier, retry] = runner.discovery_prompt(outer, parent, feedback)
        if retry == 1:
            barrier.wait(timeout=10)
        # Slot 1 deliberately implements exactly slot 0's first proposal.
        alias = "n0001" if identifier == "n0002" and retry == 1 else identifier
        p = original(outer, parent, alias, retry, workspace, directory, feedback)
        if retry > 1:
            assert identifier == "n0002"
            assert feedback[0]["reason"].startswith("batch-internal structural collision;")
            assert "synthetic n0003" not in prompts[identifier, retry]
        (workspace / PLUGIN / "retry_marker.py").write_text("# reset me\n")
        with lock:
            finished.append((identifier, retry))
        return p

    def evaluate(workspace, output, logs, **kwargs):
        assert len(finished) == 5
        assert runner.events("batch_proposals_done")
        with lock:
            active.append(1)
            assert len(active) == 1
            evals.append(workspace.parent.name)
        time.sleep(0.01)  # A concurrent evaluator would overlap this guarded interval.
        with lock:
            active.pop()
        return {"score": -0.5}

    monkeypatch.setattr(runner, "discover", discover)
    monkeypatch.setattr(runner, "evaluate_workspace", evaluate)
    runner.synthetic = False

    # Use the original synthetic implementation while mocking only evaluator dispatch.
    def synthetic_discover(*args):
        # Original method reads this flag; avoid toggling shared state across threads.
        proxy = SimpleNamespace(synthetic=True)
        return Runner.discover(proxy, *args)

    original = synthetic_discover
    runner.execute_batch(0, {"planned_actions": ["root"] * 4, "substitutions": []})
    assert evals == ["n0001", "n0002", "n0003", "n0004"]
    assert [(e["batch_slot"], e["retry"]) for e in runner.events("proposal_session")] == [
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (1, 2),
    ]
    histories = []
    for prompt in prompts.values():
        payload = json.loads(prompt[len((REPO / "rsi/prompts/discovery.txt").read_text()) :])
        histories.append(json.dumps([payload["global_history"], payload["current_tree"]]))
    assert len(set(histories)) == 1
    assert runner.events("batch_done")[-1]["reason"] == "complete"
    assert all(e["node"]["batch_slot"] == i for i, e in enumerate(runner.events("outcome")))


def test_replay_batch_causality_and_root_order():
    tree = nodes()
    seen = []

    def select(obs, seed):
        seen.append(obs)
        return "root"

    result = replay(tree, select, 0, 8, 0.0025, -1000, 0.001, CONFIG)
    assert all(o["nodes"] == [ROOT] for o in seen)
    assert result["trace"][0]["revealed"] == ["n1", "n2", "n3", "n4"]
    changed = [ROOT] + [dict(n, score=1000) for n in tree[1:]]
    seen.clear()
    replay(changed, select, 0, 8, 0.0025, -1000, 0.001, CONFIG)
    assert all(o["nodes"] == [ROOT] for o in seen)


def test_recover_batch_never_reruns(tmp_path, monkeypatch):
    runner = runner_at(tmp_path)
    runner.journal.append("batch_started", batch_id="b0001", outer=0, planned_actions=["root"] * 4)
    for i in range(4):
        runner.journal.append(
            "attempt",
            attempt=f"n{i + 1:04d}",
            outer=0,
            parent="root",
            batch_id="b0001",
            batch_slot=i,
        )

    def forbidden(*a, **k):
        raise AssertionError("external work relaunched")

    monkeypatch.setattr(runner, "discover", forbidden)
    monkeypatch.setattr(runner, "evaluate_workspace", forbidden)
    runner.recover()
    before = runner.events()
    runner.recover()
    assert before == runner.events()
    assert len(runner.events("outcome")) == 4
    assert runner.events("batch_done")[0]["reason"] == "interrupted"


@pytest.mark.parametrize(
    "filename", ["model.safetensors", "weights.pt", "model.pth", "__0_0.distcp", ".metadata"]
)
def test_artifact_guard(tmp_path, filename):
    (tmp_path / "metrics.json").write_text("{}")
    assert_no_model_artifacts(tmp_path)
    bad = tmp_path / filename
    bad.write_bytes(b"heavy")
    with pytest.raises(ValueError, match="heavyweight"):
        assert_no_model_artifacts(tmp_path)
    assert bad.exists()


def test_disk_validation_and_stop(tmp_path, monkeypatch):
    runner = runner_at(tmp_path)
    monkeypatch.setattr(
        "rsi.runner.shutil.disk_usage", lambda p: SimpleNamespace(free=19 * 1024**3)
    )
    with pytest.raises(InterruptedError, match="free disk"):
        runner.execute_batch(0, {"planned_actions": ["root"], "substitutions": []})
    assert not runner.events("batch_started")
    assert runner.stop.exists()
    for key, value in [
        ("min_free_disk_gib", 0),
        ("physical_evaluator_workers", 2),
        ("proposal_batch_size", 0),
        ("proposal_parallel_workers", 0),
    ]:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(dict(CONFIG, **{key: value})))
        with pytest.raises(ValueError):
            load_config(path)


def test_checkpoint_contract():
    assert "--save_checkpoint=false" in train_args(CONFIG)
    assert "--save_checkpoint=true" in train_args(CONFIG, save_checkpoint=True)


def test_online_batch_boundary(tmp_path, monkeypatch):
    runner = runner_at(tmp_path)
    source = (REPO / "rsi/policies/initial.py").read_text()

    def select(obs, seed):
        if runner.events("batch_started"):
            assert len(runner.events("batch_started")) == len(runner.events("batch_done"))
        return execute(source, obs, seed)

    monkeypatch.setattr(runner, "policy", lambda name: select)
    # Replay is covered independently; keep this test focused on online boundaries.
    monkeypatch.setattr(runner, "improve", lambda *a: None)
    runner.run()
    assert len(runner.events("batch_done")) == 2
    assert len(runner.events("attempt")) == 8


def test_confirmation_keeps_checkpoint_and_completed_history_import(tmp_path):
    from rsi.confirmation import plan

    runner = runner_at(tmp_path)
    runner.attempt(0, "root")
    # Confirmation is a plan only; emulate real metadata without executing training.
    runner.synthetic = False
    runner.verify = lambda: None
    confirmation = plan(runner)
    for arm in confirmation["arms"].values():
        assert "--save_checkpoint=true" in arm["train_argv"]
        assert any("checkpoints/last/pretrained_model" in a for a in arm["eval_argv"])
    imported = Runner(REPO, tmp_path / "imported", synthetic=True)
    import_config = dict(CONFIG, continuation_anchor_ids=["n0001"])
    import_path = tmp_path / "import-config.json"
    import_path.write_text(json.dumps(import_config))
    with pytest.raises(ValueError, match="finished or exhausted"):
        imported.initialize(import_path, runner.state)
    runner.journal.append("finished", reason="synthetic test")
    imported.initialize(import_path, runner.state)
    imported.verify()
    assert imported.ledger() == runner.ledger()
    assert len(imported.root_references()) == len(runner.root_references())
    assert imported.anchor_metadata("n0001")["status"] == "ok"
    assert not (imported.state / "attempts").exists()


def test_no_checkpoint_or_dependency_copy(tmp_path):
    from rsi.core import manifest
    from rsi.runner import source_contract

    work = tmp_path / "candidate"
    source_contract(REPO, work)
    names = manifest(work)
    for name in names:
        assert not any(
            part
            in {
                ".venv",
                "models",
                "cache",
                "checkpoints",
                "studies",
                ".git",
                "README.md",
                "AGENT_BOARD.jsonl",
            }
            for part in Path(name).parts
        )


def test_interrupted_baseline_not_repeated(tmp_path, monkeypatch):
    runner = runner_at(tmp_path)
    runner.journal.append("baseline_started")

    def forbidden(*a):
        raise AssertionError("baseline repeated")

    monkeypatch.setattr(runner, "evaluate_workspace", forbidden)
    with pytest.raises(InterruptedError, match="never relaunch"):
        runner.ensure_baseline()


def test_proposal_interruption_never_retried(tmp_path, monkeypatch):
    runner = runner_at(tmp_path)
    calls = []

    def interrupted(*args):
        calls.append(args)
        raise OperatorStop("operator stopped session")

    monkeypatch.setattr(runner, "discover", interrupted)
    runner.attempt(0, "root")
    assert len(calls) == 1
    assert runner.events("outcome")[0]["node"]["status"] == "interrupted"
    assert runner.events("batch_done")[0]["reason"] == "interrupted"
    runner.recover()
    assert len(calls) == 1
