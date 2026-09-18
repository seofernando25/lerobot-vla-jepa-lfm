"""V2 diversity contracts retained under V3 batching. No archive, Codex, GPU or network required."""

import json
from pathlib import Path

import pytest

from rsi.core import PLUGIN, candidate_guard, constrain_action, manifest, observation, replay
from rsi.novelty import (
    AXES,
    architecture_delta,
    architecture_fingerprint,
    architecture_similarity,
    axes_signature,
    check_novelty,
    parse_proposal,
)
from rsi.runner import ROOT, Runner, policy_feedback, source_contract

REPO = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPO / "rsi/config.json").read_text())


def proposal(label="distinct", parent="root"):
    return {
        "hypothesis": "Test structural hypothesis",
        "mechanism_family": label,
        "architecture_axes": dict.fromkeys(AXES, label),
        "relation_to_parent": "new_family" if parent == "root" else "structural_refinement",
        "structural_change": "Adds a distinct structural module",
        "why_not_parameter_only": "Adds call topology",
        "expected_effect": "Synthetic only",
        "implementation_checks": ["Syntax checked"],
        "limitations": "Synthetic only",
    }


def runner_at(tmp_path):
    runner = Runner(REPO, tmp_path / "state", synthetic=True)
    runner.initialize(REPO / "rsi/config.json")
    runner.verify()
    return runner


def plugin(tmp_path, name, text):
    root = tmp_path / name
    directory = root / PLUGIN
    directory.mkdir(parents=True)
    (directory / "module.py").write_text(text)
    return root


def test_proposal_schema_strict():
    p = proposal()
    assert parse_proposal(json.dumps(p)) == p
    for bad in [
        dict(p, extra="no"),
        dict(p, architecture_axes={}),
        dict(p, relation_to_parent="refine"),
        dict(p, hypothesis=" "),
        dict(p, implementation_checks="unchecked"),
        dict(p, limitations=42),
    ]:
        with pytest.raises(ValueError):
            parse_proposal(json.dumps(bad))
    with pytest.raises(ValueError, match="duplicate"):
        parse_proposal('{"hypothesis":"a","hypothesis":"b"}')
    with pytest.raises(ValueError):
        parse_proposal("```json\n" + json.dumps(p) + "\n```")


def test_ast_ignores_parameters_and_activation(tmp_path):
    text = """class Bridge:
    def __init__(self):
        self.proj = nn.Linear(64, 256)
        self.act = nn.GELU()
        nn.init.normal_(self.proj.weight, std=0.01)
    def forward(self, x):
        return self.proj(x)
"""
    base = plugin(tmp_path, "base", text)
    candidate = plugin(
        tmp_path,
        "candidate",
        text.replace("256", "512").replace("0.01", "0.1").replace("GELU", "SiLU"),
    )
    assert architecture_delta(base, candidate) == set()
    with pytest.raises(ValueError, match="parameter-only"):
        check_novelty(proposal(parent="n1"), set(), "n1", [], CONFIG)
    candidate.joinpath(PLUGIN, "module.py").write_text(text.replace("nn.GELU()", "nn.ReLU()"))
    assert architecture_fingerprint(base) == architecture_fingerprint(candidate)


def test_ast_detects_topology_and_module(tmp_path):
    base = plugin(tmp_path, "base", "def encode(x): return pool(route(x))\n")
    candidate = plugin(tmp_path, "candidate", "def encode(x): return route(pool(x))\n")
    assert len(architecture_delta(base, candidate)) == 2
    (candidate / PLUGIN / "new.py").write_text(
        "class Fusion:\n    def encode(self, x): return attend(mix(x))\n"
    )
    delta = architecture_delta(base, candidate)
    assert len(delta) >= 4
    check_novelty(proposal(parent="n1"), delta, "n1", [], CONFIG)
    assert architecture_similarity(delta, delta) == 1
    assert architecture_similarity(delta, {"unrelated"}) == 0


def test_tracked_reference_self_contained():
    reference = json.loads((REPO / CONFIG["novelty_reference"]).read_text())
    roots = reference["roots"]
    assert {r["id"] for r in roots} == {"n0001", "n0002", "n0003", "n0007"}
    assert all(
        set(r) == {"id", "mechanism_family", "axes_signature", "architecture_delta"} for r in roots
    )
    for root in roots:
        with pytest.raises(ValueError, match="overlaps"):
            check_novelty(proposal(), root["architecture_delta"], "root", roots, CONFIG)
    check_novelty(proposal(), {f"new:{i}" for i in range(6)}, "root", roots, CONFIG)
    p = proposal()
    with pytest.raises(ValueError, match="signature"):
        check_novelty(
            p,
            {f"new:{i}" for i in range(6)},
            "root",
            [{"axes_signature": axes_signature(p["architecture_axes"]), "architecture_delta": []}],
            CONFIG,
        )
    assert axes_signature(dict.fromkeys(AXES, " A-B  ")) == axes_signature(
        dict.fromkeys(AXES, "a b")
    )


def test_schema_and_reference_isolation(tmp_path):
    workspace = tmp_path / "work"
    source_contract(REPO, workspace)
    before = manifest(workspace)
    assert not any(
        any(x in n for x in ("rsi/", "studies", "README", "AGENT", ".git", "v1_")) for n in before
    )
    assert set(before) - {n for n in before if n.startswith(str(PLUGIN))} == {
        "pyproject.toml",
        "uv.lock",
        "tests/test_contract.py",
        "proposal_schema.json",
    }
    (workspace / "proposal_schema.json").write_text("{}")
    with pytest.raises(ValueError, match="forbidden"):
        candidate_guard(workspace, before)


def test_duplicate_retries_reset_parent_and_never_probe(tmp_path, monkeypatch):
    runner = runner_at(tmp_path)
    original = runner.discover
    seen = []

    def discover(outer, parent, identifier, retry, workspace, directory, feedback):
        assert not (workspace / PLUGIN / "rejected.py").exists()
        seen.append((retry, list(feedback)))
        p = original(outer, parent, identifier, retry, workspace, directory, feedback)
        (workspace / PLUGIN / "rejected.py").write_text("class Rejected: pass\n")
        return p

    reference = {
        "axes_signature": axes_signature(proposal("n0001_1")["architecture_axes"]),
        "architecture_delta": [],
    }
    monkeypatch.setattr(runner, "root_references", lambda: [reference])
    monkeypatch.setattr(runner, "discover", discover)
    runner.attempt(0, "root")
    assert [n for n, _ in seen] == [1, 2]
    assert seen[1][1][0]["reason"] == "duplicate root architecture_axes signature"
    assert seen[1][1][0]["rejected_mechanism"]["mechanism_family"] == "synthetic n0001"
    assert runner.status()["proposal_sessions"] == 2
    assert runner.status()["proposal_rejections"] == 1
    assert runner.status()["accepted_measured_nodes"] == 1
    assert len(runner.events("attempt")) == 1
    assert not any("n0001_1.py" in p for p in manifest(runner.workspace(0, "n0001")))

    def reject(*args, **kwargs):
        raise ValueError("duplicate structure")

    monkeypatch.setattr(runner, "discover", reject)
    runner.attempt(0, "root")
    assert runner.events("outcome")[-1]["node"]["status"] == "proposal_failure"
    assert not (runner.state / "attempts/n0002/metrics.json").exists()
    assert len(runner.tree(0)) == 2
    assert runner.status()["proposal_sessions"] == 5


def test_global_ledger_and_compact_cycle(tmp_path):
    runner = runner_at(tmp_path)
    runner.attempt(0, "root")
    runner.attempt(1, "root")
    runner.attempt(1, "n0002")
    ledger = runner.ledger()
    assert [n["outer"] for n in ledger] == [0, 1, 1]
    assert ledger[1]["family_id"] == ledger[2]["family_id"]
    assert ledger[1]["mechanism_family"] != ledger[2]["mechanism_family"]
    prompt = runner.discovery_prompt(1, "root", [{"reason": "retry"}])
    assert '"n0001"' in prompt and '"n0003"' in prompt
    for forbidden in (
        "trainability.json",
        "optimizer_steps",
        "hardware",
        "v1_",
        "baseline_eval_loss",
    ):
        assert forbidden not in prompt
    assert "metrics" not in json.dumps(ledger)
    assert all("failure_reason" in node for node in ledger)
    runner.journal.append("online_done", outer=0, reason="K1")
    runner.improve(0, "p0")
    cycle = runner.events("cycle")[0]
    assert "metrics" not in json.dumps(cycle["history"])
    versions = runner.events("version")
    assert sum(len(v["trajectories"]) for v in versions) == 100
    worlds = [[(t["seed"], t["history"]) for t in v["trajectories"]] for v in versions]
    assert all(world == worlds[0] for world in worlds)
    feedback = policy_feedback(versions[0])
    assert "trajectories" not in feedback
    assert len(feedback["representative_traces"]) <= 2


def coverage_tree():
    nodes = [dict(ROOT, score=-1)]
    for i in range(1, 7):
        nodes.append(
            {
                "id": f"n{i}",
                "parent": "root" if i < 5 else f"n{i - 1}",
                "score": -1 + i / 100,
                "status": "ok",
                "family_id": f"family{min(i, 4)}",
            }
        )
    return nodes


def test_stop_floor_shared_online_replay_and_bonus():
    nodes = coverage_tree()
    for size in range(1, 7):
        obs = observation(nodes[:size], 8, CONFIG)
        assert not obs["stop_allowed"]
        action, constrained = constrain_action(None, obs)
        assert constrained
        assert action == ("root" if size < 5 else f"n{size - 1}")
    assert observation(nodes, 8, CONFIG)["stop_allowed"]
    assert constrain_action(None, observation(nodes, 8, CONFIG)) == (None, False)
    result = replay(nodes, lambda o, s: None, 0, 8, 0.0025, -1000, 0.001, CONFIG)
    assert result["unique_families"] == 4
    assert result["constrained_actions"] == 11  # Includes planned missing continuations.
    assert result["objective"] == pytest.approx(-0.94 - 0.0025 * 6 + 0.001 * 4)
    assert all(t["constrained"] for t in result["trace"])
    # Fallback must not peek at a hidden child to evade empty_action.
    result = replay(nodes[:2], lambda o, s: None, 0, 8, 0.0025, -1000, 0.001, CONFIG)
    assert result["reason"] == "full_tree"


def test_online_early_stop_is_constrained(tmp_path, monkeypatch):
    runner = runner_at(tmp_path)
    runner.config = dict(runner.config)
    monkeypatch.setattr(runner, "policy", lambda name: lambda obs, seed: None)
    runner.run()
    assert len(runner.events("action_constrained")) == 24
    assert runner.status()["accepted_measured_nodes"] == 24
    assert runner.status()["distinct_families"] >= 4
    assert runner.status()["completed_cycles"] == 3
    assert runner.status()["finished"] == "global_attempt_cap"
    assert runner.status()["replay_trajectories"] == 300


def test_restart_after_proposal_acceptance_and_v1_state_rejected(tmp_path):
    runner = runner_at(tmp_path)
    event = runner.journal.append("attempt", attempt="n0001", outer=0, parent="root")
    p = proposal()
    runner.journal.append(
        "proposal_accepted",
        attempt=event["attempt"],
        outer=0,
        parent="root",
        proposal=p,
        architecture_delta=["a", "b", "c", "d"],
        family_id="distinct",
        axes_signature=axes_signature(p["architecture_axes"]),
    )
    runner.recover()
    runner.recover()
    assert len(runner.events("outcome")) == 1
    assert runner.ledger()[0]["status"] == "interrupted"
    assert not runner.ledger()[0]["eligible"]
    assert len(runner.root_references()) == 5
    assert runner.status()["proposal_sessions"] == 0
    from rsi.runner import load_config

    config = dict(CONFIG, protocol_version="dream-rsi-v1")
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="fresh"):
        load_config(path)


def test_real_discovery_command_mocked_rejection_precedes_gpu(tmp_path, monkeypatch):
    runner = runner_at(tmp_path)
    runner.synthetic = False  # Exercise real orchestration with both external boundaries mocked.
    sessions, probes = [], []
    workspace = runner.state / "attempts/n0001/workspace"

    monkeypatch.setattr("rsi.runner.isolated", lambda args, *a, **k: args)

    def fake_codex(argv, cwd, logs, timeout, stop, prompt):
        sessions.append(argv)
        assert "--ephemeral" in argv
        assert argv[argv.index("--output-schema") + 1] == "/tmp/work/proposal_schema.json"
        assert argv[argv.index("-o") + 1] == "/tmp/work/.proposal.json"
        assert "v1_novelty_reference" not in prompt and "trainability.json" not in prompt
        assert json.loads((workspace / "proposal_schema.json").read_text())["type"] == "object"
        (workspace / ".proposal.json").write_text(json.dumps(proposal()))
        (workspace / PLUGIN / "new.py").write_text(
            "class New:\n    def encode(self, x): return self.route(self.mix(x))\n"
            "    def route(self, x): return x\n    def mix(self, x): return x\n"
        )
        if len(sessions) == 1:
            (workspace / "proposal_schema.json").write_text("{}")
        else:
            assert "candidate changed forbidden file: proposal_schema.json" in prompt

    def fake_evaluate(work, output, logs):
        probes.append(work)
        assert len(runner.events("proposal_rejected")) == 1
        assert len(runner.events("proposal_accepted")) == 1
        assert not (work / ".proposal.json").exists()
        return {"score": -0.5, "optimizer_steps": 500}

    monkeypatch.setattr("rsi.runner.run_logged", fake_codex)
    monkeypatch.setattr(runner, "evaluate_workspace", fake_evaluate)
    runner.attempt(0, "root")
    assert len(sessions) == 2 and len(probes) == 1
    node = runner.events("outcome")[0]["node"]
    assert node["proposal"] == proposal()
    assert node["status"] == "ok"


def test_prior_outer_root_duplicate_and_refinement_label(tmp_path, monkeypatch):
    runner = runner_at(tmp_path)
    runner.attempt(0, "root")
    reference = runner.root_references()[-1]
    changed_label = proposal("renamed")
    with pytest.raises(ValueError, match="overlaps"):
        check_novelty(
            changed_label, reference["architecture_delta"], "root", runner.root_references(), CONFIG
        )
    with pytest.raises(ValueError, match="structural_refinement"):
        check_novelty(proposal(), {"a", "b", "c", "d"}, "n0001", [], CONFIG)
    with pytest.raises(ValueError, match="new_family"):
        check_novelty(proposal(parent="n1"), {"a", "b", "c", "d"}, "root", [], CONFIG)


def test_default_policy_opens_five_roots_then_refines():
    from rsi.policy import execute

    source = (REPO / "rsi/policies/initial.py").read_text()
    nodes = [ROOT]
    for i in range(5):
        assert execute(source, observation(nodes, 8, CONFIG), 42) == "root"
        nodes.append(
            {
                "id": f"n{i}",
                "parent": "root",
                "score": -1 + i / 100,
                "status": "ok",
                "family_id": str(i),
            }
        )
    assert execute(source, observation(nodes, 8, CONFIG), 42) in {n["id"] for n in nodes[1:]}


def test_failed_nodes_do_not_satisfy_coverage_or_diversity():
    nodes = [dict(ROOT, score=-1)]
    for i in range(1, 7):
        nodes.append(
            {
                "id": f"f{i}",
                "parent": "root",
                "score": -1000.0,
                "status": "implementation_failure",
                "eligible": False,
                "family_id": f"fake-family-{i}",
            }
        )
    obs = observation(nodes, 8, CONFIG)
    assert obs["measured_nodes"] == 0
    assert obs["root_branches"] == 0
    assert obs["distinct_families"] == 0
    assert not obs["stop_allowed"]

    result = replay(
        nodes,
        lambda o, s: "root",
        0,
        8,
        CONFIG["beta1"],
        CONFIG["failure_score"],
        CONFIG["beta_diversity"],
        CONFIG,
    )
    assert result["unique_families"] == 0
    assert result["objective"] == pytest.approx(-1.0 - CONFIG["beta1"] * result["revealed"])


def test_global_stop_floor_requires_two_cycles_and_twelve_measured(tmp_path, monkeypatch):
    runner = runner_at(tmp_path)
    # Make every synthetic policy ask to STOP as soon as local coverage permits.
    monkeypatch.setattr(runner, "policy", lambda name: lambda obs, seed: None)
    runner.run()
    status = runner.status()
    assert status["completed_cycles"] >= 2
    assert status["accepted_measured_nodes"] >= 12
    assert status["finished"] == "global_attempt_cap"
    assert not runner.global_stop_allowed(0)
    assert status["replay_trajectories"] >= 200


def test_global_stop_floor_config_validation(tmp_path):
    from rsi.runner import load_config

    for key, value in (
        ("min_outer_iterations_before_stop", CONFIG["max_outer_iterations"] + 1),
        ("min_total_measured_nodes_before_stop", CONFIG["max_real_attempts"] + 1),
    ):
        bad = dict(CONFIG, **{key: value})
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="coverage floors"):
            load_config(path)


def test_compact_failure_reason_without_heavy_metrics(tmp_path):
    runner = runner_at(tmp_path)
    runner.journal.append(
        "outcome",
        outer=0,
        attempt="n9999",
        node={
            "id": "n9999",
            "outer": 0,
            "parent": "root",
            "score": CONFIG["failure_score"],
            "status": "implementation_failure",
            "summary": "shape mismatch in candidate fusion path",
            "metrics": {"hardware": "must not leak", "trainability": [1, 2, 3]},
            "proposal": proposal(),
            "architecture_delta": ["a", "b", "c", "d"],
            "axes_signature": axes_signature(proposal()["architecture_axes"]),
            "family_id": "broken",
            "eligible": False,
        },
        workspace_hash=None,
    )
    item = runner.ledger()[-1]
    assert item["failure_reason"] == "shape mismatch in candidate fusion path"
    assert "metrics" not in item
    assert "hardware" not in json.dumps(item)
