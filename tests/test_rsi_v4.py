"""V4 continuation contracts: mixed exploration, frozen anchors, promotions and failures."""

from pathlib import Path

from rsi.core import observation
from rsi.novelty import compact_node
from rsi.process import ProcessTimeout
from rsi.runner import (
    ROOT,
    Runner,
    classify_external_failure,
    load_config,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO / "rsi/config.json")


def runner_at(tmp_path):
    runner = Runner(REPO, tmp_path / "state", synthetic=True)
    runner.initialize(REPO / "rsi/config.json")
    runner.verify()
    return runner


def test_v4_config_contract():
    assert CONFIG["protocol_version"] == "dream-rsi-v4-continuation"
    assert CONFIG["refinement_slots_per_batch"] == 2
    assert CONFIG["novel_slots_per_batch"] == 2
    assert CONFIG["proposal_batch_size"] == 4
    assert CONFIG["probe_steps"] == 500
    assert CONFIG["promotion_steps"] == 1500
    assert CONFIG["promotion_margin"] == 0.005
    assert CONFIG["continuation_anchor_ids"] == ["n0018", "n0014", "n0017"]


def test_first_batch_uses_two_frozen_v3_anchors_and_two_novel_slots(tmp_path):
    runner = runner_at(tmp_path)
    root = dict(ROOT, score=-0.75, status="ok")
    obs = observation([compact_node(root)], CONFIG["K1"], CONFIG)
    plan = runner.plan_online_batch(0, lambda o, s: None, obs, 4)
    assert plan["modes"] == ["refinement", "refinement", "novel", "novel"]
    assert plan["source_anchors"] == ["n0018", "n0014", None, None]
    assert plan["planned_actions"] == ["root", "root", "root", "root"]
    assert [
        e["source_anchor"] for e in plan["substitutions"] if e["reason"] == "continuation_anchor"
    ] == [
        "n0018",
        "n0014",
    ]


def test_anchor_refinement_inherits_family_and_exact_frozen_source(tmp_path):
    runner = runner_at(tmp_path)
    anchor = runner.anchor_metadata("n0018")
    before = runner.anchor_workspace("n0018")
    assert anchor["workspace_hash"]
    plan = {
        "planned_actions": ["root"],
        "source_anchors": ["n0018"],
        "modes": ["refinement"],
        "substitutions": [],
        "constrained_actions": 0,
    }
    runner.execute_batch(0, plan)
    accepted = runner.events("proposal_accepted")[0]
    outcome = runner.events("outcome")[0]["node"]
    assert accepted["source_anchor"] == "n0018"
    assert accepted["proposal"]["relation_to_parent"] == "structural_refinement"
    assert accepted["family_id"] == anchor["family_id"]
    assert outcome["source_anchor"] == "n0018"
    assert before.exists()


def test_near_root_candidate_gets_matched_long_promotion(tmp_path):
    runner = runner_at(tmp_path)
    runner.ensure_baseline()
    plan = {
        "planned_actions": ["root"],
        "source_anchors": [None],
        "modes": ["novel"],
        "substitutions": [],
        "constrained_actions": 0,
    }
    runner.execute_batch(0, plan)
    assert runner.events("promotion_baseline")[0]["metrics"]["optimizer_steps"] == 1500
    promotion = runner.events("promotion")[0]
    assert promotion["status"] == "ok"
    assert promotion["metrics"]["optimizer_steps"] == 1500
    assert promotion["baseline_score"] == -0.70
    assert promotion["delta_vs_baseline"] > 0
    tree = runner.tree(0)
    assert tree[1]["promotion"]["status"] == "ok"


def test_runtime_failure_is_not_architecture_failure(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "stderr.log").write_text("CUDA error: unspecified launch failure")
    (logs / "stdout.log").write_text("")
    assert classify_external_failure(RuntimeError("subprocess exit 1"), logs) == "runtime_failure"

    (logs / "stderr.log").write_text(
        "RuntimeError: value cannot be converted to type c10::BFloat16 without overflow"
    )
    assert (
        classify_external_failure(RuntimeError("subprocess exit 1"), logs)
        == "implementation_failure"
    )
    assert (
        classify_external_failure(ProcessTimeout("subprocess timeout"), logs) == "runtime_failure"
    )


def test_runtime_and_interruptions_do_not_spend_research_budget(tmp_path):
    runner = runner_at(tmp_path)
    runner.verify()
    for index, status in enumerate(
        ("runtime_failure", "interrupted", "implementation_failure", "ok"), 1
    ):
        event = runner.journal.append(
            "attempt",
            attempt=f"n{index:04d}",
            outer=0,
            parent="root",
            source_anchor=None,
        )
        runner.finish(event, status, status)
    assert runner.research_attempt_count() == 2
    assert runner.runtime_failure_count() == 1


def test_interrupted_promotion_baseline_can_be_re_reserved(tmp_path):
    runner = runner_at(tmp_path)
    runner.journal.append(
        "promotion_baseline_started",
        reservation=1,
        steps=CONFIG["promotion_steps"],
    )
    runner.recover()
    failures = runner.events("promotion_baseline_failure")
    assert failures[-1]["reservation"] == 1
    assert failures[-1]["status"] == "interrupted"
    metrics = runner.ensure_promotion_baseline()
    assert metrics["optimizer_steps"] == CONFIG["promotion_steps"]
    assert runner.events("promotion_baseline")[-1]["reservation"] == 2


def test_promotion_runtime_failure_counts_toward_runtime_safety_cap(tmp_path):
    runner = runner_at(tmp_path)
    runner.journal.append(
        "promotion",
        attempt="n0001",
        outer=0,
        status="runtime_failure",
        metrics={},
        baseline_score=-0.7,
        delta_vs_baseline=None,
        failure_class="runtime_failure",
    )
    runner.journal.append(
        "promotion_baseline_failure",
        reservation=1,
        status="runtime_failure",
        summary="cuda failed",
    )
    assert runner.runtime_failure_count() == 2
    assert runner.promotion_research_count() == 0
