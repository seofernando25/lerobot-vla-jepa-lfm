# RSI protocol v2

This directory is the immutable research harness. Runtime trees, candidate source snapshots, checkpoints, Codex logs, replay traces, and event journals live under ignored `.rsi/`.

## Mapping to Dream-RSI

The implementation follows the paper's core three-stage loop:

- **Online exploration:** one exploration-policy version is frozen for an entire new discovery tree; a fixed coding agent generates candidates and a fixed evaluator scores them.
- **Replay simulator:** historical outcomes are not regenerated. Replay begins with root only and reveals recorded children only after the current policy selects the corresponding revealed root/leaf.
- **Policy improvement:** a fixed policy-development agent revises executable exploration-policy code; all versions are scored on fixed history and the incumbent is always eligible; the best replay score is redeployed.

The v2 objective deliberately extends the paper's quality/cost form with `W=1`:
`best revealed quality - 0.0025 * revealed nonroot nodes + 0.001 * distinct revealed families`.
The weaker beta1 (v1: 0.01) reduces premature stopping; the diversity bonus rewards coverage. There is no parallelism bonus.

## Deliberate adaptations

1. **Single GPU:** `W=1`, so real generation/evaluation is serial.
2. **Neural training is the evaluator:** the clean base LFM is measured once, then each real node receives the same 500-step training budget. Varying optimizer steps per edge is intentionally forbidden because unobserved fidelity counterfactuals cannot be replayed exactly.
3. **Small-history dreams:** each of four policy-version slots is replayed on 25 common history/seed worlds. Seeded tie-breaking makes those trajectories useful when only a few historical trees exist. Total = exactly 100 replay trajectories/cycle.
4. **Search domain and score:** v2 searches on the fixed local IPEC LIBERO-Spatial LeRobot dataset. Node quality is negative deterministic held-out LeRobot eval loss. This is a search proxy; final claims require matched simulator confirmation.
5. **No Qwen training:** Qwen is outside the search and appears only as a published-checkpoint reference in final evaluation.

## Diversity protocol

`protocol_version=dream-rsi-v2`; K1=8 attempt reservations/tree, K2=8 replay reveals,
three outer iterations and 24 total reservations maximum. The clean-base probe,
500-step candidate evaluator, data, split, batch and seed stay fixed.

Each final proposal is strict JSON validated against `proposal_schema.json` and
stored with its node. Roots declare `new_family`; refinements declare
`structural_refinement`. The trusted harness fingerprints Python class/function
definitions and meaningful scoped calls/dependency edges, ignoring constants,
width/head/layer counts, initialization scales, activation choices and utilities.
Delta is the symmetric difference against parent source; at least four features
must change. Root deltas must have Jaccard similarity <=0.45 to every accepted v2
root and the tracked v1 reference, and cannot duplicate normalized root axes.
This is a conservative syntax heuristic, not proof of semantic novelty; renamed
or unused code can evade it. Family labels alone never establish acceptance.

At most `max_proposal_retries=3` fresh proposal sessions are allowed per n-ID
(including the first session). Every rejected session emits `proposal_rejected`;
the next session starts from unchanged parent source and receives only rejection
reasons plus compact allowed history. Exhaustion closes `proposal_failure` without
a GPU probe or tree node. Accepted proposals with evaluator failures remain
terminal recorded outcomes; failed reservations still consume K1/global budget.
Accepted proposal metadata is recovered after interruptions without relaunching.

Discovery sees all accepted v2 attempts across trees, compacted to identity,
parent, outer, score, family, axes, structural change and status. No full metrics
or trainability arrays enter prompts/replay histories. Refinements inherit their
parent's family identity, so relabeling cannot earn a bonus. Root family labels
are normalized after structural screening.

The initial policy targets five independent roots. Per-tree STOP requires >=6 measured nonroot
nodes, >=4 measured root branches and >=4 measured distinct families. A STOP ends the current tree but cannot end the full study before two completed online/dream cycles and 12 measured nodes. Online and replay both coerce
early STOP to root until root/family floors hold, then the highest-scoring eligible
leaf (root if none). Coercions are recorded. Hard budgets and missing recorded
continuations can still end exploration/replay below coverage; no hidden outcomes
are synthesized. Policy feedback contains means, termination counts, constraint
counts and at most two traces. Four slots share 25 worlds each: exactly 100 dreams.

V1 remains sealed under tag `rsi-v1-complete` and summarized in
`studies/v1_summary.json`. `studies/v1_novelty_reference.json` contains only AST
deltas, normalized axes signatures and labels for v1 roots n0001/n0002/n0003/n0007,
derived by reading the sealed source snapshots. It has no scores or full source,
is trusted-harness-only, and works without the local archive. V2 never reuses v1
runtime state. The complete harness/config/reference manifest freezes at init.

## Isolation

A discovery worker sees only:

- its selected parent's candidate plugin source;
- minimal package metadata/lock file;
- a small immutable contract test and JSON proposal schema;
- explicitly supplied observed-node summaries and fixed protocol in its prompt.

It receives no Git history, README, agent board, rsi/ directory, v1 summary/reference, RSI runtime state, or previous experiment directory. Sibling repository clones under the repository parent are masked by the OS sandbox. Each real proposal is a fresh ephemeral Codex invocation; sessions are never resumed/forked.

The candidate may modify only Python files under `src/lerobot_policy_vla_jepa_lfm/`. Registration and processor files are frozen. Static and runtime guards reject overrides/monkeypatches of inherited VLA-JEPA loss/action methods or changes to the upstream action/world-model modules and their published initialization.

## State model

`.rsi/events/NNNNNNNN.json` is the authoritative append-only event journal with a hash chain. Attempt/revision reservations are written before external work. On restart, an interrupted reservation is closed as interrupted rather than silently rerun.

Each outer iteration creates a new tree. Root can open multiple independent branches. A non-root node can have only one recorded child, matching the replay contract.

## Commands

```bash
uv run python -m rsi dry-run
uv run python -m rsi init
uv run python -m rsi run
uv run python -m rsi run --resume
uv run python -m rsi status
uv run python -m rsi stop
uv run python -m rsi confirm-plan
```

`dry-run` is synthetic and is required to perform 100 replay trajectories without Codex/GPU/network work.
