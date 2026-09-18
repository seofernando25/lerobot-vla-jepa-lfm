# RSI protocol

This directory is the immutable research harness. Runtime trees, candidate source snapshots, checkpoints, Codex logs, replay traces, and event journals live under ignored `.rsi/`.

## Mapping to Dream-RSI

The implementation follows the paper's core three-stage loop:

- **Online exploration:** one exploration-policy version is frozen for an entire new discovery tree; a fixed coding agent generates candidates and a fixed evaluator scores them.
- **Replay simulator:** historical outcomes are not regenerated. Replay begins with root only and reveals recorded children only after the current policy selects the corresponding revealed root/leaf.
- **Policy improvement:** a fixed policy-development agent revises executable exploration-policy code; all versions are scored on fixed history and the incumbent is always eligible; the best replay score is redeployed.

The replay objective is the paper's quality/cost form with `W=1`: best revealed node score minus `beta1 * attempted_nodes`. There is no parallelism bonus.

## Deliberate adaptations

1. **Single GPU:** `W=1`, so real generation/evaluation is serial.
2. **Neural training is the evaluator:** the clean base LFM is measured once, then each real node receives the same 500-step training budget. Varying optimizer steps per edge is intentionally forbidden because unobserved fidelity counterfactuals cannot be replayed exactly.
3. **Small-history dreams:** each of four policy-version slots is replayed on 25 common history/seed worlds. Seeded tie-breaking makes those trajectories useful when only a few historical trees exist. Total = exactly 100 replay trajectories/cycle.
4. **Search domain and score:** v1 searches on the fixed local IPEC LIBERO-Spatial LeRobot dataset. Node quality is negative deterministic held-out LeRobot eval loss. This is a search proxy; final claims require matched simulator confirmation.
5. **No Qwen training:** Qwen is outside the search and appears only as a published-checkpoint reference in final evaluation.

## Isolation

A discovery worker sees only:

- its selected parent's candidate plugin source;
- minimal package metadata/lock file;
- a small immutable contract test;
- explicitly supplied observed-node summaries and fixed protocol in its prompt.

It receives no Git history, README, agent board, RSI runtime state, or previous experiment directory. Sibling repository clones under the repository parent are masked by the OS sandbox. Each real proposal is a fresh ephemeral Codex invocation; sessions are never resumed/forked.

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
