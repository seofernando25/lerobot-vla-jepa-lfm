# Dream-RSI for LeRobot VLA-JEPA + LFM2.5-VL

Computer-vision class project: use a **Dream-RSI-style exploration loop** to discover a better LFM2.5-VL-450M conditioning architecture for VLA-JEPA.

The fixed reference stack is Hugging Face LeRobot VLA-JEPA. Qwen3-VL is **never trained here**; the published `lerobot/VLA-JEPA-LIBERO` checkpoint is evaluation-only. Search compute is spent only on LFM candidates.

## Research question

Can a fixed coding agent plus an evolving exploration policy discover an LFM-side representation/fusion architecture that improves held-out VLA-JEPA training quality and, after matched confirmation training, LIBERO closed-loop performance?

## Fixed boundary

Search may change the LFM plugin's architecture surface: hidden-state selection, adapters, fusion, conditioning, new LFM-side modules, and LFM-side trainability.

The harness rejects changes to LeRobot, preprocessing, normalization, data split, scoring, Qwen, the VLA-JEPA action-head/world-model architecture, their published initialization, or their inherited loss/action semantics.

## Dream-RSI loop

1. **Online explore.** A frozen exploration-policy program selects the root or a currently revealed leaf. One fresh ephemeral `gpt-6-astra` / low-reasoning Codex call proposes one architecture from that parent's isolated source snapshot.
2. **Fixed real probe.** The clean base is measured once; every accepted node then trains for exactly 500 optimizer steps on the same local LIBERO-Spatial protocol and receives `score = -held_out_eval_loss`.
3. **Replay world.** Completed trees are replayed prefix-causally. Selecting root reveals the earliest recorded unopened branch; selecting a leaf reveals only its recorded child.
4. **Dream.** Four exploration-policy versions (incumbent + up to three revisions) are evaluated on the same history/seed worlds, 25 trajectories each = **100 dream trajectories per cycle**.
5. **Redeploy.** The best replay-scoring policy controls the next new online tree.

Runtime state is append-only under ignored `.rsi/`. Prior experiment records were removed from `main` to avoid discovery bias and remain recoverable at tag `pre-dream-rsi`.

See [rsi/README.md](rsi/README.md) for the exact paper mapping and deviations.

## Setup

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --extra dev --extra eval
uv run pytest
uv run ruff check rsi tests
uv run python -m rsi dry-run
```

## Run

```bash
uv run python -m rsi init
uv run python -m rsi run
```

The runner is restart-safe. After an interruption:

```bash
uv run python -m rsi run --resume
```

Inspect or stop cleanly:

```bash
uv run python -m rsi status
uv run python -m rsi stop
```

Defaults: one RTX 3090 worker, one matched 500-step base measurement, 500 steps/probe, up to 6 real probes per online tree, up to 3 online/dream cycles, hard cap 18 real probes, deterministic 10% per-task held-out split, and 64 held-out evaluation samples.

## Confirmation

Search loss is a discovery proxy, not a LIBERO success claim. After search, `uv run python -m rsi confirm-plan` emits a matched fixed-budget retraining/evaluation plan for:

- clean base LFM architecture;
- best measured RSI architecture;
- untouched published Qwen checkpoint (evaluation only).

The confirmation template uses LIBERO-Spatial with 10 episodes for each of 10 tasks (100 rollouts).

## Attribution

Dream-RSI: Zheng et al. (2026), arXiv:2609.14858. VLA-JEPA: Sun et al. (2026). LeRobot is maintained by Hugging Face; LFM2.5-VL is by Liquid AI. No model weights or datasets are redistributed here.
