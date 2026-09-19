# Dream-RSI for LeRobot VLA-JEPA + LFM2.5-VL

Computer-vision research project using a Dream-RSI-style exploration loop to discover improved LFM2.5-VL conditioning architectures for LeRobot VLA-JEPA.

The current protocol is **`dream-rsi-v4-continuation`**. It starts a fresh state from the sealed/exhausted V3 history rather than resuming V3.

## V4 continuation

Each full logical batch has **2 structural refinements + 2 genuinely novel root families**. The first refinement lineage uses frozen copies of V3 candidates `n0018` (ordered visual relations), `n0014` (prompt-written sparse memory), and `n0017` (global spectral temporal conditioning). Later refinements may branch from strong V4 leaves.

Every candidate receives the same **500-step** screen. Candidates within `0.005` of clean root may receive a separately recorded **1500-step matched promotion**, compared with a 1500-step clean-root baseline. Promotion does not replace the screening score and saves no search checkpoint.

CUDA/driver failures and evaluator timeouts are recorded as runtime failures rather than architecture evidence. Runtime/interrupted reservations do not spend the 24 research-probe budget; replacements on explicit resume are bounded by 32 total reservations and four runtime failures.

Replay keeps four policy versions × 25 paired worlds = **100 dream trajectories per completed cycle**, with objective:

`best quality - 0.0025 * revealed nonroot nodes + 0.001 * distinct families`

See [`rsi/README.md`](rsi/README.md) for the full scientific and recovery contract.

## Fixed boundary

Search may change only the LFM plugin architecture surface: hidden-state selection, representation mixing, adapters, multimodal fusion/conditioning, new LFM-side modules, and explicit LFM-side trainability.

The harness rejects changes to LeRobot, preprocessing/normalization, data/split, scoring, Qwen, the VLA-JEPA action-head/world-model architecture or published initialization, inherited losses, and inherited prediction semantics.

Qwen is never trained. Search loss is a discovery proxy, not a LIBERO success claim.

## Setup / validation

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --extra dev --extra eval
PYTHONPATH=src uv run pytest
PYTHONPATH=src uv run ruff check rsi tests
PYTHONPATH=src uv run python -m rsi dry-run
```

## Start V4

Initialize from the sealed/exhausted V3 state:

```bash
PYTHONPATH=src:. uv run python -m rsi init --prior-study <v3-state>
PYTHONPATH=src:. uv run python -m rsi status
PYTHONPATH=src:. uv run python -m rsi run
```

After an interruption:

```bash
PYTHONPATH=src:. uv run python -m rsi run --resume
```

Inspect or stop cleanly:

```bash
PYTHONPATH=src:. uv run python -m rsi status
PYTHONPATH=src:. uv run python -m rsi stop
```

Defaults: one RTX 3090 evaluator worker; B=4 logical proposals; 2 refinement + 2 novel slots; one matched 500-step root screen; 500 steps/candidate screen; up to six matched 1500-step promotions; 8 research probes per online tree; up to 3 online/dream cycles; hard cap 24 research probes; 32 total reservations; deterministic 10% held-out split; 64 held-out evaluation samples.

## Confirmation

After search, `python -m rsi confirm-plan` emits the matched confirmation plan. Final claims require fresh matched-budget training and closed-loop LIBERO evaluation.

## Attribution

Dream-RSI: Zheng et al. (2026), arXiv:2609.14858. VLA-JEPA: Sun et al. (2026). LeRobot is maintained by Hugging Face; LFM2.5-VL is by Liquid AI. No model weights or datasets are redistributed here.
