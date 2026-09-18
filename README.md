# Dream-RSI for LeRobot VLA-JEPA + LFM2.5-VL

Computer-vision class project: use a **Dream-RSI-style exploration loop** to discover a better LFM2.5-VL-450M conditioning architecture for VLA-JEPA.

The fixed reference stack is Hugging Face LeRobot VLA-JEPA. Qwen3-VL is **never trained here**; the published `lerobot/VLA-JEPA-LIBERO` checkpoint is evaluation-only. Search compute is spent only on LFM candidates.

## Research question

Can a fixed coding agent plus an evolving exploration policy discover an LFM-side representation/fusion architecture that improves held-out VLA-JEPA training quality and, after matched confirmation training, LIBERO closed-loop performance?

## Fixed boundary

Search may change the LFM plugin's architecture surface: hidden-state selection, adapters, fusion, conditioning, new LFM-side modules, and LFM-side trainability.

The harness rejects changes to LeRobot, preprocessing, normalization, data split, scoring, Qwen, the VLA-JEPA action-head/world-model architecture, their published initialization, or their inherited loss/action semantics.

## Dream-RSI loop

1. **Online explore.** A frozen exploration-policy program selects the root or a currently revealed leaf. Fresh ephemeral `gpt-6-astra` / low-reasoning Codex sessions propose structured architectures from that parent's isolated source snapshot (at most three sessions per attempt, resetting source after each rejection).
2. **Fixed real probe.** The clean base is measured once; every accepted node then trains for exactly 500 optimizer steps on the same local LIBERO-Spatial protocol and receives `score = -held_out_eval_loss`.
3. **Replay world.** Completed trees are replayed prefix-causally. Selecting root reveals the earliest recorded unopened branch; selecting a leaf reveals only its recorded child.
4. **Dream.** Four exploration-policy versions (incumbent + up to three revisions) are evaluated on the same history/seed worlds, 25 trajectories each = **100 dream trajectories per cycle**.
5. **Redeploy.** The best replay-scoring policy controls the next new online tree.

Protocol **dream-rsi-v3-batched** retains AST structural novelty screening and a global accepted-mechanism ledger across fresh outer trees. Roots must be distinct from prior v2 roots and a trusted, score-free v1 reference; parameter-only changes are rejected before evaluation. The initial policy targets five roots. Per-tree STOP requires six accepted nodes, four root branches and four distinct mechanism families; a STOP cannot end the study before two online/dream cycles and 12 measured nodes. Replay uses `best quality - 0.0025 * revealed nonroot nodes + 0.001 * distinct families`; these coverage floors, weaker cost penalty and novelty bonus deliberately extend Dream-RSI.

V1 is preserved by tag `rsi-v1-complete`, the sealed local archive, and [tracked summary](rsi/studies/v1_summary.json). Neither v1 results nor novelty references are visible to discovery workers. V2 requires fresh state; never reuse or modify the sealed archive.

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

Defaults: one RTX 3090 worker, one matched 500-step base measurement, 500 steps/probe, up to 8 attempt reservations per online tree, up to 3 online/dream cycles, hard cap 24 attempts, deterministic 10% per-task held-out split, and 64 held-out evaluation samples.

## Confirmation

Search loss is a discovery proxy, not a LIBERO success claim. After search, `uv run python -m rsi confirm-plan` emits a matched fixed-budget retraining/evaluation plan for:

- clean base LFM architecture;
- best measured RSI architecture;
- untouched published Qwen checkpoint (evaluation only).

The confirmation template uses LIBERO-Spatial with 10 episodes for each of 10 tasks (100 rollouts).

## Attribution

Dream-RSI: Zheng et al. (2026), arXiv:2609.14858. VLA-JEPA: Sun et al. (2026). LeRobot is maintained by Hugging Face; LFM2.5-VL is by Liquid AI. No model weights or datasets are redistributed here.

## V3 logical batches and shared resources

`dream-rsi-v3-batched` uses physical evaluator **W=1** and logical proposal
batch **B=4**, with four fresh ephemeral Astra-Low implementation sessions in a
ThreadPoolExecutor. This is a single-evaluator adaptation of Dream-RSI parallel
exploration, not equivalent to physical W=4. Each slot sees the same frozen
pre-batch global/current history; it never sees sibling proposal content or scores.
Trusted retry rounds reset each rejected workspace from its unchanged parent.
Root collisions within a batch receive generic feedback only. Main-thread journal
records are ordered by round and slot. All proposals resolve before any GPU probe;
accepted candidates are evaluated sequentially by slot. All reserved outcomes and
`batch_done` must exist before the next policy/proposal batch begins. K1=8 normally
means two logical batches. Interrupted batches are closed without repeating any
possibly executed external work.

Online and replay share `plan_batch_actions`. Policy retains its single-action
interface and receives `batch_slot`, `batch_size`, and `planned_actions` in addition
to the unchanged measured prefix. Planned choices earn no STOP coverage. Root may
repeat; a non-root parent may appear once. Replay reveals selected recorded
children together after planning, preserving root creation order and never using
future scores. Four policy slots share 25 worlds: exactly 100 dreams per cycle.

Candidate workspaces contain only plugin Python source, minimal metadata/lock,
contract tests and schema. They exclude .venv, models, caches, datasets, checkpoints,
archives, studies, README, board and Git history. Dependencies use the repository's
shared .venv read-only; model weights use the machine's shared HF cache read-only.
The evaluator binds RSI_DATASET_ROOT read-only. There are no per-candidate
environments or model copies. uv uses its normal shared cache when maintaining
the repository environment; probe execution does not run uv or install packages.
Search baseline/probes pass `--save_checkpoint=false`; a post-probe guard fails
without deleting unexpected safetensors, pt/pth or DCP shards. Only confirmation
training explicitly saves checkpoints for final LIBERO evaluation. The harness
checks `shutil.disk_usage(state).free` against `min_free_disk_gib=20` before every
batch/evaluation and stops cleanly below the threshold.

Fresh initialization may use `python -m rsi init --prior-study <completed-state>`
to import compact accepted v2/v3 mechanism history and score-free root novelty
references into the trusted journal. The source study must be completed. No
archive, result directory, weights or full source is copied into a candidate.
Without this option the study starts with the tracked v1 novelty reference and
its own global v3 history. Do not initialize from or modify live v2 state. Harness
construction and `dry-run` require no real study, Codex discovery or GPU training.

Search probes log every 50 optimizer steps; the harness parses loss, action/world-model loss, gradient norm, LR, memory and throughput plus final held-out loss into a compact per-run telemetry.json. Raw logs are retained.
