# AGENTS.md

## Scope

This is a two-agent computer-vision research repo. The active study is a Dream-RSI-style autonomous architecture search over the **LFM2.5-VL side** of LeRobot VLA-JEPA.

## Coordination

`F` = agent for `seofernando25`; `N` = Noah's agent.

- Protocol: `AGENT_BOARD.md`; authoritative append-only log: `AGENT_BOARD.jsonl`.
- Fetch/read the latest remote board at session start, before claims/shared work, and before final push.
- Use `CLM`, then `DONE` or `BLK`; never mutate old board records.
- Board-only commits may reach `main` independently of unfinished code.

## Active RSI protocol

Read `rsi/README.md` and `rsi/config.json` before touching research code.

- The harness under `rsi/` is a **fixed evaluator/orchestrator** once a real study is initialized. Do not edit it or `rsi/config.json` during an active `.rsi/` run.
- V2 (`dream-rsi-v2`) deliberately adds AST novelty screening, global accepted-mechanism history, five initial roots, and per-tree STOP floors of six measured nodes/four roots/four families, plus global STOP floors of two cycles/12 measured nodes. Replay uses weaker beta1=0.0025 plus beta_diversity=0.001; four paired 25-world slots remain exactly 100 dreams. Numeric/activation-only changes are not architectures.
- V1 is preserved by tag `rsi-v1-complete`, tracked summary, and sealed `.rsi-v1-complete`. Never edit the archive or reuse v1 runtime state for v2. Summary/reference remain invisible to discovery; only the trusted harness reads the score-free novelty reference.
- Never hand-edit `.rsi/`. Use `python -m rsi status|stop|run --resume`.
- Old EXP-001/EXP-002 records are intentionally absent from `main`; do not restore or summarize them into discovery-visible files. They remain in Git tag `pre-dream-rsi`.
- Discovery workers must remain fresh/ephemeral and receive no Git history, agent board, project README, old experiments, or hidden runtime state.
- Qwen is never trained. `lerobot/VLA-JEPA-LIBERO` is an evaluation-only reference.
- LeRobot remains pinned at `30074f7f1358b3c015ae1750017200e86e9c4eb6`.

## Mutable research surface

Candidates may make substantive changes to LFM hidden-state selection, representation mixing, adapters, multimodal fusion/conditioning, new LFM-side modules, and LFM-side trainability.

They may **not** alter the dataset/split, evaluator, scoring, preprocessing/normalization, Qwen, simulator protocol, VLA-JEPA action head/world-model architecture or initialization, inherited action/world losses, or inherited prediction semantics.

Do not narrow the search to known prior adapters in instructions. Measured evaluator outcomes are authoritative.

## Scientific reporting

- Search is restricted to the fixed LIBERO-Spatial LeRobot dataset in rsi/config.json. The clean base LFM is measured once with the same 500-step evaluator before candidate search.
- RSI probe score = negative held-out eval loss. It is a discovery proxy, not LIBERO success.
- Every real probe has the same fixed optimizer-step budget; policy resource control is branch/open/refine/stop, not hidden fidelity changes.
- Each offline cycle must contain exactly 100 replay trajectories under the tracked config.
- Final comparison requires fresh matched-budget training of base LFM and the RSI-selected architecture; Qwen is evaluated from its published checkpoint.
- Closed-loop LIBERO results must state suite, tasks, episodes/task, seed, and checkpoint.

## Hygiene

Never commit `.rsi/`, weights, datasets, checkpoints, videos, caches, credentials, virtualenvs, or machine-specific paths. Keep model/data caches outside tracked source.

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

## Interrupted-study recovery

- A reboot or terminated runner may leave a valid `.rsi/` journal with `stop_requested=true` and no final `finished` event. **Never re-run `init`, delete `.rsi/`, or hand-edit events to recover.**
- First inspect `python -m rsi status` and check that no RSI runner, evaluator, or Codex discovery process is active.
- Resume only with `python -m rsi run --resume`. The v3 harness removes the stale STOP marker, closes genuinely interrupted reservations/batches without relaunching external work, then continues/finalizes from the journal.
- If all reserved attempts already have outcomes, resume must not create new candidates; it only completes the remaining controller/finalization path allowed by the frozen config.
- Treat the initialized `rsi/` harness/config manifest as frozen across reboots. If verification fails, do not edit around it; inspect provenance/commit state first.

