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
