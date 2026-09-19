# AGENTS.md

## Scope

This is a two-agent computer-vision research repo. The active study is **Dream-RSI V4 continuation** over the LFM2.5-VL side of LeRobot VLA-JEPA.

## Coordination

`F` = agent for `seofernando25`; `N` = Noah's agent.

- Protocol: `AGENT_BOARD.md`; authoritative append-only log: `AGENT_BOARD.jsonl`.
- Fetch/read the latest remote board at session start, before claims/shared work, and before final push.
- Use `CLM`, then `DONE` or `BLK`; never mutate old board records.

## Active RSI protocol

Read `rsi/README.md` and `rsi/config.json` before touching research code.

- V4 (`dream-rsi-v4-continuation`) imports sealed/exhausted V3 history into a **new** state. Do not resume or mutate the V3 journal to perform V4.
- Initialization freezes exact V3 anchor workspaces `n0018`, `n0014`, and `n0017` into `.rsi/anchors/`.
- Full B=4 batches are exactly **2 structural refinements + 2 genuinely novel root families**.
- Every candidate has the same 500-step screening budget. Candidates within 0.005 of root may additionally receive the separately recorded matched 1500-step promotion protocol; promotion does not replace the screening score.
- Runtime/CUDA failures are not architecture evidence. Runtime/interrupted reservations do not consume the 24 research-probe budget but remain terminal under their original IDs.
- The harness under `rsi/` is frozen once a real `.rsi` state is initialized. Never edit `rsi/`, `rsi/config.json`, `.rsi/`, or anchor state during an active study.
- Never hand-edit runtime state. Use `python -m rsi status|stop|run --resume`.
- Discovery workers must remain fresh/ephemeral and receive no Git history, agent board, project README, prior journal, sibling workspace, or unrelated runtime state.
- Qwen is never trained. `lerobot/VLA-JEPA-LIBERO` is evaluation-only.
- LeRobot remains pinned at `30074f7f1358b3c015ae1750017200e86e9c4eb6`.

## Mutable research surface

Candidates may make substantive changes to LFM hidden-state selection, representation mixing, adapters, multimodal fusion/conditioning, new LFM-side modules, and explicit LFM-side trainability.

They may **not** alter dataset/split, evaluator, scoring, preprocessing/normalization, Qwen, simulator protocol, VLA-JEPA action-head/world-model architecture or initialization, inherited action/world losses, or inherited prediction semantics.

## Scientific reporting

- Search uses the fixed LIBERO-Spatial LeRobot dataset in `rsi/config.json`.
- 500-step score = negative held-out eval loss. It is a discovery proxy, not LIBERO success.
- Promotion is a separate matched 1500-step evidence channel, bounded to six candidates.
- Each completed offline cycle must contain exactly 100 replay trajectories.
- Final comparison requires fresh matched-budget training and closed-loop LIBERO evaluation.

## V4 batching, anchors, and recovery

- Physical evaluator W=1; logical batch B=4.
- Frozen V3 anchors are copied into new state and hash-verified on every run.
- Novel slots remain screened against imported V1/V2/V3 structural history.
- The evaluator performs a CUDA health check before real training.
- A reboot/operator stop closes unfinished attempts/promotions without relaunching the same external work.
- Resume only with `python -m rsi run --resume`. V4 may create replacement attempt IDs for runtime/interrupted reservations until research/safety caps are reached.
- Real initialization requires `python -m rsi init --prior-study <sealed-or-exhausted-v3-state>`.

## Hygiene

Never commit `.rsi/`, weights, datasets, checkpoints, videos, caches, credentials, virtualenvs, or machine-specific paths. Keep model/data caches outside tracked source.
