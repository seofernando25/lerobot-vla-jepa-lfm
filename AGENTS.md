# AGENTS.md

## Scope

This repo is a two-agent research project comparing upstream LeRobot VLA-JEPA/Qwen3 with an out-of-tree LFM2.5-VL replacement.

## Coordination

`F` = agent for `seofernando25`; `N` = Noah's agent.

- Protocol: `AGENT_BOARD.md`; authoritative append-only log: `AGENT_BOARD.jsonl`.
- At session start and before shared work, fetch `main` and read the latest remote JSONL; do not trust a stale local board.
- File order is authoritative; UTC `ts` is metadata only.
- Use `CLM` before duplicate-able work and close with `DONE` or `BLK`.
- Board-only commits may go to `main` independently of unfinished code. If local history contains unpushed code commits, use a clean worktree for the board commit.
- Never mutate old board records; use `CORR` with `ref`.

## Framework boundary

- **Do not copy or patch LeRobot core into this repo.** LeRobot is pinned at `30074f7f1358b3c015ae1750017200e86e9c4eb6`; changing that pin is a protocol change and must be recorded in a new experiment or explicit migration.
- Qwen baseline means the untouched upstream `vla_jepa` policy/checkpoint.
- Our code belongs under `src/lerobot_policy_vla_jepa_lfm/` and must register as `vla_jepa_lfm`.
- Keep LFM vision/language weights frozen by default. Any unfreezing is an explicit experiment variable.
- Do not silently change preprocessing, action normalization, horizon, world model, or simulator protocol between arms.

## Experiment method

Every new experiment lives at `experiments/exp-XXX-short-name/` and must have:

- `experiment.yaml`: ID, status, hypothesis, independent variable, controls, protocol, code/LeRobot revisions.
- `README.md`: concise interpretation and limitations.
- `configs/`: immutable config/command snapshots for every arm when applicable.
- `runs/<run-id>/run.json` + `metrics.json`: one immutable record per execution.

One experiment tests one primary question. Changed hypothesis/protocol => new experiment ID. Do not overwrite prior records.

Each run records: agent, UTC times, git SHA, LeRobot SHA, arm/config hash, seed, dataset, steps, effective batch size, precision, hardware, checkpoint source, status, and metric provenance.

Separate training-batch diagnostics from held-out or simulator results. Never call a smoke test an official LIBERO score.

## Before long runs

1. Refresh `AGENT_BOARD.jsonl`; claim work.
2. Create experiment/run record and freeze configs.
3. Record git + LeRobot revisions.
4. Run `uv run pytest` and a short forward/backward smoke test.
5. Verify trainable parameter groups and GPU memory.
6. Use the same evaluation protocol for compared arms.

## Hygiene

Never commit weights, datasets, videos, caches, credentials, virtualenvs, or machine-specific absolute paths. Keep root README concise; evidence belongs in `experiments/`.
