# EXP-002 — LeRobot baseline reproduction

## Purpose

Verify the official LeRobot `lerobot/VLA-JEPA-LIBERO` Qwen3 checkpoint on our machine before any LFM comparison.

## Gate

Start with the documented 10-episode `libero_spatial` command from the root README. If the pipeline is healthy, run the larger matched protocol defined in `experiment.yaml`.

## Why

EXP-001 trained only 1,000 steps on the legacy ginwind/starVLA stack and its 0/10 smoke result is **not** an upstream VLA-JEPA baseline. EXP-002 establishes the real reference using untouched LeRobot code and checkpoint.

## Status

Planned. No result should be reported until a run record exists under `runs/`.
