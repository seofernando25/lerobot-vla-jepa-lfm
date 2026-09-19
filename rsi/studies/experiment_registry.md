# Dream-RSI experiment registry

This is the compact tracked scientific index for every V1-V4 candidate attempt preserved in the authoritative RSI journals at generation time.

Raw RSI journals, candidate workspaces, logs, checkpoints, datasets, and machine-specific runtime state remain intentionally ignored by Git.

## Coverage

| Study | Protocol | Code commit | Screen baseline | Attempts | Completed cycles | Promotion baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| V1 | dream-rsi-v1 | cdc503ddd2ed | -0.3118 | 8 | 2 | n/a |
| V2 | dream-rsi-v2 | 246ab026fe8c | -0.3116 | 14 | 2 | n/a |
| V3 | dream-rsi-v3-batched | 7bec8414138d | -0.2998 | 24 | 2 | n/a |
| V4 | dream-rsi-v4-continuation | 8ee433e168a0 | -0.3144 | 4 | 0 | -0.2968 |

Total attempts: **50**.

## Semantics

- recorded_status and recorded_score preserve authoritative journal values.
- screen_score exists only for completed held-out measurements; failure sentinels are not scientific scores.
- screen_delta_vs_baseline > 0 means the candidate beat that study's fixed-step root screen.
- promotion stores V4 matched 1500-step evidence; positive promotion delta means it beat the matched 1500-step root.
- failure_class is the compact scientific classification. V3 n0020 is normalized to runtime_failure because its retained evaluator log shows a CUDA launch failure.
- Compare scores directly within the same study and budget. Root baselines changed across protocol versions.

## Provenance

Generated from the preserved V1, V2, V3, and V4 append-only RSI journals, plus the tracked V1 summary for V1 mechanism labels. No raw weights, datasets, checkpoints, credentials, or machine paths are included.
