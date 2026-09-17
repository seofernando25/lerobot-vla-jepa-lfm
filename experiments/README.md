# Experiments

Start new records from [`TEMPLATE.md`](TEMPLATE.md). Every experiment gets its own immutable directory:

```text
experiments/
  exp-XXX-short-name/
    README.md
    configs/
      arm-a.yaml
      arm-b.yaml
    metrics.json        # optional, small scalar/structured results only
```

Do **not** commit checkpoints, datasets, videos, TensorBoard logs, or full run directories.

## Required experiment record

Each experiment `README.md` must contain:

1. **Hypothesis** — one testable question.
2. **Change** — the independent variable.
3. **Controls** — what is held fixed.
4. **Protocol** — dataset, seed, steps, batch size, scheduler, hardware, and pretrained initialization.
5. **Metrics** — exact definitions and where they are computed.
6. **Results** — compact table with raw values.
7. **Conclusion** — what the result supports and what it does not support.
8. **Next experiment** — the smallest follow-up that reduces uncertainty.

Before reporting results, copy the exact config for every experimental arm into `configs/`. Record the git commit SHA used for the run. If a run predates a clean commit, say so explicitly rather than inventing a SHA.

## Naming

Use sequential IDs:

```text
exp-001-qwen3-vs-lfm25
exp-002-lfm-adapter-ablation
exp-003-lfm-unfreeze-depth
```

Do not overwrite old experiments. A changed hypothesis or protocol gets a new experiment ID.
