# Experiments

Experiments are immutable research records, not scratch folders.

```text
experiments/exp-XXX-name/
  experiment.yaml
  README.md
  configs/
  runs/
    <run-id>/
      run.json
      metrics.json
```

`experiment.yaml` defines the question and controls. A run is one execution. Multiple agents never edit the same run directory; each creates a new run ID. Heavy artifacts live outside Git and are referenced by URI/path only when portable.

Required run lifecycle: `planned -> running -> completed`, or terminal `failed` / `aborted`.

`EXP-001` predates this schema and is explicitly marked legacy.
