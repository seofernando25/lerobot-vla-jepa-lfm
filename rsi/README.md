# RSI protocol V4 continuation

V4 is a fresh Dream-RSI continuation study. It does **not** resume or mutate the V3 journal. At initialization it imports compact V1/V2/V3 mechanism history and copies selected measured V3 candidate workspaces into the new state as immutable refinement anchors.

## Continuation anchors

The configured V3 anchors are:

- `n0018`: ordered visual relation memory conditioning (V3 score `-0.3012`);
- `n0014`: prompt-written sparse prototype conditioning (V3 score `-0.3027`);
- `n0017`: global spectral temporal conditioning (V3 score `-0.3046`).

They are copied from the prior state into `.rsi/anchors/` and hash-frozen in the initialized journal event. Discovery receives only its selected isolated source plus compact observed history; it cannot browse the prior study, Git history, other candidate workspaces, project docs, or runtime state.

A prior V3 state is importable when it is explicitly finished **or** exhausted with every reservation closed. This intentionally supports the capped V3 study whose 24/24 reservations already have terminal outcomes even though its controller did not write a final `finished` event.

## Online exploration

Physical evaluator workers remain `W=1`; logical proposal batch size remains `B=4`. A full batch is fixed to:

- two **structural refinement** slots;
- two **genuinely novel root-family** slots.

Unused configured V3 anchors are consumed first in priority order. After the anchors have been seeded, refinement slots may branch from strong eligible leaves in the current continuation tree; if no suitable leaf is available the least-used anchor can be reused. Novel slots always start from clean root and must pass the existing AST/axes novelty checks against imported V1/V2/V3 history.

Anchor refinements use the exact frozen V3 candidate source as their implementation workspace. In the new causal replay tree they are represented as root branches, but their mechanism-family identity is inherited from the anchor so relabeling cannot earn a diversity bonus.

Proposal implementation remains four concurrent fresh ephemeral Astra-Low sessions. Every slot sees the same frozen pre-batch history and no sibling proposal content or score. Accepted candidates are evaluated sequentially on the single GPU.

## Screening and promotion

Every accepted candidate first receives the unchanged **500 optimizer-step** screening probe. Clean root is measured once at the same 500-step budget.

A candidate whose screening score is within `promotion_margin=0.005` of root may receive a separate **1500-step promotion probe**. Promotions are capped at six candidates. The first qualifying candidate triggers one matched 1500-step clean-root measurement; promoted candidates are compared with that longer-budget root.

Search and promotion probes keep `--save_checkpoint=false`. Promotion retrains from the same published initialization rather than continuing a 500-step checkpoint. The 500-step score remains the causal tree score; the matched promotion delta is attached as extra evidence for later exploration-policy decisions.

## Failure semantics

V4 separates software/architecture evidence from machine/runtime evidence:

- `ok`: completed fixed screening probe;
- `implementation_failure`: candidate/software error such as a dtype or shape bug;
- `runtime_failure`: evaluator timeout or recognized CUDA/driver failure;
- `interrupted`: explicit operator stop / reboot recovery;
- `proposal_failure`: proposal sessions exhausted before a probe.

Before each real evaluator launch the runner checks `nvidia-smi` and performs a minimal PyTorch CUDA allocation/synchronize health check. Recognized GPU launch or initialization failures stop the batch instead of silently falling into unusably slow execution.

`runtime_failure` and `interrupted` reservations do **not** spend the 24-probe research budget. They are never relaunched under the same attempt ID; a later explicit resume may reserve replacements. V4 bounds pathological environments at 32 total reservations and four runtime failures.

## Replay and policy improvement

The replay objective remains:

`best revealed quality - 0.0025 * revealed nonroot nodes + 0.001 * distinct families`

The protocol keeps four exploration-policy versions, 25 paired history/seed worlds per version, and exactly **100 dream trajectories per completed cycle**.

Replay uses the same mixed-batch contract: refinement slots prefer revealed leaves and novel slots open root branches. Frozen V3 anchor source is an online continuation mechanism only; replay never invents hidden anchor outcomes.

The initial V4 policy ranks leaves by fixed 500-step score plus any available matched promotion delta. Policy-development revisions remain sandboxed and replayed against immutable recorded history.

## State and recovery

`.rsi/events/NNNNNNNN.json` is the authoritative append-only hash-chained journal. Attempt, promotion, and policy-revision reservations are written before external work. Restart recovery closes interrupted reservations without repeating potentially executed work.

The harness/config/base/anchor manifests freeze at initialization. Never hand-edit `.rsi/`, delete state to recover, or modify `rsi/` after a real V4 study is initialized.

## Commands

A real V4 study requires a prior sealed/exhausted RSI state:

```bash
python -m rsi init --prior-study <v3-state>
python -m rsi status
python -m rsi run
python -m rsi stop
python -m rsi run --resume
python -m rsi confirm-plan
```

The evaluator requires `RSI_DATASET_ROOT` to point to the fixed local LIBERO-Spatial LeRobot dataset.

Synthetic validation requires no prior state, network service, Codex session, or GPU:

```bash
python -m rsi dry-run
```

## Fixed research boundary

Candidates may modify only Python source under `src/lerobot_policy_vla_jepa_lfm/**`. Registration/processor files, LeRobot, dataset/split, preprocessing/normalization, scoring, Qwen baseline, VLA-JEPA action/world-model architecture and published initialization, inherited losses, and prediction semantics remain protected.

The search metric is negative held-out eval loss and is a discovery proxy, not a LIBERO success claim. Final claims still require matched confirmation training and closed-loop LIBERO evaluation.
