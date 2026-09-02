# Online Resident-FP8 Critic and Mechanism Diagnostics

## Motivation

The first full-persistent-state stage must isolate online Critic weight writes
from the already diagnosed target-state problem. The four large residual-Dense
kernels therefore become E4M3 payloads without a same-size FP32 master, while
target parameters/EMA and AdamW moments remain FP32 controls. Both online and
target residual GEMMs remain FP8; target storage uses the existing
`fp8_direct` path.

## Changes

- Added `critic_precision=fp8_resident`. The four online residual kernels store
  E4M3 codes plus one FP32 current-amax scale per ensemble member. Other online
  parameters remain FP32.
- Resident backward constructs the physical FP32 tree only inside the compiled
  update, obtains physical-weight gradients, applies AdamW with FP32 moments,
  and immediately requantizes the candidate. Neither the physical tree nor the
  candidate is returned as model state, checkpointed, or reused by the next
  update.
- FP32/direct/resident target initialization, Polyak updates, lag
  reconstruction, Actor updates, and gradient diagnostics now consume the
  physical online weights where required. Same-mode resident recovery requires
  FP8 metadata.
- The existing tensor-stat boundary samples the final real update. It records
  only nonredundant scalar evidence for update retention/direction,
  swallowed/code-unchanged coordinates, weight error, scale movement, a
  fixed-old-scale counterfactual, and same-batch expected-Q/JS/loss error.
- Recovery manifests split parameter, FP8 payload, FP8 metadata, optimizer, and
  total persistent bytes and report the metadata fraction. Added a foreground
  Dogs runner at `scripts/run_dogs_online_fp8_resident.sh`.

## Evaluation

Use `EXP-FP8-ONLINE-RESIDENT-DIAG-S42` against the completed target-forward-only
control. Diagnose the first divergence with the recorded parameter-write and
function-write scalars; do not infer bootstrap amplification without the later
matched teacher-target/open-loop control.

## Pitfalls

- `code_unchanged_fraction` is not swallowed-update fraction: a scale change can
  move every dequantized weight while all codes remain unchanged.
- The transient FP32 candidate is arithmetic, not persistent state. No result
  should describe this phase as low-precision optimizer-state training because
  AdamW moments deliberately remain FP32.
- Tensor diagnostics use a sampled real update and add compilation/work only at
  the configured interval; their timing is not training-throughput evidence.

## Validation and limits

- `python -m unittest tests.test_fp8`: 18 tests passed in 129.061 s.
- `python -m unittest discover -s tests`: 46 tests passed in 168.575 s.
- Small CPU execution confirmed E4M3 resident leaves, exclusively FP32 floating
  optimizer leaves, FP32 target parameters, finite write diagnostics, and
  deterministic same-mode checkpoint replay.
- `bash -n scripts/run_dogs_online_fp8_resident.sh` and `git diff --check`
  passed.
- This initial validation did not launch a GPU job; the later 2026-09-02
  extension below records the completed Blackwell smoke and formal Dogs runs.

## 2026-09-02 mechanism-diagnostic extension

This pass changes diagnostics only. It leaves the optimizer, quantizer,
resident payload, target update, and all training hyperparameters unchanged.
At the existing tensor-stat interval, every resident kernel and ensemble member
now records the following scalars from the exact pre-update physical weight,
the real Optax candidate, and the dequantized post-write resident weight:

- intended/actual tangential update L2, effective angular step, and angular
  retention;
- intended/actual first-order radial term, second-order update-squared term,
  predicted squared-norm change, and directly measured squared-norm change;
- intended AdamW decay L2/radial motion, post-write decay-channel radial
  motion, and radial retention;
- next-code L2/cosine, relative scale change, scale-only/code-only motion L2,
  and scale-motion fraction of actual L2.

The AdamW split follows installed Optax 0.2.5 exactly:
`scale_by_adam -> add_decayed_weights -> scale_by_learning_rate`. At a sampled
diagnostic update, the same transform is replayed from the same gradients and
old optimizer state with a zero parameter tree. This preserves adaptive
moments, bias correction, schedule, and ordering; total minus zero-parameter
replay is the decay update. `wd_applied_radial` is the signed radial motion of
the real post-FP8 update after subtracting the exact intended adaptive radial
motion, so write-time quantization loss is deliberately included in
`wd_radial_retention`.

Scale/code attribution uses an exact ordered path:

1. hold old codes fixed and change `old_scale -> new_scale` (`scale_only`);
2. hold the new scale fixed and change `old_codes -> new_codes` (`code_only`).

Consequently `scale_only + code_only == post_write - pre_write` up to FP32
roundoff. `scale_update_fraction_of_actual_l2` is a ratio of norms and may be
greater than one when scale-only and code-only motions cancel.

Validation:

- `python -m unittest tests.test_fp8`: 22 tests passed in 129.822 s.
- `python -m unittest discover -s tests`: 50 tests passed in 175.988 s.
- A three-update fixed-seed synthetic trajectory was bitwise identical with
  diagnostics disabled and enabled across actor, critic, target, temperature,
  optimizer states, RNGs, counters, and returned update metrics.
- Projection, norm-growth identity, Optax decay split, and exact scale/code
  additivity have dedicated nontrivial synthetic tests.
- GPU1 smoke `online_resident_mechanism_s0_20260902`: `cheetah-run`, width 512,
  batch 256, 200 env steps / 203 learner updates; `run_finished` present,
  training NaN/Inf counts `0/0`, and all 8 member-wise diagnostic rows finite.
  Early residual-kernel adaptive motion was mostly zero, so this smoke is an
  implementation check rather than mechanism evidence.
- `git diff --check` and `bash -n scripts/run_dogs_online_fp8_resident.sh`
  passed.
