# Offline FP8 target-state simulation

## Motivation

The Dogs C/D runs established a finite but severe learning failure when the four
target-Critic residual kernels are stored in E4M3 and requantized after every
`tau=0.005` EMA. A target-forward-only control with FP32 storage tracks the
healthy online-FP8 baseline, so the next question is not whether FP8 GEMMs work,
but which low-precision state update can follow the healthy FP32 target through
time. Closed-loop training would confound each candidate's target error with the
online trajectory it subsequently creates; this change therefore adds an
open-loop shared-teacher screen first.

## Changes

- Added pure E4M3 target-state operators for the four residual Dense kernels:
  per-ensemble lag coding, naive per-tensor EMA, naive dynamic block-scale EMA,
  and deterministic lattice-matched interleaved block EMA. The default block is
  `128x128`; codes remain E4M3 and scales/phases remain FP32.
- The interleaved method searches powers-of-two coefficients from `tau` through
  `1`, requires enough positive projection along the current online-target gap,
  minimizes projected update variance, and commits a block only when its
  globally staggered phase crosses one. A non-event leaves both codes and scale
  bitwise unchanged; there is no residual, Kahan state, stochastic rounding, or
  fallback.
- Added a standalone teacher learner update and a separate shadow JIT. Keeping
  the compiled teacher graph independent prevents shadow fusion from changing
  even the last bits of the healthy trajectory. Every shadow receives the same
  consecutive `online_old/online_new` pair and cannot feed back into the teacher.
- Added an offline CLI that restores the complete model/optimizer/RNG,
  reward-normalizer statistics, and replay buffer without creating an
  environment. It samples the checkpoint's original two-batch groups, writes
  JSON config/events/metrics/summary plus a fixed 256-sample probe, and never
  saves per-step parameter snapshots.
- Diagnostics compare each reconstructed shadow against the counterfactual FP32
  teacher in parameter and function space. They include cumulative displacement,
  direction, relative state error, code/block events, scale/phase/lag summaries,
  expected-Q error, 101-bin JS divergence, complete categorical Bellman-target
  error, per-task/member views, and the target FP8-Direct computation noise floor.

## Fixed input and protocol

The healthy target-forward-only Dogs seed-42 recovery at environment step 100k
(eval return `340.61`) was hard-linked to
`runs/offline_inputs/dogs_target_fwd_s42_step100000` before implementation. The
fixed input contains the FP8-Direct online Critic with FP32 parameters/Adam, the
FP8-Direct target forward with FP32 parameters/EMA, model RNGs, normalizer state,
and the complete replay buffer. Runtime data remains ignored by Git.

The registered full screen performs 50,000 learner updates as 25,000 restored
two-batch groups, with no environment interaction, W&B, return evaluation, or
performance claim. Diagnostics run every 500 updates on the same private probe
and diagnostic RNG. The 100-update Blackwell smoke and full screen are launched
only after this implementation commit is clean.

## Validation

- All 38 repository CPU tests pass. Eight new tests cover exact unquantized lag
  algebra, target-kernel scope/dtypes, independent ensemble scales, block
  quantization, bitwise unchanged non-events, interleaved phases/events, exact
  reproduction of the existing resident per-tensor EMA, finite reconstruction,
  and teacher invariance.
- The standalone teacher and teacher-plus-shadow paths are equal leaf by leaf for
  parameters, optimizer state, FP8 metadata, RNG, temperature, and update
  metrics. This is stronger than a tolerance comparison.
- A width-16 functional diagnostic compile returns finite aggregate, per-task,
  per-ensemble, parameter, Bellman-target, and FP8 compute-noise metrics for all
  four methods.
- Blackwell width-4096 smoke and the 50k result are pending at the time of this
  implementation commit; their concrete output IDs and outcomes will be added
  in the experiment-ledger follow-up commit.

## Limits

- This is an open-loop update-fidelity screen, not a new training precision mode
  and not evidence of closed-loop return recovery.
- Lag reconstruction uses a temporary FP32 `online + lag` only for diagnostics;
  it does not settle the eventual native-FP8 forward/cache design.
- Interleaved blocks intentionally trade per-block continuity for network-wide
  spatially dispersed changes. Its bootstrap behavior must still be tested after
  it passes the shared-trajectory screen.
- The methods use different scale geometries by design. Attribution of temporal
  scheduling comes from naive block-scale versus interleaved block, not from the
  failed per-tensor control directly.
