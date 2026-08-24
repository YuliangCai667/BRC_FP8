# FP8 Kahan-momentum target-state baseline

## Motivation

The 50k shared-teacher screen showed that lag-coded target state closely tracks
the healthy FP32 EMA, while naive resident target state and the first
interleaved-block design drift substantially. The FP16 SAC paper specifically
uses Kahan-momentum to protect small target-network soft updates, so a direct
FP8 adaptation is needed as a prior-art baseline before attributing the gain to
lag coding.

## Changes

- Added a fifth offline shadow, `kahan_momentum`, over the same four target
  residual-Dense kernels and the same teacher `online_old/online_new` sequence.
- Kept the paper's `C=1e4` scaled target buffer and Kahan update ordering.
  The scaled target and compensation are separate persistent E4M3 tensors with
  independent current-amax FP32 scales per ensemble member. There is no FP32
  target master and the shadow never feeds back into the teacher.
- Extended reconstruction, parameter/functional diagnostics, run metadata,
  tests, README, idea record, experiment ledger, and research index.

## Evaluation

The main comparison is `kahan_momentum ↔ lag_coded` on
`EXP-FP8-TARGET-KAHAN-OFFLINE-30K`, restored from the fixed healthy Dogs
seed-42 100k recovery. It performs 30,000 learner updates with diagnostics every
500 updates. The original naive and block shadows remain present as controls.

## Pitfalls

- Under current-amax dynamic scaling, multiplying a tensor by `C` also
  multiplies its scale, so `C=1e4` alone does not add E4M3 mantissa resolution.
  Any improvement over naive per-tensor EMA must come from the compensation
  buffer. This differs from the fixed-range FP16 setting.
- Kahan persists two FP8 matrix states per kernel (scaled target plus
  compensation); lag-coded persists one. Accuracy comparisons must report this
  storage difference.
- This is an open-loop mechanism screen. It measures target-state tracking and
  bootstrap-function error, not closed-loop return or throughput.

## Validation and limits

- `JAX_PLATFORMS=cpu python -m unittest tests.test_target_simulation`: 10/10.
- `JAX_PLATFORMS=cpu python -m unittest discover -s tests`: 40/40.
- GPU1 smoke: 100 updates, two diagnostics, complete output set, all
  teacher/shadow metrics finite, NaN/Inf `0/0`.
- GPU1 30k screen: completed in 1,316.8 seconds with 60 diagnostics and no
  NaN/Inf. Lag/Kahan endpoint target relative error was
  `0.007725/0.258575`, displacement cosine `0.999540/0.030080`, and
  expected-Q MAE `0.001216/0.141415`. Kahan matched naive per-tensor EMA to
  within `6.26e-7` target relative error across the window despite nonzero
  compensation.
- The run used an uncommitted worktree at base commit `50814df`; the
  experiment ledger records that state. No closed-loop Kahan precision mode or
  formal environment run was added.
