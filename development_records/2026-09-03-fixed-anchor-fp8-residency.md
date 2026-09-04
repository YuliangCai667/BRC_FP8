# Fixed-Anchor FP8 Residency

## Motivation

The previous canonicalization recursively used the preceding physical kernel
norm. Its two formal Dogs runs reproduced scale-dominated norm growth and were
stopped, so the moving anchor is an implementation failure rather than a result
about fixed-anchor canonicalization.

## Changes

- Each of the four online resident kernels now stores one initialization norm
  per ensemble member as `kernel_anchor_norm`: eight persistent FP32 scalars.
- An AdamW candidate is quantized once with the existing current-amax E4M3
  quantizer. The selected code is retained, while the stored scale is computed
  directly as `rho0 / ||code.astype(float32)||_F`; the paired bias is multiplied
  by `fixed_scale / raw_scale`.
- The write path no longer reads the previous physical kernel norm. AdamW
  moments, target critic, non-resident parameters, and the disabled path remain
  unchanged.
- Anchor metadata is included automatically in analysis/recovery model state.
  Every saved checkpoint containing anchors also writes
  `fixed_anchor_norms.json`, computed by loading the serialized critic and using
  NumPy float64 for `scale * ||code|| / rho0`.
- Resume validation distinguishes the fixed-anchor resolved method from legacy
  moving-anchor checkpoints. The canonical run entry point is
  `scripts/run_dogs_online_fp8_resident_fixed_anchor.sh`; the independent
  inspector is `scripts/check_fp8_fixed_anchor.py`.

## Evaluation

The first gate is a seed-42 Dogs run to 25k on GPU3. All four kernels/eight
members must be present in the final checkpoint report and show stable
`anchor_norm_ratio`; only then is the same run resumed to 500k. Long-horizon
return remains a later outcome, not part of the mechanism gate.

## Pitfalls

- The flag retains its existing public name, but its only enabled implementation
  is now fixed-initial-anchor. `resolved_online_fp8_canonicalization` records the
  exact semantics.
- JAX FP32 per-write ratios are useful telemetry but are not the sole authority;
  checkpoint acceptance uses the independently deserialized NumPy float64
  report.
- The 25k→500k stage is a recovery resume of the same run, not a fresh seed-42
  restart.

## Validation and limits

- `git diff --check` and Python syntax checks pass.
- Complete CPU FP8/checkpoint suite: 29 tests passed in 160.334s, including a
  10,000-write fixed-code invariant test and fixed-anchor checkpoint round-trip.
- On GPU3, the 10,000-write test and a real resident update passed. A combined
  checkpoint smoke observed a one-ULP (`2.38e-7`) difference in a separately
  recomputed loss after restore; serialized state equality and the strict CPU
  next-update test pass.
- The width-4096 seed42 run completed 25k / update 40003. Independent NumPy
  float64 reports from both analysis and recovery checkpoints agree: all eight
  ratios are `0.9999168–1.0002157` with maximum deviation `2.157e-4`; 25k
  critic pnorm is `359.29`. The mechanism gate passed, allowing recovery resume
  to 500k.

## 425k reward-lag diagnosis

- Observed: the resumed run reached eval return `572.84` at 400k and `499.04`
  at 425k, versus `766.69/764.57` for the matched FP8-direct online,
  FP32-storage target-forward control. Fixed anchor therefore permits learning
  but does not recover the expected curve.
- Observed: the 400k serialized NumPy float64 anchor ratios are
  `0.9996764–1.0035631`, and retained checkpoints fluctuate rather than drift
  monotonically. At 425k the combined resident-kernel norm remains `255.995`;
  rising total critic pnorm comes predominantly from non-resident FP32 kernels.
- Observed: all `68/68` recorded fixed-anchor resident-kernel gradients are
  exactly zero. The completed naive-resident seed42 run has the same `80/80`
  pattern, while all `80/80` corresponding gradients in the matched
  FP8-direct control are nonzero. Resident Dense bias gradients on the same
  batches remain nonzero.
- Root cause: `ResidentFp8Dense` uses autodiff through a raw FP8
  `lax.dot_general` and does not implement the scaled custom backward used by
  Flax FP8 Direct. Its output cotangent is converted to FP8 before reciprocal
  scale recovery and underflows. A GPU isolation probe with representative
  `activation_scale=0.5` and `kernel_scale=2e-4` produces a 100%-zero resident
  kernel gradient for output-gradient sigma `1e-3`, while the equivalent FP32
  gradient has L2 `151.17`.
- Interpretation: anchor canonicalization itself passed its norm invariant.
  The reward result rejects norm runaway as the primary return-gap cause in
  the current implementation; the online-resident backward path must be made
  scale-aware before any further return experiment can evaluate fixed-anchor
  canonicalization. Candidate-to-write expected-Q metrics do not cover this
  pre-candidate backward failure.
