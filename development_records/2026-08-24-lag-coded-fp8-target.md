# Lag-Coded FP8 Target Critic

## Motivation

Naive E4M3 target kernels failed to track the online Critic under the original
`tau=0.005` per-step EMA: Dogs return stagnated even though target FP8 forward
with FP32 storage completed normally. Shared-teacher offline simulation showed
that storing `target - online` in E4M3 preserved the healthy FP32 target
trajectory far better than naive resident state or the FP8 Kahan baseline. This
change promotes that representation into the real BRC bootstrap loop without
changing the Bellman target.

## Changes

- Added `target_critic_precision=fp8_lag`. Only the four residual-Dense
  kernel leaves persist E4M3 lag codes; each ensemble member has an independent
  current-amax FP32 lag scale. Other target leaves retain FP32 EMA.
- Initialized lag to zero and reconstruct every target consumer through the
  same transient FP32 `online + dequantized_lag` tree. Training bootstrap,
  time-limit bootstrap, gradient diagnostics, tensor diagnostics, and the FP32
  forward reference therefore use one target definition.
- After each online Critic update, write
  `(1-tau) * (lag - (online_new-online_old))` back to E4M3. The online
  parameters, Adam state, online delta, and reconstruction remain FP32. There
  is no FP32 target-kernel master, Kahan state, random rounding, block scale, or
  derived target cache.
- Reused the existing target FP8 Direct forward. Only the training bootstrap
  advances input/kernel delayed-scaling metadata; read-only queries do not
  mutate it and no target backward is introduced.
- Added lag scale/amax/RMS, effective subnormal, underflow, quantization error,
  online-delta, intended/applied EMA, direction, gap, and same-weight FP32
  forward-error diagnostics. Same-mode lag checkpoint recovery is supported;
  precision conversion remains unsupported.

## Evaluation

The first closed-loop experiment is D-lag: online `fp8_direct`, target
`fp8_lag`, Dogs seed 42, width 4096, 500k steps. It is compared with the
online-FP8/target-FP32 baseline (`816.37`), target-FP8-forward/FP32-storage
control (`757.81`), and failed naive resident D-R1 (`7.23` at 125k).

## Pitfalls

- The persistent E4M3 object is the lag, not a target kernel. The target GEMM
  currently reconstructs FP32 `online + lag` and then quantizes its RHS, so
  this mode does not remove repeated RHS conversion.
- The method relies on a FP32 online anchor. It does not solve future
  persistent-FP8 online parameters or Adam updates.
- Per-tensor lag scaling protects the lag's own dynamic range but does not prove
  closed-loop bootstrap stability. The 200-step smoke is implementation
  evidence only; return requires the full environment run.

## Validation and limits

- CPU: all 43 repository tests passed. Added coverage verifies the four E4M3
  lag leaves, exact initial reconstruction, two consecutive old/new online
  recurrences, synthetic underflow, target metadata mutability boundaries,
  same-mode checkpoint replay, and history-length rejection.
- Blackwell GPU1 / CUDA 12.8: optimized target-only HLO exposes four E4M3 lag
  parameters and four FP32 online kernels. Each RHS path performs
  `E4M3 lag -> FP32 * lag_scale + FP32 online -> E4M3`, followed by exactly
  four `__cublas$lt$matmul$f8` residual GEMMs with FP32 output. Input/head
  remain FP32 and the target HLO contains no E5M2 path.
- GPU1 smoke: `cheetah-run`, width 512, batch 256, 200 env steps and 203
  learner updates completed with `run_finished`. All sampled loss/gradient/Q
  values and all 498 tensor-stat rows were finite; update NaN/Inf was `0/0`.
  Across eight physical kernels, lag-candidate underflow was
  `1.14e-5–5.39e-5`, applied/intended EMA L2 ratio
  `0.999993–1.000002`, relative update error
  `1.09e-4–1.55e-4`, and cosine approximately one.
- No throughput conclusion is drawn: this implementation intentionally favors
  a clean numerical test and currently reconstructs/requantizes every target
  RHS.
